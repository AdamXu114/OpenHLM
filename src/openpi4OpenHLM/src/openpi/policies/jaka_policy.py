"""
Policy transforms for the Jaka whole-body robot dataset (OpenHLM format).

This module provides input and output transforms for the Jaka teleop dataset
(see ``simple/cli/teleop_jaka_mf.py`` for the data collection code).

The recordings are exported in MuJoCo joint order and are reordered into the
G1/OpenPI convention by ``examples/jaka/merge_teleop_jaka_mf.py``. Layouts after
that reordering -- arms first, root last (G1's layout has root first instead):

    state (30 dims), and action dims 0-29:
      0-11 : arms  -- left shoulder pitch/roll/yaw, elbow, wrist roll/yaw (6),
                      right arm's same six joints (6)
      12-23: legs  -- left hip pitch/roll/yaw, knee, ankle pitch/roll (6),
                      right leg's same six joints (6)
      24   : waist -- waist yaw
      25-26: neck  -- neck yaw, neck pitch
      27-29: root  -- roll, pitch (absolute), yaw_vel (an angular velocity)

    action dims 30-39: the recorded anchor dims, appended untouched by the merge
      30-32: anchor_lin_vel (x, y, z)
      33-35: anchor_pos_w   (x, y, z)
      36-39: anchor_quat_w  (w, x, y, z)

Training uses only the first 33 action dims -- the 30 shared dims plus
``anchor_lin_vel``; dims 33-39 are dropped by ``JakaInputs``. That matches
``action_dim=33`` in the ``jaka_tabletop_pick`` config and mirrors what
``JakaOutputs`` returns at inference time.

Actions are the absolute reference joint targets of the pico motion stream
(not deltas); the config applies DeltaActions/AbsoluteActions to convert them
relative to the current state for the model.

Only a single head camera is recorded. The model expects three views
(base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb), so the two wrist views are
filled with black placeholders and masked out.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Number of action dims the model is trained on. The dataset stores 40 dims; the
# last 7 (anchor_pos_w + anchor_quat_w) are dropped. Must match `action_dim` in
# the `jaka_tabletop_pick` config.
ACTION_DIM = 33


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class JakaInputs(transforms.DataTransformFn):
    """
    Transform inputs to the model format for the Jaka robot.

    Expected inputs:
    - head_image_left: Head-mounted camera image [height, width, channel] or
      [channel, height, width]. This is the only recorded camera.
    - state: Robot state [30] = 27 joints + root [roll, pitch, yaw_vel].
    - actions: Action sequence [action_horizon, 40] (only during training).
      Only the first ACTION_DIM (33) dims are kept.
    - prompt: Language instruction string.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse image to uint8 (H, W, C) format since LeRobot automatically
        # stores as float32 (C, H, W), gets skipped for policy inference.
        base_image = _parse_image(data["head_image_left"])

        # No wrist cameras are recorded: fill with black placeholders and mask
        # them out so the model ignores them entirely.
        zeros = np.zeros_like(base_image)

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Actions are only available during training. The dataset stores 40 dims;
        # keep the first ACTION_DIM so they match the model's action_dim, since
        # PadStatesAndActions only pads and never truncates.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])[..., :ACTION_DIM]

        # Pass the prompt (language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Forward last_action from previous inference chunk if provided.
        # Used by AbsoluteActions(use_first_action=True) to recover absolute actions.
        if "observation/last_action" in data:
            inputs["last_action"] = np.asarray(data["observation/last_action"])

        return inputs


@dataclasses.dataclass(frozen=True)
class JakaOutputs(transforms.DataTransformFn):
    """
    Transform outputs from the model back to the dataset format for the Jaka robot.

    This class is used for inference only. It extracts the relevant action dimensions
    from the model output, since the model may output padded actions.

    The dataset stores 40 action dims, of which the model is trained on the first
    ACTION_DIM (33) -- the 30 shared joint/rpy dims plus `anchor_lin_vel`.
    """

    def __call__(self, data: dict) -> dict:
        # Return the first ACTION_DIM dims since the model may output padded actions.
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
