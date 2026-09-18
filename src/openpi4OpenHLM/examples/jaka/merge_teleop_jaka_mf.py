"""Merge the `teleop_jaka_mf` recording sessions into one LeRobot dataset,
and reorder the `state` / `actions` dimensions.

Each session under `data/teleop_jaka_mf/<timestamp>/level-0` is already a
standalone LeRobot v2.1 dataset. This script concatenates them into a single
dataset whose `repo_id` matches the `jaka_tabletop_pick` TrainConfig, so the
training pipeline can load it via `HF_LEROBOT_HOME` without further changes.

This is a direct pyarrow rewrite: the `head_image_left` column stores PNG bytes
inline (`struct<bytes, path>`), so it is copied through byte-for-byte and never
decoded. Only `state` / `actions` are touched, plus the bookkeeping columns
(`index`, `episode_index`, `task_index`). `meta/info.json`, `meta/tasks.jsonl`,
`meta/episodes.jsonl` and `meta/episodes_stats.jsonl` are written by this script,
following the lerobot v2.1 conventions.

The reordering is plain slicing, same style as
`examples/unitree_g1/convert_g1_data_to_lerobot.py`. Note that the Jaka
recordings put `root` LAST (G1's `state_body` has it first), and that the
30 recorded dims are shared by `state` and `actions`:

    state   (30) = [leg_left 6, leg_right 6, waist 1, arm_left 6, arm_right 6,
                    neck 2, root 3]
    actions (40) = the same 30 + [anchor_lin_vel 3, anchor_pos_w 3, anchor_quat_w 4]

`reorder()` reorders the 30 shared dims to the saved order and appends the 10
anchor action dims untouched.

Run inside the `xujinfan_dev` container:

    /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
        /workspace/OpenHLM/src/openpi4OpenHLM/examples/jaka/merge_teleop_jaka_mf.py --dry-run

then drop `--dry-run` to write the merged dataset.
"""

import dataclasses
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG -- edit this block
# ---------------------------------------------------------------------------

DEFAULT_HF_HOME = "/workspace/OpenHLM/data"

# Source datasets. Each entry is the dataset ROOT itself (the `level-0` dir),
# and all of them must share the same feature names. Episodes are merged in the
# order listed here, then by ascending episode_index within a session.
DEFAULT_SOURCE_DIRS = [
    "/workspace/OpenHLM/data/teleop_jaka_mf/20260915-142439/level-0",  # 3 episodes
    "/workspace/OpenHLM/data/teleop_jaka_mf/20260915-143308/level-0",  # 5 episodes
    "/workspace/OpenHLM/data/teleop_jaka_mf/20260915-145329/level-0",  # 39 episodes
]

# Output dataset. `OUTPUT_REPO_ID` must match `LeRobotJakaDataConfig.repo_id`
# in src/openpi/training/config.py, and the dataset lands at
# `$HF_LEROBOT_HOME / OUTPUT_REPO_ID`.
DEFAULT_OUTPUT_REPO_ID = "teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0"

# Metadata only -- the training pipeline never reads `robot_type`; the sources
# leave it null.
DEFAULT_ROBOT_TYPE = "jaka"

# --- joint groups, as recorded ON DISK (MuJoCo joint order) ----------------

LEG_LEFT_JOINTS = [
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
]
LEG_RIGHT_JOINTS = [
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint"]
ARM_LEFT_JOINTS = [
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
]
ARM_RIGHT_JOINTS = [
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
]
NECK_JOINTS = ["Neck_yaw_joint", "Neck_pitch_joint"]
# `yaw_vel` is an angular velocity; `root_roll` / `root_pitch` are absolute.
ROOT_DIMS = ["root_roll", "root_pitch", "yaw_vel"]

# The 10 trailing action dims, kept as-is after the reordered 30.
ACTION_TAIL_DIMS = [
    "anchor_lin_vel_x", "anchor_lin_vel_y", "anchor_lin_vel_z",
    "anchor_pos_w_x", "anchor_pos_w_y", "anchor_pos_w_z",
    "anchor_quat_w_w", "anchor_quat_w_x", "anchor_quat_w_y", "anchor_quat_w_z",
]

# What the recordings actually contain, in order. Checked against
# `meta/info.json` so a differently-exported session fails loudly instead of
# silently mislabelling dims. Matches the slices in `reorder()`.
SOURCE_STATE_NAMES = (
    LEG_LEFT_JOINTS + LEG_RIGHT_JOINTS + WAIST_JOINTS
    + ARM_LEFT_JOINTS + ARM_RIGHT_JOINTS + NECK_JOINTS + ROOT_DIMS
)

# What we SAVE (the same 30 names, resliced). Currently the G1/OpenPI
# convention -- arms first -- so this dataset lines up with the G1 layout.
SAVED_STATE_NAMES = (
    ARM_LEFT_JOINTS + ARM_RIGHT_JOINTS + LEG_LEFT_JOINTS + LEG_RIGHT_JOINTS
    + WAIST_JOINTS + NECK_JOINTS + ROOT_DIMS
)

# ---------------------------------------------------------------------------


def load_info(session_dir: Path) -> dict:
    info_path = session_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"'{session_dir}' is not a LeRobot dataset: missing {info_path}")
    with info_path.open() as f:
        return json.load(f)


def read_jsonlines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonlines(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_sources(source_dirs: list[Path]) -> tuple[dict, list[str]]:
    """Check every source agrees on layout, and return (ref_info, action_tail_names)."""
    infos = [load_info(d) for d in source_dirs]
    ref = infos[0]

    problems = []
    if list(ref["features"]["state"]["names"]) != SOURCE_STATE_NAMES:
        problems.append(
            "SOURCE_STATE_NAMES does not describe the dataset on disk.\n"
            f"      expected: {SOURCE_STATE_NAMES}\n"
            f"      on disk : {list(ref['features']['state']['names'])}"
        )
    if list(ref["features"]["actions"]["names"]) != SOURCE_STATE_NAMES + ACTION_TAIL_DIMS:
        problems.append(
            "action names are not SOURCE_STATE_NAMES + ACTION_TAIL_DIMS.\n"
            f"      on disk : {list(ref['features']['actions']['names'])}"
        )
    if problems:
        raise SystemExit(f"Source '{source_dirs[0]}' is incompatible:\n  - " + "\n  - ".join(problems))

    state_names = list(ref["features"]["state"]["names"])
    action_names = list(ref["features"]["actions"]["names"])
    for path, info in zip(source_dirs[1:], infos[1:]):
        problems = []
        if info["codebase_version"] != ref["codebase_version"]:
            problems.append(f"codebase_version '{info['codebase_version']}' != '{ref['codebase_version']}'")
        if info["fps"] != ref["fps"]:
            problems.append(f"fps {info['fps']} != {ref['fps']}")
        if list(info["features"]["state"]["names"]) != state_names:
            problems.append("state names differ from the reference dataset")
        if list(info["features"]["actions"]["names"]) != action_names:
            problems.append("action names differ from the reference dataset")
        if problems:
            raise SystemExit(f"Source '{path}' is incompatible:\n  - " + "\n  - ".join(problems))

    return ref, action_names[len(state_names) :]


def reorder(state: np.ndarray, actions: np.ndarray, identity: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reorder one episode's state/actions from the recorded order to the saved order.

    state   is (num_frames, 30)
    actions is (num_frames, 40) -- the same 30 dims followed by the 10 anchor dims
    """
    if identity:
        return state, actions

    # state/actions layout on disk (30 dims), MuJoCo joint order:
    #   [0:6]   leg_left:   hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    #   [6:12]  leg_right:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    #   [12:13] waist:      waist_yaw
    #   [13:19] arm_left:   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw
    #   [19:25] arm_right:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw
    #   [25:27] neck:       neck_yaw, neck_pitch
    #   [27:30] root:       root_roll, root_pitch, yaw_vel
    state_leg_left = state[:, 0:6]
    state_leg_right = state[:, 6:12]
    state_waist = state[:, 12:13]
    state_arm_left = state[:, 13:19]
    state_arm_right = state[:, 19:25]
    state_neck = state[:, 25:27]
    state_root = state[:, 27:30]

    action_leg_left = actions[:, 0:6]
    action_leg_right = actions[:, 6:12]
    action_waist = actions[:, 12:13]
    action_arm_left = actions[:, 13:19]
    action_arm_right = actions[:, 19:25]
    action_neck = actions[:, 25:27]
    action_root = actions[:, 27:30]

    # Assemble the saved order (30 dims):
    #   arm_left(6), arm_right(6), leg_left(6), leg_right(6), waist(1), neck(2), root(3)
    new_state = np.concatenate([
        state_arm_left,
        state_arm_right,
        state_leg_left,
        state_leg_right,
        state_waist,
        state_neck,
        state_root,
    ], axis=1)

    # Actions are those same 30 dims, then the 10 anchor dims appended untouched.
    new_actions = np.concatenate([
        action_arm_left,
        action_arm_right,
        action_leg_left,
        action_leg_right,
        action_waist,
        action_neck,
        action_root,
        actions[:, 30:],
    ], axis=1)

    return new_state, new_actions


# ---------------------------------------------------------------------------
# stats -- mirrors lerobot's compute_stats.py so meta/episodes_stats.jsonl is
# compatible with what LeRobotDataset.create would have produced.
# ---------------------------------------------------------------------------


def estimate_num_samples(dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def image_stats(image_bytes: list[bytes]) -> dict:
    """Stats over a sample of the episode's frames, in lerobot's format.

    Returns min/max/mean/std of shape (3, 1, 1) in [0, 1] plus a `count`.
    """
    indices = sample_indices(len(image_bytes))
    sampled = None
    for i, idx in enumerate(indices):
        with Image.open(io.BytesIO(image_bytes[idx])) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)  # HWC, 224x224 so no downsampling
        chw = np.transpose(arr, (2, 0, 1))
        if sampled is None:
            sampled = np.empty((len(indices), *chw.shape), dtype=np.uint8)
        sampled[i] = chw

    axes = (0, 2, 3)  # keep the channel dim
    stats = {
        "min": np.min(sampled, axis=axes, keepdims=True),
        "max": np.max(sampled, axis=axes, keepdims=True),
        "mean": np.mean(sampled, axis=axes, keepdims=True),
        "std": np.std(sampled, axis=axes, keepdims=True),
        "count": np.array([len(sampled)]),
    }
    return {
        key: (value if key == "count" else np.squeeze(value / 255.0, axis=0))
        for key, value in stats.items()
    }


def feature_stats(values: np.ndarray) -> dict:
    """Stats over axis 0 of an (N, D) array."""
    return {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([len(values)]),
    }


def to_jsonable(stats: dict) -> dict:
    return {key: value.tolist() for key, value in stats.items()}


# ---------------------------------------------------------------------------


def list_column(values: np.ndarray, like: pa.DataType) -> pa.Array:
    """Build a list column for an (N, D) array, keeping the source column's type."""
    n, dim = values.shape
    flat = pa.array(values.reshape(-1))
    if pa.types.is_fixed_size_list(like):
        return pa.FixedSizeListArray.from_arrays(flat, dim)
    offsets = pa.array(np.arange(0, n * dim + 1, dim, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


def rewrite_episode(table: pa.Table, out_episode: int, frame_offset: int, task_index: int) -> pa.Table:
    """Renumber the bookkeeping columns of one episode (index / episode_index / task_index)."""
    schema = table.schema
    num_frames = table.num_rows
    for name, values in (
        ("index", np.arange(frame_offset, frame_offset + num_frames, dtype=np.int64)),
        ("episode_index", np.full(num_frames, out_episode, dtype=np.int64)),
        ("task_index", np.full(num_frames, task_index, dtype=np.int64)),
    ):
        field = schema.field(name)
        table = table.set_column(schema.get_field_index(name), field, pa.array(values, type=field.type))
    return table


def write_meta(
    out_path: Path,
    ref_info: dict,
    state_names: list[str],
    action_names: list[str],
    robot_type: str,
    episodes: list[dict],
    tasks: list[str],
    episodes_stats: list[dict],
) -> None:
    meta_dir = out_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads(json.dumps(ref_info))  # deep copy
    chunks_size = info["chunks_size"]
    total_frames = sum(ep["length"] for ep in episodes)
    info.update(
        {
            "robot_type": robot_type,
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_videos": 0,
            "total_chunks": (len(episodes) + chunks_size - 1) // chunks_size if episodes else 0,
            "splits": {"train": f"0:{len(episodes)}"},
            "video_path": None,
        }
    )
    info["features"]["state"]["names"] = state_names
    info["features"]["actions"]["names"] = action_names

    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    write_jsonlines([{"task_index": i, "task": t} for i, t in enumerate(tasks)], meta_dir / "tasks.jsonl")
    write_jsonlines(episodes, meta_dir / "episodes.jsonl")
    write_jsonlines(episodes_stats, meta_dir / "episodes_stats.jsonl")


@dataclasses.dataclass
class Args:
    source_dirs: list[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_SOURCE_DIRS))
    output_repo_id: str = DEFAULT_OUTPUT_REPO_ID
    hf_home: str = DEFAULT_HF_HOME
    robot_type: str = DEFAULT_ROBOT_TYPE
    max_episodes_per_session: int | None = None
    identity: bool = False
    """Keep the recorded MuJoCo order instead of reordering. Useful for diffing against the sources."""
    overwrite: bool = False
    dry_run: bool = False


def main(args: Args) -> None:
    hf_home = Path(args.hf_home)
    source_dirs = [Path(d) for d in args.source_dirs]
    out_path = hf_home / args.output_repo_id

    for session_dir in source_dirs:
        if not session_dir.is_dir():
            raise SystemExit(f"Source dataset not found: {session_dir}")

    ref_info, tail_names = validate_sources(source_dirs)
    if tail_names != ACTION_TAIL_DIMS:
        raise SystemExit(f"Action tail dims are unexpected.\n  on disk: {tail_names}\n  expected: {ACTION_TAIL_DIMS}")
    state_names = SOURCE_STATE_NAMES if args.identity else SAVED_STATE_NAMES
    action_names = state_names + tail_names
    chunks_size = ref_info["chunks_size"]

    print(f"Sources     : {len(source_dirs)}")
    for session_dir in source_dirs:
        info = load_info(session_dir)
        print(f"  {session_dir}  ({info['total_episodes']} episodes, {info['total_frames']} frames)")
    print(f"Output      : {out_path}")
    print(f"State dims  : {len(state_names)}")
    print(f"Action dims : {len(action_names)}  (last {len(tail_names)} are anchor dims, never reordered)")
    print("\nSaved state order:")
    for i, name in enumerate(state_names):
        origin = SOURCE_STATE_NAMES.index(name)
        mark = "  (unchanged)" if origin == i else ""
        print(f"  [{i:2d}] {name:<32} <- recorded [{origin:2d}]{mark}")

    if args.dry_run:
        print("\n[dry-run] nothing was written.")
        return

    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {out_path}\nRe-run with --overwrite to delete and recreate it."
            )
        if out_path.resolve() in {d.resolve() for d in source_dirs}:
            raise SystemExit(f"Refusing to delete the output path because it is one of the sources: {out_path}")
        print(f"\nRemoving existing output at {out_path}")
        shutil.rmtree(out_path)
    (out_path / "meta").mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    episodes_stats: list[dict] = []
    tasks: list[str] = []
    frame_offset = 0

    for session_dir in source_dirs:
        data_dir = session_dir / "data"
        files = sorted(data_dir.rglob("*.parquet"))
        if not files:
            raise SystemExit(f"No parquet files under {data_dir}")

        task_names = [
            t["task"]
            for t in sorted(read_jsonlines(session_dir / "meta" / "tasks.jsonl"), key=lambda x: x["task_index"])
        ]
        max_episodes = len(files)
        if args.max_episodes_per_session is not None:
            max_episodes = min(max_episodes, args.max_episodes_per_session)

        for src_file in tqdm.tqdm(files[:max_episodes], desc=session_dir.parent.name):
            out_episode = len(episodes)
            task = task_names[0] if task_names else "..."
            if task not in tasks:
                tasks.append(task)

            table = pq.read_table(src_file)
            metadata = table.schema.metadata
            state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
            actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
            new_state, new_actions = reorder(state, actions, args.identity)

            for name, values in (("state", new_state), ("actions", new_actions)):
                field = table.schema.field(name)
                table = table.set_column(table.schema.get_field_index(name), field, list_column(values, field.type))
            table = rewrite_episode(table, out_episode, frame_offset, tasks.index(task))
            # Keep the source's `huggingface` schema metadata (it records only
            # dtypes/shapes, not joint names, so it stays valid).
            table = table.replace_schema_metadata(metadata)

            chunk = out_episode // chunks_size
            out_file = out_path / "data" / f"chunk-{chunk:03d}" / f"episode_{out_episode:06d}.parquet"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_file)

            episodes.append({"episode_index": out_episode, "tasks": [task], "length": table.num_rows})
            episodes_stats.append(
                {
                    "episode_index": out_episode,
                    "stats": {
                        "state": to_jsonable(feature_stats(new_state)),
                        "actions": to_jsonable(feature_stats(new_actions)),
                        "head_image_left": to_jsonable(
                            image_stats([cell["bytes"].as_py() for cell in table.column("head_image_left")])
                        ),
                    },
                }
            )
            frame_offset += table.num_rows

    write_meta(
        out_path, ref_info, state_names, action_names, args.robot_type, episodes, tasks, episodes_stats
    )

    print(f"\nWrote {len(episodes)} episodes / {frame_offset} frames to {out_path}")
    print(f"Load it with HF_LEROBOT_HOME={hf_home} and repo_id='{args.output_repo_id}'.")


if __name__ == "__main__":
    main(tyro.cli(Args))
