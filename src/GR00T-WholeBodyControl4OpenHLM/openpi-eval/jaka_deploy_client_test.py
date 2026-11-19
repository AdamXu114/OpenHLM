"""JAKA rollout checks at the policy-client boundary."""

from unittest.mock import Mock

import numpy as np
import pytest

import main as client


@pytest.mark.parametrize("invalid", ["wrong_width", "short", "nonfinite"])
def test_bad_chunk_cannot_partially_publish(monkeypatch, invalid):
    good = np.zeros((50, 33), dtype=np.float32)
    bad = good.copy()
    if invalid == "wrong_width":
        bad = bad[:, :-1]
    elif invalid == "short":
        bad = bad[:24]
    else:
        bad[24, 32] = np.nan
    env = Mock()
    observation = {"state": np.zeros(30, dtype=np.float32),
                   "head_image_left": np.zeros((16, 16, 3), dtype=np.uint8)}
    env.reset.return_value = env.get_observation.return_value = observation
    make_env = Mock(return_value=env)
    policy = Mock()
    policy.infer.side_effect = [{"actions": good}] * 10 + [{"actions": bad}]
    monkeypatch.setattr(client, "JakaTabletopEnv", make_env)
    monkeypatch.setattr(client, "FakePolicyClient", Mock(return_value=policy))
    monkeypatch.setattr("builtins.input", lambda *args: "s")

    with pytest.raises(ValueError, match="JAKA"):
        client.main(client.Args(env="jaka_tabletop_pick", mock=True, use_fake_policy=True,
                                max_steps=1, opencv_visualize=False))
    assert make_env.call_args.kwargs["motion_zmq_address"] == "127.0.0.1"
    env.step.assert_not_called()
    env.close.assert_called_once()
