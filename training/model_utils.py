from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rl.action_space import stage_action_mask

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover - handled at runtime by requirements.txt
    MaskablePPO = None  # type: ignore[assignment]


def preferred_algorithm_name() -> str:
    return "MaskablePPO"


def require_maskable_ppo():
    if MaskablePPO is None:
        raise ImportError("Stage-aware action masking requires sb3-contrib. Install dependencies with: pip install -r requirements.txt")
    return MaskablePPO


def make_model(env: Any, params: dict[str, Any]) -> Any:
    model_cls = require_maskable_ppo()
    return model_cls(policy="MlpPolicy", env=env, **params)


def load_preferred_model(path: str | Path, env: Any | None = None, device: str = "cpu") -> Any:
    model_cls = require_maskable_ppo()
    return model_cls.load(str(path), env=env, device=device)


def load_any_sb3_model(path: str | Path, env: Any | None = None, device: str = "cpu") -> Any:
    errors: list[str] = []
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(str(path), env=env, device=device)
        except Exception as exc:
            errors.append(f"MaskablePPO: {exc}")
    try:
        return PPO.load(str(path), env=env, device=device)
    except Exception as exc:
        errors.append(f"PPO: {exc}")
    raise RuntimeError(f"Could not load SB3 model from {path}. " + " | ".join(errors))


def predict_with_action_masks(model: Any, obs: np.ndarray, env_or_stage: Any, deterministic: bool = True) -> tuple[np.ndarray, Any]:
    _assert_model_compatible_observation(model, obs, env_or_stage)

    masks = None
    if isinstance(env_or_stage, str):
        masks = stage_action_mask(env_or_stage)
    elif hasattr(env_or_stage, "action_masks"):
        masks = env_or_stage.action_masks()

    if masks is not None:
        try:
            # MaskablePPO consumes this argument; PPO falls through to TypeError.
            return model.predict(obs, deterministic=deterministic, action_masks=masks)
        except TypeError:
            pass
    return model.predict(obs, deterministic=deterministic)


def _assert_model_compatible_observation(model: Any, obs: np.ndarray, env_or_stage: Any) -> None:
    model_space = getattr(model, "observation_space", None)
    expected_shape = getattr(model_space, "shape", None)
    if expected_shape is None:
        return

    actual_shape = tuple(np.asarray(obs).shape)
    expected_shape = tuple(expected_shape)
    if actual_shape == expected_shape:
        return
    if len(actual_shape) >= 2 and actual_shape[1:] == expected_shape:
        return

    stage = env_or_stage if isinstance(env_or_stage, str) else getattr(env_or_stage, "stage_id", "unknown")
    observation_schema = getattr(env_or_stage, "observation_stage_id", stage)
    raise RuntimeError(
        "Observation shape mismatch before model.predict: "
        f"stage={stage}, observation_schema={observation_schema}, "
        f"env_obs_shape={actual_shape}, model_expected_shape={expected_shape}. "
        "Use a model-compatible observation schema for previous-skill retention "
        "instead of evaluating a later-stage model on an old compact observation vector."
    )
