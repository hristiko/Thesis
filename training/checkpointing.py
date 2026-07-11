from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.stages import GLOBAL_LATEST_TRAINING_STATE, get_stage


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def capture_rng_states(env: Any | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    if env is not None:
        # Gymnasium's generator object is not always available through SB3's
        # vector wrappers, so store what we can without making resume brittle.
        try:
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "np_random"):
                state["env_np_random_state"] = unwrapped.np_random.bit_generator.state
        except Exception:
            state["env_np_random_state"] = None
    return state


def restore_rng_states(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python_random") is not None:
        random.setstate(state["python_random"])
    if state.get("numpy_random") is not None:
        np.random.set_state(state["numpy_random"])
    if state.get("torch_random") is not None:
        torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and state.get("torch_cuda_random") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def save_json_snapshot(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(data), file, indent=2)


def save_training_checkpoint(
    *,
    model: Any,
    stage_id: str,
    skill_table: dict[str, Any],
    callback_state: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: Path,
    model_path: Path,
    env: Any | None = None,
    reason: str = "periodic",
) -> Path:
    """Save MaskablePPO/PPO model plus resume metadata.

    SB3's MaskablePPO/PPO .zip contains policy/value-network weights and optimizer state.
    The .pt file stores curriculum/training metadata, counters, RNG state, and
    the path to that model file.
    """
    checkpoint_path = Path(checkpoint_path)
    model_path = Path(model_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(model_path))

    rollout_buffer = getattr(model, "rollout_buffer", None)
    rollout_progress = {
        "buffer_pos": getattr(rollout_buffer, "pos", None),
        "buffer_full": getattr(rollout_buffer, "full", None),
        "note": "MaskablePPO/PPO resume restarts rollout collection from the next learn() call; completed model/optimizer state is restored.",
    }

    payload = {
        "format": "hk24_ppo_training_checkpoint_v1",
        "stage_id": stage_id,
        "model_path": str(model_path),
        "num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "skill_table": skill_table,
        "callback_state": callback_state,
        "rng_states": capture_rng_states(env),
        "rollout_progress": rollout_progress,
        "config": config,
        "reason": reason,
    }
    torch.save(payload, checkpoint_path)

    # Also keep a convenient global latest.pt for the requested CLI pattern.
    GLOBAL_LATEST_TRAINING_STATE.parent.mkdir(parents=True, exist_ok=True)
    try:
        if checkpoint_path.resolve() != GLOBAL_LATEST_TRAINING_STATE.resolve():
            shutil.copy2(checkpoint_path, GLOBAL_LATEST_TRAINING_STATE)
    except FileNotFoundError:
        shutil.copy2(checkpoint_path, GLOBAL_LATEST_TRAINING_STATE)

    return checkpoint_path


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def checkpoint_paths_for_reason(stage_id: str, reason: str, checkpoint_dir: Path | None = None, episode: int | None = None) -> tuple[Path, Path]:
    stage = get_stage(stage_id)
    base_dir = Path(checkpoint_dir) / stage_id if checkpoint_dir is not None else stage.checkpoint_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if reason == "best":
        return stage.best_training_state_path, stage.best_model_path
    if reason == "candidate":
        return (
            base_dir / f"candidateTrainingStateStage{stage.stage_number}.pt",
            base_dir / f"candidateBotStage{stage.stage_number}.zip",
        )
    if reason == "interrupted":
        return stage.interrupted_training_state_path, stage.interrupted_model_path
    if reason == "periodic" and episode is not None:
        return base_dir / f"checkpoint_ep_{episode:06d}.pt", base_dir / f"checkpoint_ep_{episode:06d}.zip"
    return stage.latest_training_state_path, stage.latest_model_path
