from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config.stages import get_stage
from game.agent import AgentCommand


# Consistent PPO action interface for every stage:
# move:   0 none, 1 forward, 2 backward
# strafe: 0 none, 1 left,    2 right
# rotate: 0 none, 1 left,    2 right
# shoot:  0 no,   1 yes
# reload: 0 no,   1 yes
ACTION_NVECS = np.asarray([3, 3, 3, 2, 2], dtype=np.int64)
ACTION_LABELS = ("move", "strafe", "rotate", "shoot", "reload")


@dataclass(frozen=True)
class ActionDecodeResult:
    command: AgentCommand
    raw_action: tuple[int, int, int, int, int]
    invalid_actions: tuple[str, ...]


def noop_action() -> np.ndarray:
    return np.zeros_like(ACTION_NVECS, dtype=np.int64)


def always_shoot_action() -> np.ndarray:
    return np.asarray([0, 0, 0, 1, 0], dtype=np.int64)


def action_count(stage_id: str | None = None) -> int:
    # Kept for compatibility with old UI/debug text. PPO now uses MultiDiscrete.
    return int(np.prod(ACTION_NVECS))


def stage_action_mask(stage_id: str, shoot_allowed: bool | None = None, reload_allowed: bool | None = None) -> np.ndarray:
    """Flat MaskablePPO-compatible mask for the current stage.

    No-op remains valid in every branch, while locked branches expose only
    their no-op choice. ``shoot_allowed`` is used by Stage 2 to block wasteful
    shoot commands during cooldown/reload without changing the global action
    interface used by later stages.
    """
    stage = get_stage(stage_id)
    enabled = stage.actions_enabled
    shoot_enabled = enabled["shoot"] if shoot_allowed is None else enabled["shoot"] and bool(shoot_allowed)
    reload_enabled = enabled["reload"] if reload_allowed is None else enabled["reload"] and bool(reload_allowed)
    mask = np.asarray(
        [
            True,
            enabled["move"],
            enabled["move"],
            True,
            enabled["strafe"],
            enabled["strafe"],
            True,
            enabled["rotate"],
            enabled["rotate"],
            True,
            shoot_enabled,
            True,
            reload_enabled,
        ],
        dtype=bool,
    )
    return mask


def sample_valid_action(stage_id: str, rng: Any | None = None) -> np.ndarray:
    """Sample from the stage mask for random baselines and sanity checks."""
    flat_mask = stage_action_mask(stage_id)
    result: list[int] = []
    offset = 0
    for branch_size in ACTION_NVECS:
        branch_mask = flat_mask[offset : offset + int(branch_size)]
        choices = np.flatnonzero(branch_mask)
        if choices.size == 0:
            choices = np.asarray([0], dtype=np.int64)
        if rng is not None and hasattr(rng, "choice"):
            value = int(rng.choice(choices))
        elif rng is not None and hasattr(rng, "integers"):
            value = int(choices[int(rng.integers(0, len(choices)))])
        else:
            value = int(np.random.choice(choices))
        result.append(value)
        offset += int(branch_size)
    return np.asarray(result, dtype=np.int64)


def _int_to_multidiscrete(action: int) -> np.ndarray:
    values: list[int] = []
    remaining = max(0, int(action))
    for base in ACTION_NVECS:
        values.append(remaining % int(base))
        remaining //= int(base)
    return np.asarray(values, dtype=np.int64)


def _coerce_action(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.int64)
    if arr.shape == ():
        arr = _int_to_multidiscrete(int(arr.item()))
    arr = arr.reshape(-1)
    if arr.size < ACTION_NVECS.size:
        padded = np.zeros_like(ACTION_NVECS, dtype=np.int64)
        padded[: arr.size] = arr
        arr = padded
    elif arr.size > ACTION_NVECS.size:
        arr = arr[: ACTION_NVECS.size]
    return np.clip(arr, 0, ACTION_NVECS - 1).astype(np.int64)


def decode_action(stage_id: str, action: Any) -> ActionDecodeResult:
    stage = get_stage(stage_id)
    raw = _coerce_action(action)
    move_idx, strafe_idx, rot_idx, shoot_value, reload_value = [int(v) for v in raw]

    invalid: list[str] = []
    if move_idx and not stage.actions_enabled["move"]:
        invalid.append("move")
        move_idx = 0
    if strafe_idx and not stage.actions_enabled["strafe"]:
        invalid.append("strafe")
        strafe_idx = 0
    if rot_idx and not stage.actions_enabled["rotate"]:
        invalid.append("rotate")
        rot_idx = 0
    shoot = bool(shoot_value)
    if shoot and not stage.actions_enabled["shoot"]:
        invalid.append("shoot")
        shoot = False
    reload = bool(reload_value)
    if reload and not stage.actions_enabled["reload"]:
        invalid.append("reload")
        reload = False

    return ActionDecodeResult(
        command=AgentCommand(move_idx=move_idx, strafe_idx=strafe_idx, rot_idx=rot_idx, shoot=shoot, reload=reload),
        raw_action=tuple(int(v) for v in raw),
        invalid_actions=tuple(invalid),
    )


def action_to_command(stage_id: str, action: Any) -> AgentCommand:
    return decode_action(stage_id, action).command
