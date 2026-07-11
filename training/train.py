from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from config.stages import TRAINABLE_STAGE_ORDER, ensure_runtime_directories, get_stage, normalize_stage_id
from rl.env import ShooterEnv
from training.callbacks import StageTrainingCallback
from training.checkpointing import load_training_checkpoint, restore_rng_states
from training.skill_table import build_skill_table
from training.stage_manager import ensure_stage_stats_file, get_global_skill_table
from training.model_utils import load_any_sb3_model, load_preferred_model, make_model, preferred_algorithm_name


TRAINING_CHUNK_TIMESTEPS = 20_000
DEFAULT_STAGE_TIMESTEPS = {
    "stage1": 350_000,
    "stage2": 600_000,
    "stage3": 800_000,
    "stage4": 1_000_000,
    "stage5": 1_200_000,
    "stage6": 1_800_000,
    "stage7": 2_800_000,
    "stage8": 2_000_000,
}
DEFAULT_MAX_TIMESTEPS = 1_500_000


def _set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ppo_hyperparameters(stage_id: str | None = None) -> dict[str, Any]:
    normalized_stage_id = normalize_stage_id(stage_id or "stage1")
    if normalized_stage_id in {"stage5", "stage6", "stage7"}:
        ent_coef = 0.01
    elif normalized_stage_id in {"stage3", "stage4"}:
        ent_coef = 0.02
    else:
        ent_coef = 0.05
    params: dict[str, Any] = {
        # MaskablePPO keeps PPO but removes locked stage actions from sampling.
        # Stage 4 uses a lower entropy bonus and a KL guard because the policy
        # already discovers real dodging sometimes; the main risk now is noisy
        # destructive updates when projectile pressure changes.
        "learning_rate": 2.5e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.10,
        "ent_coef": ent_coef,
        "vf_coef": 0.50,
        "max_grad_norm": 0.50,
        "policy_kwargs": dict(net_arch=dict(pi=[160, 160], vf=[160, 160])),
        "verbose": 1,
        "device": "cpu",
    }
    if normalized_stage_id in {"stage4", "stage5", "stage6", "stage7"}:
        params["target_kl"] = 0.02
    return params


def _make_new_model(env: Any, stage_id: str | None = None) -> Any:
    params = _ppo_hyperparameters(stage_id)
    return make_model(env, params)


def _choose_same_stage_resume_path(stage, resume_same_stage: bool) -> Path | None:
    if not resume_same_stage:
        return None
    for candidate in (stage.interrupted_model_path, stage.latest_model_path, stage.best_model_path):
        if candidate.exists():
            return candidate
    return None


def _choose_previous_stage_source(stage_id: str, prev_source: str = "best") -> Path | None:
    stage = get_stage(stage_id)
    if stage.required_previous_stage is None:
        return None
    previous_stage = get_stage(stage.required_previous_stage)
    candidates = {
        "best": previous_stage.best_model_path,
        "latest": previous_stage.latest_model_path,
        "interrupted": previous_stage.interrupted_model_path,
    }
    if prev_source not in candidates:
        raise ValueError(f"Unknown prev_source: {prev_source}")
    path = candidates[prev_source]
    return path if path.exists() else None


def _transfer_compatible_policy_weights(target_model: Any, source_checkpoint: Path) -> None:
    """Copy matching actor/critic tensors from a previous stage.

    Stage 1 now uses a compact observation while later stages keep the richer
    vector, so direct loading may fail across that boundary. Matching tensors are
    still copied to preserve reusable hidden-layer knowledge where shapes allow.
    """
    source_model = load_any_sb3_model(source_checkpoint, device="cpu")
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    copied_keys: list[str] = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
            copied_keys.append(key)
    target_model.policy.load_state_dict(target_state, strict=False)
    print(f"Transferred {len(copied_keys)} compatible actor-critic tensors from {source_checkpoint}")


def _training_config(
    stage_id: str,
    visible: bool,
    save_every: int,
    checkpoint_dir: str | Path | None,
    n_envs: int,
    seed: int | None,
) -> dict[str, Any]:
    stage = get_stage(stage_id)
    return {
        "algorithm": preferred_algorithm_name(),
        "stage_id": stage.stage_id,
        "stage_name": stage.name,
        "visible": visible,
        "n_envs": n_envs,
        "seed": seed,
        "save_every_episodes": save_every,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else str(stage.checkpoint_dir),
        "ppo_hyperparameters": _ppo_hyperparameters(stage.stage_id),
        "stage_config": {
            "trainable": stage.trainable,
            "learning_skill": stage.learning_skill,
            "active_skills": list(stage.active_skills),
            "active_actions": list(stage.active_actions),
            "locked_actions": list(stage.locked_actions),
            "actions_enabled": stage.actions_enabled,
            "observation_inputs": list(stage.observation_inputs),
            "reward_components": list(stage.reward_components),
            "success_criteria": stage.success_criteria,
            "promotion_criteria": stage.promotion_criteria,
            "validation_criteria": stage.validation_criteria,
            "previous_skill_eval_stage_ids": list(stage.previous_skill_eval_stage_ids),
            "previous_skill_min_success": stage.previous_skill_min_success,
            "max_steps": stage.max_steps,
            "success_threshold": stage.success_threshold,
            "success_window_size": stage.success_window_size,
            "required_consecutive_checks": stage.required_consecutive_checks,
            "frame_stack": stage.frame_stack,
        },
    }


def _make_env_factory(
    stage_id: str,
    visible: bool,
    show_raycast_debug: bool,
    skill_table: dict[str, Any],
    validation_mode: bool,
    seed: int | None,
    rank: int,
):
    def factory() -> ShooterEnv:
        env = ShooterEnv(
            stage_id=stage_id,
            render_mode="human" if visible else None,
            show_raycast_debug=show_raycast_debug,
            skill_table=skill_table,
            validation_mode=validation_mode,
        )
        if seed is not None:
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            env.reset(seed=seed + rank)
        return env

    return factory


def _create_envs(
    stage_id: str,
    visible: bool,
    show_raycast_debug: bool,
    skill_table: dict[str, Any],
    n_envs: int,
    seed: int | None,
) -> tuple[Any, ShooterEnv]:
    if visible and n_envs != 1:
        print("Visual training supports one environment; using n_envs=1 for this run.")
        n_envs = 1

    env_fns = [
        _make_env_factory(stage_id, visible, show_raycast_debug, skill_table, validation_mode=False, seed=seed, rank=rank)
        for rank in range(max(1, n_envs))
    ]
    train_env: Any = env_fns[0]() if n_envs == 1 else DummyVecEnv(env_fns)

    eval_skill_table = build_skill_table(stage_id, previous_table=skill_table, evaluation_mode=True)
    eval_env = ShooterEnv(stage_id=stage_id, render_mode=None, skill_table=eval_skill_table, validation_mode=True)
    if seed is not None:
        eval_env.reset(seed=seed + 10_000)
    return train_env, eval_env


def _load_or_create_model(stage_id: str, train_env: Any, resume_same_stage: bool, prev_source: str) -> Any:
    stage = get_stage(stage_id)
    same_stage_resume_path = _choose_same_stage_resume_path(stage, resume_same_stage=resume_same_stage)
    if same_stage_resume_path is not None:
        print(f"Resuming {stage.name} MaskablePPO model from same-stage checkpoint: {same_stage_resume_path}")
        try:
            return load_preferred_model(same_stage_resume_path, env=train_env, device="cpu")
        except Exception as exc:
            print(f"Direct same-stage MaskablePPO load failed ({exc}). Creating a masked model and transferring compatible tensors.")
            model = _make_new_model(train_env, stage.stage_id)
            _transfer_compatible_policy_weights(model, same_stage_resume_path)
            return model

    previous_source = _choose_previous_stage_source(stage.stage_id, prev_source=prev_source)
    if previous_source is None:
        print(f"Starting {stage.name} with a new MaskablePPO actor-critic model.")
        return _make_new_model(train_env, stage.stage_id)

    try:
        print(f"Loading previous validated MaskablePPO stage model directly: {previous_source}")
        return load_preferred_model(previous_source, env=train_env, device="cpu")
    except Exception as exc:
        print(f"Direct masked load failed ({exc}). Creating a new model and transferring compatible tensors.")
        model = _make_new_model(train_env, stage.stage_id)
        _transfer_compatible_policy_weights(model, previous_source)
        return model


def _default_max_timesteps(stage_id: str) -> int:
    stage = get_stage(stage_id)
    return DEFAULT_STAGE_TIMESTEPS.get(stage.stage_id, DEFAULT_MAX_TIMESTEPS)


def run_training_for_stage(
    stage_id: str = "stage1",
    visible: bool = False,
    resume_same_stage: bool | None = None,
    prev_source: str = "best",
    resume_checkpoint: str | Path | None = None,
    save_every: int = 50,
    checkpoint_dir: str | Path | None = None,
    eval_every_steps: int = 4000,
    eval_episodes_per_check: int = 10,
    max_timesteps: int | None = None,
    show_raycast_debug: bool = False,
    n_envs: int = 1,
    seed: int | None = None,
) -> None:
    ensure_runtime_directories()
    ensure_stage_stats_file()
    _set_global_seed(seed)

    stage_id = normalize_stage_id(stage_id)
    stage = get_stage(stage_id)

    if not stage.trainable:
        from tools.sanity_check import run_stage0_sanity_check

        run_stage0_sanity_check(verbose=True)
        return

    max_timesteps = int(max_timesteps or _default_max_timesteps(stage.stage_id))
    resume_payload: dict[str, Any] | None = None
    callback_resume_state: dict[str, Any] | None = None
    append_logs = False

    if resume_checkpoint is not None:
        resume_payload = load_training_checkpoint(resume_checkpoint)
        stage_id = normalize_stage_id(str(resume_payload["stage_id"]))
        stage = get_stage(stage_id)
        restore_rng_states(resume_payload.get("rng_states"))
        skill_table = resume_payload.get("skill_table") or build_skill_table(stage.stage_id)
        callback_resume_state = resume_payload.get("callback_state") or {}
        append_logs = True
        print(f"Resuming MaskablePPO training state from: {resume_checkpoint}")
    else:
        previous_table = get_global_skill_table()
        skill_table = build_skill_table(stage.stage_id, previous_table=previous_table)

    if resume_same_stage is None:
        resume_same_stage = stage.allow_same_stage_resume

    train_env, eval_env = _create_envs(stage.stage_id, visible, show_raycast_debug, skill_table, n_envs=n_envs, seed=seed)

    if resume_payload is not None:
        model_path = Path(resume_payload["model_path"])
        print(f"Loading MaskablePPO model/optimizer from checkpoint model file: {model_path}")
        try:
            model = load_preferred_model(model_path, env=train_env, device="cpu")
        except Exception as exc:
            print(f"Preferred MaskablePPO checkpoint load failed ({exc}). Creating a masked model and transferring compatible tensors.")
            model = _make_new_model(train_env, stage.stage_id)
            _transfer_compatible_policy_weights(model, model_path)
    else:
        model = _load_or_create_model(stage.stage_id, train_env, resume_same_stage, prev_source)

    if stage.stage_id == "stage3" and eval_episodes_per_check < 25:
        eval_episodes_per_check = 25

    config = _training_config(stage.stage_id, visible, save_every, checkpoint_dir, n_envs=n_envs, seed=seed)
    callback = StageTrainingCallback(
        stage_id=stage.stage_id,
        eval_env=eval_env,
        skill_table=skill_table,
        training_config=config,
        eval_every_steps=eval_every_steps,
        eval_episodes_per_check=eval_episodes_per_check,
        success_window_size=stage.success_window_size,
        success_threshold=stage.success_threshold,
        required_consecutive_checks=stage.required_consecutive_checks,
        save_every_episodes=save_every,
        checkpoint_dir=checkpoint_dir,
        resume_state=callback_resume_state,
        append_logs=append_logs,
        verbose=1,
    )

    try:
        mode = "visual" if visible else "headless"
        print(f"Training {stage.name} with MaskablePPO in {mode} mode until validation passes...")
        print("Use Ctrl+C to save an interrupted checkpoint. In visual mode use P to pause, F3 for rays, and F5 to save.")
        # Treat max_timesteps as the additional budget for this stage, not as
        # an absolute global PPO counter.  Previous-stage checkpoints already
        # carry their own num_timesteps value.
        start_timesteps = int(model.num_timesteps)
        target_timesteps = start_timesteps + int(max_timesteps)
        while not callback.success_reached and model.num_timesteps < target_timesteps:
            remaining = max(1, target_timesteps - int(model.num_timesteps))
            chunk = min(TRAINING_CHUNK_TIMESTEPS, remaining)
            model.learn(total_timesteps=chunk, callback=callback, reset_num_timesteps=False)
            if callback.success_reached:
                break
            callback._save_checkpoint(reason="latest")
            print(f"Latest MaskablePPO checkpoint saved to: {callback.last_checkpoint_path}")

        if callback.success_reached:
            print(f"{stage.name} finished successfully: {stage.best_model_path}")
        else:
            print(f"Stopped after reaching stage budget ({max_timesteps} additional timesteps). Latest checkpoint: {callback.last_checkpoint_path}")

    except KeyboardInterrupt:
        callback.stop_reason = "keyboard_interrupt"
        callback._save_checkpoint(reason="interrupted")
        print(f"Training interrupted. Saved MaskablePPO resume checkpoint to: {callback.last_checkpoint_path}")

    finally:
        callback.close()
        train_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MaskablePPO curriculum stage for the shooter thesis project.")
    parser.add_argument("--stage", type=str, default="stage1", help="stage0 through stage8; old stageA-stageG aliases still map to the new stages.")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--visual", action="store_true", help="Run with visible pygame rendering and optional ray debug.")
    render_group.add_argument("--headless", action="store_true", help="Run without visible pygame rendering. This is the default.")
    parser.add_argument("--show-rays", action="store_true", help="Draw raycast debug lines in visual mode; F3 toggles while running.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a .pt training checkpoint, for example checkpoints/latest.pt")
    parser.add_argument("--resume-same-stage", action="store_true", help="Resume from interrupted/latest same-stage MaskablePPO model if it exists.")
    parser.add_argument("--prev-source", type=str, default="best", choices=["best", "latest", "interrupted"], help="Previous-stage MaskablePPO checkpoint source.")
    parser.add_argument("--save-every", type=int, default=50, help="Save periodic checkpoint every N completed episodes.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional directory for periodic checkpoints.")
    parser.add_argument("--eval-every", type=int, default=4000, help="Run deterministic validation every N timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Validation episodes per check.")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Safety cap; per-stage thesis defaults are used when omitted.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel headless training environments.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible training runs.")
    args = parser.parse_args()

    visible = bool(args.visual)
    if normalize_stage_id(args.stage) not in TRAINABLE_STAGE_ORDER and normalize_stage_id(args.stage) != "stage0":
        raise ValueError(f"Unknown trainable stage: {args.stage}")

    run_training_for_stage(
        stage_id=args.stage,
        visible=visible,
        resume_same_stage=True if args.resume_same_stage else False,
        prev_source=args.prev_source,
        resume_checkpoint=args.resume,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        eval_every_steps=args.eval_every,
        eval_episodes_per_check=args.eval_episodes,
        max_timesteps=args.max_timesteps,
        show_raycast_debug=args.show_rays,
        n_envs=max(1, args.n_envs),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
