"""Watch a trained MaskablePPO/PPO model play the curriculum environment visually.

Examples:
    python watch.py --stage stage6
    python watch.py --stage stage8 --show-rays
    python watch.py --checkpoint checkpoints/curriculum_masked_ppo_v3/stage6/bestBotStage6.zip --stage stage6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config.stages import get_stage, normalize_stage_id
from rl.env import ShooterEnv
from training.model_utils import load_any_sb3_model, predict_with_action_masks
from training.skill_table import build_skill_table


def watch(stage_id: str, checkpoint: str | None, episodes: int = 10, show_rays: bool = False, validation: bool = True) -> None:
    stage_id = normalize_stage_id(stage_id)
    stage = get_stage(stage_id)
    checkpoint_path = Path(checkpoint) if checkpoint else stage.best_model_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading MaskablePPO/PPO model from: {checkpoint_path}")
    skill_table = build_skill_table(stage.stage_id, evaluation_mode=True)
    env = ShooterEnv(
        stage_id=stage.stage_id,
        render_mode="human",
        show_raycast_debug=show_rays,
        skill_table=skill_table,
        validation_mode=validation,
    )
    model = load_any_sb3_model(str(checkpoint_path), device="cpu")

    try:
        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            info = {}
            while not done:
                action, _ = predict_with_action_masks(model, obs, env, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
                time.sleep(1 / 60)

            result = "SUCCESS" if info.get("success") else "FAIL"
            print(
                f"Episode {episode:3d} | {result} | "
                f"Reward: {total_reward:8.2f} | "
                f"Steps: {info.get('steps', 0):4d} | "
                f"Metric: {float(info.get('stage_success_metric', 0.0)):5.2f} | "
                f"Aim: {float(info.get('aim_error_mean', 0.0)):.3f} | "
                f"OnTarget: {float(info.get('time_on_target_ratio', 0.0)):.2%} | "
                f"Invalid: {int(info.get('invalid_actions', 0))} | "
                f"Hit rate: {float(info.get('hit_rate', 0.0)):.2f} | "
                f"Damage dealt/taken: {float(info.get('damage_dealt', 0.0)):.1f}/{float(info.get('damage_taken', 0.0)):.1f}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained MaskablePPO/PPO bot with pygame rendering.")
    parser.add_argument("--stage", type=str, default="stage6", help="stage1 through stage8; old aliases still work")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional MaskablePPO/PPO .zip model path; defaults to that stage's best model.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--show-rays", action="store_true", help="Show observation/raycast debug lines; F3 toggles while running.")
    parser.add_argument("--training-scenarios", action="store_true", help="Watch training scenarios instead of validation scenarios.")
    args = parser.parse_args()
    watch(args.stage, args.checkpoint, args.episodes, args.show_rays, validation=not args.training_scenarios)


if __name__ == "__main__":
    main()
