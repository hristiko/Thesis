from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from game.agent import AgentCommand
from game.bot import StockBot
from rl.action_space import always_shoot_action, noop_action
from rl.env import ShooterEnv
from training.model_utils import load_any_sb3_model, predict_with_action_masks
from training.skill_table import build_skill_table


METRICS = [
    "success",
    "total_reward",
    "steps",
    "stage_success_metric",
    "kills",
    "deaths",
    "damage_dealt",
    "damage_taken",
    "damage_dealt_taken_ratio",
    "hits_taken",
    "shots_fired",
    "shots_hit",
    "shots_missed",
    "shots_blocked",
    "hit_rate",
    "miss_rate",
    "bad_shots",
    "bad_shot_rate",
    "moving_shots_fired",
    "moving_shots_hit",
    "moving_shot_rate",
    "moving_hit_rate",
    "moving_damage_dealt",
    "stage5_curriculum_level",
    "stage5_curriculum_total_levels",
    "stage5_curriculum_phase_index",
    "stage6_curriculum_level",
    "stage6_curriculum_total_levels",
    "stage6_curriculum_phase_index",
    "stage7_curriculum_level",
    "stage7_curriculum_total_levels",
    "stage7_curriculum_phase_index",
    "stage7_max_steps",
    "stage7_obstacle_count",
    "stage7_obstacle_variant_id",
    "stage7_spawn_index",
    "stage7_spawn_pair_id",
    "invalid_actions",
    "shot_unavailable_count",
    "reload_failures",
    "blocked_moves",
    "blocked_move_rate",
    "max_consecutive_blocked_moves",
    "line_of_sight_gained_count",
    "line_of_sight_lost_count",
    "clear_shot_gained_count",
    "clear_shot_lost_count",
    "steps_with_enemy_clear_shot",
    "enemy_clear_shot_ratio",
    "stage7_unclean_shot_attempts",
    "enemy_visible_ratio",
    "pickups_collected",
    "useful_pickups",
    "useless_pickups",
    "stage8_reload_started",
    "stage8_good_reloads",
    "stage8_acceptable_reloads",
    "stage8_early_reloads",
    "stage8_late_reloads",
    "stage8_empty_reloads",
    "stage8_reload_while_killable",
    "stage8_reload_quality",
    "stage8_target_ammo_reached",
    "stage8_target_ammo_steps",
    "stage8_shot_past_target_before_reload",
    "stage8_first_shot_rewarded",
    "stage8_no_first_shot_pressure_steps",
    "stage8_target_no_reload_timeout_steps",
    "stage8_8a_sequence_success",
    "stage8_low_ammo_steps",
    "stage8_critical_ammo_steps",
    "stage8_empty_ammo_steps",
    "stage8_clean_aligned_shots",
    "stage8_shots_after_reload",
    "stage8_unaligned_shot_attempts",
    "stage8_no_los_shot_attempts",
    "stage8_aim_hold_steps",
    "stage8_aim_improve_steps",
    "stage8_aim_worse_steps",
    "stage8_large_turn_count",
    "stage8_turn_flip_count",
    "stage8_turn_flip_rate",
    "stage8_blocked_spin_steps",
    "stage8_movement_flip_count",
    "stage8_movement_flip_rate",
    "stage8_smooth_move_steps",
    "projectile_danger_steps",
    "projectile_dodges",
    "projectile_hits_taken",
    "projectile_avoidance_rate",
    "aim_error_mean",
    "aim_error_min",
    "aim_error_final",
    "time_on_target_ratio",
    "max_aligned_steps",
    "heading_jitter_count",
    "heading_delta_total",
    "stage7_heading_delta_total",
    "stage7_heading_delta_mean",
    "stage7_large_turn_count",
    "stage7_turn_flip_count",
    "stage7_spin_steps",
    "stage7_blocked_spin_steps",
    "stage7_visible_turn_away_steps",
    "stage7_smooth_useful_turn_steps",
    "good_tracking_steps",
    "useful_range_steps",
    "range_streak_steps",
    "max_range_streak_steps",
    "stage3_usable_aim_steps",
    "stage3_usable_aim_ratio",
    "stage3_range_control_opportunities",
    "stage3_correct_range_control_steps",
    "stage3_wrong_range_control_steps",
    "stage3_range_control_ratio",
    "stage3_stable_range_steps",
    "stage3_stable_range_ratio",
    "stage3_avg_range_drift",
    "boundary_safe_steps",
    "boundary_safe_ratio",
    "boundary_hugging_steps",
    "nav_valid_steps",
    "nav_progress_positive_steps",
    "nav_progress_ratio",
    "nav_regress_steps",
    "nav_waypoint_reached_count",
    "nav_avg_path_deviation",
    "nav_path_deviation_bad_steps",
    "nav_avg_desired_distance",
    "nav_path_progress_sum",
    "nav_desired_progress_sum",
    "nav_direction_good_steps",
    "nav_direction_good_ratio",
    "nav_recompute_count",
    "movement_jitter_count",
    "movement_distance_total",
    "stationary_steps",
    "passive_steps",
    "survival_ratio",
]

COMPARISON_METRICS = [
    "success",
    "stage_success_metric",
    "total_reward",
    "steps",
    "aim_error_mean",
    "stage3_usable_aim_ratio",
    "stage3_range_control_ratio",
    "stage3_stable_range_ratio",
    "useful_range_steps",
    "enemy_visible_ratio",
    "enemy_clear_shot_ratio",
    "stage7_unclean_shot_attempts",
    "stage8_reload_started",
    "stage8_good_reloads",
    "stage8_shots_after_reload",
    "stage8_reload_quality",
    "nav_progress_ratio",
    "nav_avg_path_deviation",
    "movement_jitter_count",
    "movement_distance_total",
    "blocked_move_rate",
    "boundary_safe_ratio",
]


def _command_to_action(command: AgentCommand) -> np.ndarray:
    return np.asarray([
        int(command.move_idx),
        int(command.strafe_idx),
        int(command.rot_idx),
        1 if command.shoot else 0,
        1 if command.reload else 0,
    ], dtype=np.int64)


def _policy_from_name(name: str, env: ShooterEnv, model: Any | None = None) -> Callable[[np.ndarray], np.ndarray]:
    if name == "ppo":
        if model is None:
            raise ValueError("PPO/MaskablePPO policy requires a loaded model")

        def policy(obs: np.ndarray) -> np.ndarray:
            action, _ = predict_with_action_masks(model, obs, env, deterministic=True)
            return np.asarray(action, dtype=np.int64)

        return policy
    if name == "random":
        # Random baselines should test random valid behavior, not invalid-action spam.
        return lambda _obs: env.sample_valid_action()
    if name == "noop":
        return lambda _obs: noop_action()
    if name == "always_shoot":
        return lambda _obs: always_shoot_action()
    if name == "scripted_stock":
        holder: dict[str, StockBot | None] = {"bot": None}

        def policy(_obs: np.ndarray) -> np.ndarray:
            if env.world is None or env.learner is None:
                return noop_action()
            if holder["bot"] is None or holder["bot"].world is not env.world:
                holder["bot"] = StockBot(env.learner, env.world, target_id="enemy", profile="standard", rng=env.np_random)
            return _command_to_action(holder["bot"].tick())

        return policy
    raise ValueError(f"Unknown policy/baseline: {name}")


def evaluate_policy(
    *,
    stage_id: str,
    policy_name: str = "ppo",
    model_path: str | Path | None = None,
    episodes: int = 20,
    validation_mode: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    skill_table = build_skill_table(stage_id, evaluation_mode=True)
    env = ShooterEnv(stage_id=stage_id, render_mode=None, skill_table=skill_table, validation_mode=validation_mode)
    model = load_any_sb3_model(model_path, device="cpu") if policy_name == "ppo" and model_path is not None else None
    if policy_name == "ppo" and model is None:
        raise ValueError("policy_name='ppo' requires --model")

    policy = _policy_from_name(policy_name, env, model=model)
    episode_infos: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            obs, _ = env.reset(seed=episode_seed)
            done = False
            total_reward = 0.0
            info: dict[str, Any] = {}
            while not done:
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            info = dict(info)
            info["total_reward"] = total_reward
            if episode_seed is not None:
                info["eval_seed"] = episode_seed
            episode_infos.append(info)
    finally:
        env.close()

    return summarize_results(stage_id, policy_name, episode_infos, validation_mode=validation_mode)


def summarize_results(stage_id: str, policy_name: str, infos: list[dict[str, Any]], validation_mode: bool) -> dict[str, Any]:
    means: dict[str, float] = {}
    totals: dict[str, float] = {}
    for metric in METRICS:
        values: list[float] = []
        for info in infos:
            value = info.get(metric)
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            means[metric] = sum(values) / len(values)
            totals[metric] = sum(values)

    comparison_metrics = {metric: means[metric] for metric in COMPARISON_METRICS if metric in means}
    success_count = int(round(totals.get("success", 0.0)))

    return {
        "stage_id": stage_id,
        "policy": policy_name,
        "episodes": len(infos),
        "validation_mode": validation_mode,
        "scenario_mode": "validation" if validation_mode else "training",
        "success_count": success_count,
        "failure_count": len(infos) - success_count,
        "win_rate_or_stage_success_rate": means.get("success", 0.0),
        "average_reward": means.get("total_reward", 0.0),
        "comparison_metrics": comparison_metrics,
        "mean_metrics": means,
        "total_metrics": totals,
        "stage_promotion_reasons": [info.get("stage_promotion_reason", "") for info in infos],
        "stage8_phase_names": [info.get("stage8_phase_name") for info in infos if info.get("stage8_phase_name")],
        "eval_seeds": [info.get("eval_seed") for info in infos if info.get("eval_seed") is not None],
        "notes": (
            "Use the same --model, --episodes, and --seed with and without "
            "--training-scenarios to compare training-style spawns against "
            "held-out validation spawns. Stage 3 metrics such as usable aim, "
            "range control, nav progress, movement distance, jitter, and boundary "
            "safety are included in mean_metrics and comparison_metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO/PPO bot or simple baseline policies.")
    parser.add_argument("--stage", type=str, default="stage6")
    parser.add_argument("--policy", type=str, default="ppo", choices=["ppo", "random", "noop", "always_shoot", "scripted_stock"])
    parser.add_argument("--model", type=str, default=None, help="Path to MaskablePPO/PPO .zip model when --policy ppo is used.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--training-scenarios", action="store_true", help="Evaluate on training scenarios instead of held-out validation scenarios.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path. If omitted, the report is printed to stdout.")
    args = parser.parse_args()

    report = evaluate_policy(
        stage_id=args.stage,
        policy_name=args.policy,
        model_path=args.model,
        episodes=args.episodes,
        validation_mode=not args.training_scenarios,
        seed=args.seed,
    )
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
