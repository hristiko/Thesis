from __future__ import annotations

import csv
from collections import Counter, deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from config.stages import get_stage
from training.checkpointing import checkpoint_paths_for_reason, save_json_snapshot, save_training_checkpoint
from training.skill_table import build_skill_table, update_skill_table_after_stage
from training.stage_manager import update_training_stats
from training.model_utils import predict_with_action_masks
from training.promotion import evaluate_stage8_phase

NA = "NA"

METRIC_FIELDS = [
    "total_reward",
    "steps",
    "success",
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
    "stage5_episodes_at_curriculum_level",
    "stage5_enemy_fire_rate_scale",
    "stage5_first_shot_delay_steps",
    "stage5_max_active_enemy_projectiles",
    "stage5_enemy_can_move",
    "stage5_enemy_can_shoot",
    "stage5_curriculum_rolling_success",
    "stage5_curriculum_rolling_metric",
    "stage5_curriculum_rolling_death_rate",
    "stage5_curriculum_rolling_damage_dealt",
    "stage6_curriculum_level",
    "stage6_curriculum_total_levels",
    "stage6_curriculum_phase_index",
    "stage6_episodes_at_curriculum_level",
    "stage6_enemy_fire_rate_scale",
    "stage6_first_shot_delay_steps",
    "stage6_max_active_enemy_projectiles",
    "stage6_enemy_can_move",
    "stage6_enemy_can_shoot",
    "stage6_curriculum_rolling_success",
    "stage6_curriculum_rolling_metric",
    "stage6_curriculum_rolling_death_rate",
    "stage6_curriculum_rolling_damage_taken",
    "stage7_curriculum_level",
    "stage7_curriculum_total_levels",
    "stage7_curriculum_phase_index",
    "stage7_max_steps",
    "stage7_episodes_at_curriculum_level",
    "stage7_obstacle_count",
    "stage7_obstacle_variant_id",
    "stage7_spawn_index",
    "stage7_spawn_pair_id",
    "stage7_enemy_fire_rate_scale",
    "stage7_first_shot_delay_steps",
    "stage7_max_active_enemy_projectiles",
    "stage7_enemy_can_move",
    "stage7_enemy_can_shoot",
    "stage7_curriculum_rolling_success",
    "stage7_curriculum_rolling_metric",
    "stage7_curriculum_rolling_death_rate",
    "stage7_curriculum_rolling_blocked_rate",
    "stage7_curriculum_rolling_blocked_shots",
    "stage7_curriculum_rolling_damage_taken",
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
    "stage8_curriculum_level",
    "stage8_curriculum_total_levels",
    "stage8_curriculum_phase_index",
    "stage8_enemy_waves",
    "stage8_current_wave",
    "stage8_waves_cleared",
    "stage8_wave_clear_ratio",
    "stage8_max_steps",
    "stage8_episodes_at_curriculum_level",
    "stage8_obstacle_count",
    "stage8_obstacle_variant_id",
    "stage8_spawn_index",
    "stage8_spawn_pair_id",
    "stage8_last_enemy_respawn_index",
    "stage8_controlled_respawn_used",
    "stage8_enemy_fire_rate_scale",
    "stage8_first_shot_delay_steps",
    "stage8_max_active_enemy_projectiles",
    "stage8_enemy_can_move",
    "stage8_enemy_can_shoot",
    "stage8_curriculum_rolling_success",
    "stage8_curriculum_rolling_metric",
    "stage8_curriculum_rolling_death_rate",
    "stage8_curriculum_rolling_reload_quality",
    "stage8_curriculum_rolling_early_reloads",
    "stage8_curriculum_rolling_empty_ammo_steps",
    "stage8_reload_started",
    "stage8_timely_reloads",
    "stage8_late_rescue_reloads",
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
    "stage8_consecutive_empty_ammo_steps",
    "stage8_empty_ammo_timeout",
    "stage8_low_ammo_decision_steps",
    "stage8_low_ammo_decision_active",
    "stage8_clean_aligned_shots",
    "stage8_shots_after_reload",
    "stage8_unaligned_shot_attempts",
    "stage8_no_los_shot_attempts",
    "stage8_obstacle_no_los_shot_attempts",
    "stage8_no_los_shot_rate",
    "stage8_no_los_stationary_penalty_steps",
    "stage8_blind_rotation_steps",
    "stage8_blind_rotation_penalty_steps",
    "stage8_blind_rotation_shots",
    "stage8_consecutive_no_los_shots",
    "stage8_repeated_no_los_shot_penalties",
    "stage8_too_close_steps",
    "stage8_danger_close_steps",
    "stage8_close_withdraw_steps",
    "stage8_closing_when_too_close_steps",
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
    "stage8_passivity_penalty_steps",
    "stage8_passivity_penalty_sum",
    "stage8_steps_since_damage_dealt",
    "stage8_wasted_reloads",
    "stage8_magazine_shots_since_reload",
    "stage8_magazine_hits_since_reload",
    "stage8_magazine_damage_since_reload",
    "stage8_magazine_no_los_shots_since_reload",
    "projectile_danger_steps",
    "projectile_dodges",
    "projectile_hits_taken",
    "projectile_avoidance_rate",
    "stage4_curriculum_level",
    "stage4_curriculum_total_levels",
    "stage4_episodes_at_curriculum_level",
    "stage4_projectile_speed_scale",
    "stage4_enemy_fire_rate_scale",
    "stage4_first_shot_delay_steps",
    "stage4_max_active_enemy_projectiles",
    "stage4_curriculum_rolling_success",
    "stage4_curriculum_rolling_metric",
    "stage4_curriculum_rolling_death_rate",
    "stage4_curriculum_rolling_damage",
    "stage4_curriculum_rollbacks",
    "enemy_projectile_speed",
    "aim_error_mean",
    "aim_error_min",
    "aim_error_final",
    "time_on_target_ratio",
    "max_aligned_steps",
    "heading_jitter_count",
    "heading_delta_total",
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

TRAIN_LOG_FIELDS = [
    "episode",
    "stage",
    "stage_name",
    "timesteps",
    "rolling_success_rate",
    "stage_promotion_reason",
    "stage8_phase_name",
    "stage8_phase_passed",
    "stage8_eval_reason",
    "stage8_last_promotion_event",
    *METRIC_FIELDS,
    "PPO_policy_loss",
    "PPO_value_loss",
    "PPO_entropy",
    "PPO_approx_kl",
    "PPO_clip_fraction",
    "learning_rate",
    "checkpoint_path",
]

EVAL_LOG_FIELDS = [
    "eval_check",
    "eval_episode",
    "eval_seed",
    "stage",
    "stage_name",
    "timesteps",
    "rolling_success_rate",
    "validation_score",
    "good_checks_count",
    "required_good_checks",
    "stage_promotion_reason",
    "stage8_phase_name",
    "stage8_phase_passed",
    "stage8_eval_reason",
    "stage8_last_promotion_event",
    *METRIC_FIELDS,
]


# Stage 4 uses a training-only projectile-pressure curriculum. Validation keeps
# the official speed and official enemy pressure because validation environments
# ignore these temporary training settings.
#
# Advancement is intentionally conservative: a level is increased only after
# stable hard success plus acceptable damage/death statistics.  The previous
# version could advance from stage_success_metric alone, which let the agent
# jump to harder projectile timing before it had a stable dodging policy.
#
# A bridge level is inserted between the previous Level 2 and Level 3 because
# logs showed Level 2 was useful while the next jump caused a sharp collapse.
STAGE4_CURRICULUM_WINDOW = 40
STAGE4_ROLLBACK_MIN_EPISODES = 100

STAGE4_PROJECTILE_SPEED_LEVELS = (
    {
        "scale": 0.40,
        "fire_rate_scale": 0.22,
        "first_shot_delay_steps": 105,
        "max_active_projectiles": 1,
        "required_success": 0.08,
        "required_metric": 0.22,
        "max_death_rate": 0.70,
        "max_avg_damage": 92.0,
        "min_episodes": 160,
        "required_eval_checks": 3,
    },
    {
        "scale": 0.50,
        "fire_rate_scale": 0.30,
        "first_shot_delay_steps": 90,
        "max_active_projectiles": 1,
        "required_success": 0.10,
        "required_metric": 0.25,
        "max_death_rate": 0.68,
        "max_avg_damage": 90.0,
        "min_episodes": 160,
        "rollback_death_rate": 0.84,
        "rollback_max_success": 0.02,
        "rollback_min_damage": 97.0,
    },
    {
        "scale": 0.55,
        "fire_rate_scale": 0.35,
        "first_shot_delay_steps": 80,
        "max_active_projectiles": 1,
        "required_success": 0.11,
        "required_metric": 0.265,
        "max_death_rate": 0.67,
        "max_avg_damage": 89.0,
        "min_episodes": 170,
        "rollback_death_rate": 0.83,
        "rollback_max_success": 0.02,
        "rollback_min_damage": 96.5,
    },
    {
        "scale": 0.60,
        "fire_rate_scale": 0.40,
        "first_shot_delay_steps": 70,
        "max_active_projectiles": 1,
        "required_success": 0.12,
        "required_metric": 0.28,
        "max_death_rate": 0.65,
        "max_avg_damage": 88.0,
        "min_episodes": 180,
        "rollback_death_rate": 0.82,
        "rollback_max_success": 0.02,
        "rollback_min_damage": 96.0,
    },
    {
        "scale": 0.72,
        "fire_rate_scale": 0.55,
        "first_shot_delay_steps": 50,
        "max_active_projectiles": 1,
        "required_success": 0.16,
        "required_metric": 0.34,
        "max_death_rate": 0.60,
        "max_avg_damage": 84.0,
        "min_episodes": 200,
        "rollback_death_rate": 0.80,
        "rollback_max_success": 0.03,
        "rollback_min_damage": 94.0,
    },
    {
        "scale": 0.84,
        "fire_rate_scale": 0.70,
        "first_shot_delay_steps": 35,
        "max_active_projectiles": 2,
        "required_success": 0.22,
        "required_metric": 0.42,
        "max_death_rate": 0.55,
        "max_avg_damage": 78.0,
        "min_episodes": 220,
        "rollback_death_rate": 0.78,
        "rollback_max_success": 0.04,
        "rollback_min_damage": 91.0,
    },
    {
        "scale": 0.94,
        "fire_rate_scale": 0.85,
        "first_shot_delay_steps": 20,
        "max_active_projectiles": 2,
        "required_success": 0.30,
        "required_metric": 0.52,
        "max_death_rate": 0.48,
        "max_avg_damage": 70.0,
        "min_episodes": 240,
        "rollback_death_rate": 0.72,
        "rollback_max_success": 0.06,
        "rollback_min_damage": 84.0,
    },
    {
        "scale": 1.00,
        "fire_rate_scale": 1.00,
        "first_shot_delay_steps": 0,
        "max_active_projectiles": 99,
        "required_success": 1.01,
        "required_metric": 1.01,
        "min_episodes": 0,
        "rollback_death_rate": 0.68,
        "rollback_max_success": 0.08,
        "rollback_min_damage": 80.0,
    },
)


STAGE5_CURRICULUM_WINDOW = 30

STAGE5_CURRICULUM_PHASES = (
    {
        "phase": "5A",
        "phase_index": 1,
        "enemy_can_move": False,
        "enemy_can_shoot": False,
        "fire_rate_scale": 0.0,
        "first_shot_delay_steps": 9999,
        "max_active_projectiles": 0,
        "required_success": 0.35,
        "required_metric": 0.55,
        "min_avg_damage_dealt": 30.0,
        "max_death_rate": 1.0,
        "min_episodes": 120,
    },
    {
        "phase": "5B",
        "phase_index": 2,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.35,
        "first_shot_delay_steps": 120,
        "max_active_projectiles": 1,
        "required_success": 0.30,
        "required_metric": 0.52,
        "min_avg_damage_dealt": 40.0,
        "max_death_rate": 0.70,
        "min_episodes": 150,
    },
    {
        "phase": "5C",
        "phase_index": 3,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.55,
        "first_shot_delay_steps": 80,
        "max_active_projectiles": 1,
        "required_success": 1.01,
        "required_metric": 1.01,
        "min_avg_damage_dealt": 0.0,
        "max_death_rate": 1.0,
        "min_episodes": 0,
        "required_eval_checks": 3,
    },
)


STAGE6_CURRICULUM_WINDOW = 30

STAGE6_CURRICULUM_PHASES = (
    {
        "phase": "6A",
        "phase_index": 1,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.55,
        "first_shot_delay_steps": 90,
        "max_active_projectiles": 1,
        # 6A should prove the bot can win light duels reliably before pressure rises.
        "required_success": 0.70,
        "required_metric": 0.62,
        "max_death_rate": 0.35,
        "max_avg_damage_taken": 70.0,
        "min_episodes": 180,
        "required_eval_checks": 3,
    },
    {
        "phase": "6B",
        "phase_index": 2,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        # Medium pressure bridge.
        "fire_rate_scale": 0.75,
        "first_shot_delay_steps": 45,
        "max_active_projectiles": 2,
        "required_success": 0.55,
        "required_metric": 0.62,
        "max_death_rate": 0.45,
        "max_avg_damage_taken": 78.0,
        "min_episodes": 240,
    },
    {
        "phase": "6C",
        "phase_index": 3,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        # Hard bridge: more sustained fire than 6B, but still capped so the
        # policy can learn dodging under combat instead of collapsing into
        # immediate full-pressure death loops.
        "fire_rate_scale": 0.85,
        "first_shot_delay_steps": 20,
        "max_active_projectiles": 3,
        "required_success": 0.45,
        "required_metric": 0.60,
        "max_death_rate": 0.50,
        "max_avg_damage_taken": 84.0,
        "min_episodes": 320,
    },
    {
        "phase": "6D",
        "phase_index": 4,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        # Final full open-arena duel.  Official Stage 6 validation/best saving
        # only happens here.
        "fire_rate_scale": 1.00,
        "first_shot_delay_steps": 0,
        "max_active_projectiles": 99,
        "required_success": 1.01,
        "required_metric": 1.01,
        "max_death_rate": 1.0,
        "max_avg_damage_taken": 105.0,
        "min_episodes": 0,
        "required_eval_checks": 3,
    },
)

STAGE7_CURRICULUM_WINDOW = 30

STAGE7_CURRICULUM_PHASES = (
    {
        "phase": "7A",
        "phase_index": 1,
        "max_steps": 1200,
        "obstacle_count": 1,
        # Stationary enemy forces the learner to recover line of sight instead
        # of waiting for the stock bot to walk around the obstacle.
        "enemy_can_move": False,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.55,
        "first_shot_delay_steps": 120,
        "max_active_projectiles": 1,
        "required_success": 0.60,
        "required_metric": 0.62,
        "required_eval_success": 0.50,
        "required_eval_metric": 0.50,
        "required_eval_checks": 1,
        "max_eval_death_rate": 0.35,
        "max_eval_blocked_rate": 0.180,
        "min_eval_movement": 250.0,
        "min_eval_visible_ratio": 0.16,
        "min_eval_clear_shot_ratio": 0.035,
        "min_eval_time_on_target": 0.060,
        "min_eval_nav_progress": 0.080,
        "max_death_rate": 0.40,
        "max_avg_blocked_rate": 0.115,
        "max_avg_blocked_shots": 3.0,
        "max_avg_damage_taken": 92.0,
        "min_episodes": 200,
    },
    {
        "phase": "7B",
        "phase_index": 2,
        "max_steps": 1400,
        "obstacle_count": 2,
        # Still stationary: the learner must actively find the firing angle.
        "enemy_can_move": False,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.65,
        "first_shot_delay_steps": 90,
        "max_active_projectiles": 2,
        "required_success": 0.58,
        "required_metric": 0.62,
        "required_eval_success": 0.50,
        "required_eval_metric": 0.50,
        "required_eval_checks": 1,
        "max_eval_death_rate": 0.40,
        "max_eval_blocked_rate": 0.170,
        "min_eval_movement": 280.0,
        "min_eval_visible_ratio": 0.14,
        "min_eval_clear_shot_ratio": 0.030,
        "min_eval_time_on_target": 0.055,
        "min_eval_nav_progress": 0.080,
        "max_death_rate": 0.45,
        "max_avg_blocked_rate": 0.105,
        "max_avg_blocked_shots": 3.0,
        "max_avg_damage_taken": 94.0,
        "min_episodes": 240,
    },
    {
        "phase": "7C",
        "phase_index": 3,
        "max_steps": 1600,
        "obstacle_count": 3,
        # Transition phase: moving enemy returns, but pressure is still capped.
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.75,
        "first_shot_delay_steps": 45,
        "max_active_projectiles": 2,
        "required_success": 0.52,
        "required_metric": 0.60,
        "required_eval_success": 0.45,
        "required_eval_metric": 0.48,
        "required_eval_checks": 1,
        "max_eval_death_rate": 0.45,
        "max_eval_blocked_rate": 0.160,
        "min_eval_movement": 300.0,
        "min_eval_visible_ratio": 0.12,
        "min_eval_clear_shot_ratio": 0.028,
        "min_eval_time_on_target": 0.050,
        "min_eval_nav_progress": 0.070,
        "max_death_rate": 0.52,
        "max_avg_blocked_rate": 0.095,
        "max_avg_blocked_shots": 4.0,
        "max_avg_damage_taken": 96.0,
        "min_episodes": 280,
    },
    {
        "phase": "7D",
        "phase_index": 4,
        "max_steps": 1800,
        "obstacle_count": 4,
        # Four-obstacle bridge before full final validation.
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.85,
        "first_shot_delay_steps": 25,
        "max_active_projectiles": 3,
        "required_success": 0.45,
        "required_metric": 0.58,
        "required_eval_success": 0.40,
        "required_eval_metric": 0.46,
        "required_eval_checks": 1,
        "max_eval_death_rate": 0.50,
        "max_eval_blocked_rate": 0.150,
        "min_eval_movement": 320.0,
        "min_eval_visible_ratio": 0.10,
        "min_eval_clear_shot_ratio": 0.025,
        "min_eval_time_on_target": 0.045,
        "min_eval_nav_progress": 0.060,
        "max_death_rate": 0.55,
        "max_avg_blocked_rate": 0.090,
        "max_avg_blocked_shots": 5.0,
        "max_avg_damage_taken": 98.0,
        "min_episodes": 320,
    },
    {
        "phase": "7E",
        "phase_index": 5,
        "max_steps": 2000,
        "obstacle_count": 4,
        "enemy_can_move": True,
        "enemy_can_shoot": True,
        "fire_rate_scale": 1.00,
        "first_shot_delay_steps": 0,
        "max_active_projectiles": 99,
        "required_success": 1.01,
        "required_metric": 1.01,
        "max_death_rate": 1.0,
        "max_avg_blocked_rate": 1.0,
        "max_avg_blocked_shots": 999.0,
        "max_avg_damage_taken": 105.0,
        "min_episodes": 0,
    },
)


STAGE8_CURRICULUM_WINDOW = 30

# Intermediate Stage 8 phases are bridge phases, not final thesis validation.
# They use lighter gate counts so the curriculum can move forward once the
# behavior appears in deterministic evaluation. The final 8F phase keeps the
# stricter three-check gate.
STAGE8_CURRICULUM_PHASES = (
    {
        "phase": "8A",
        "phase_index": 1,
        "max_steps": 1600,
        "obstacle_count": 0,
        "enemy_waves": 5,
        "enemy_can_move": False,
        "enemy_can_shoot": False,
        "fire_rate_scale": 0.0,
        "first_shot_delay_steps": 9999,
        "max_active_projectiles": 0,
        "required_success": 0.25,
        "required_metric": 0.35,
        "max_death_rate": 0.35,
        "min_reload_quality": 0.20,
        "min_timely_reloads": 0.20,
        "max_avg_early_reloads": 2.5,
        "max_avg_empty_ammo_steps": 220.0,
        "min_episodes": 100,
        "required_eval_checks": 2,
    },
    {
        "phase": "8B",
        "phase_index": 2,
        "max_steps": 1800,
        "obstacle_count": 0,
        "enemy_waves": 5,
        "enemy_can_move": False,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.25,
        "first_shot_delay_steps": 180,
        "max_active_projectiles": 1,
        "required_success": 0.22,
        "required_metric": 0.34,
        "max_death_rate": 0.65,
        "min_reload_quality": 0.25,
        "min_timely_reloads": 0.20,
        "max_avg_early_reloads": 2.5,
        "max_avg_empty_ammo_steps": 220.0,
        "min_episodes": 140,
        "required_eval_checks": 1,
    },
    {
        "phase": "8C",
        "phase_index": 3,
        "max_steps": 1700,
        "obstacle_count": 1,
        "enemy_waves": 3,
        "enemy_can_move": False,
        "enemy_can_shoot": False,
        "fire_rate_scale": 0.0,
        "first_shot_delay_steps": 999999,
        "max_active_projectiles": 0,
        "required_success": 0.30,
        "required_metric": 0.38,
        "max_death_rate": 0.75,
        "min_reload_quality": 0.0,
        "min_timely_reloads": 0.0,
        "max_avg_early_reloads": 3.0,
        "max_avg_empty_ammo_steps": 500.0,
        "max_avg_no_los_shot_rate": 0.50,
        "max_avg_no_los_shot_attempts": 9.0,
        "max_avg_blind_rotation_steps": 60.0,
        "max_avg_too_close_steps": 180.0,
        "max_avg_danger_close_steps": 80.0,
        "min_episodes": 120,
        "required_eval_checks": 2,
    },
    {
        "phase": "8D",
        "phase_index": 4,
        "max_steps": 2000,
        "obstacle_count": 1,
        "enemy_waves": 4,
        "enemy_can_move": False,
        "enemy_can_shoot": False,
        "fire_rate_scale": 0.0,
        "first_shot_delay_steps": 999999,
        "max_active_projectiles": 0,
        "required_success": 0.35,
        "required_metric": 0.42,
        "max_death_rate": 0.75,
        "min_reload_quality": 0.0,
        "min_timely_reloads": 0.0,
        "max_avg_early_reloads": 3.0,
        "max_avg_empty_ammo_steps": 450.0,
        "max_avg_no_los_shot_rate": 0.40,
        "max_avg_no_los_shot_attempts": 8.0,
        "max_avg_blind_rotation_steps": 45.0,
        "max_avg_too_close_steps": 150.0,
        "max_avg_danger_close_steps": 60.0,
        "min_episodes": 160,
        "required_eval_checks": 3,
    },
    {
        "phase": "8E",
        "phase_index": 5,
        "max_steps": 2400,
        "obstacle_count": 0,
        "enemy_waves": 5,
        "enemy_can_move": False,
        "enemy_can_shoot": False,
        "fire_rate_scale": 0.0,
        "first_shot_delay_steps": 999999,
        "max_active_projectiles": 0,
        "required_success": 0.25,
        "required_metric": 0.34,
        "max_death_rate": 0.80,
        "min_reload_quality": 0.15,
        "min_timely_reloads": 0.20,
        "max_avg_early_reloads": 1.5,
        "max_avg_empty_ammo_steps": 500.0,
        "max_avg_no_los_shot_rate": 0.35,
        "max_avg_no_los_shot_attempts": 8.0,
        # 8E is reload endurance. Obstacle-control diagnostics are logged, but
        # they should not be the main promotion blocker here. 8F is the harder
        # integrated obstacle + enemy-fire validation.
        "max_avg_blind_rotation_steps": 9999.0,
        "max_avg_too_close_steps": 9999.0,
        "max_avg_danger_close_steps": 9999.0,
        "min_episodes": 160,
        "required_eval_checks": 3,
    },
    {
        "phase": "8F",
        "phase_index": 6,
        "max_steps": 2600,
        "obstacle_count": 1,
        "enemy_waves": 5,
        "enemy_can_move": False,
        "enemy_can_shoot": True,
        "fire_rate_scale": 0.25,
        "first_shot_delay_steps": 150,
        "max_active_projectiles": 1,
        "required_success": 0.20,
        "required_metric": 0.32,
        "max_death_rate": 0.80,
        "min_reload_quality": 0.15,
        "min_timely_reloads": 0.20,
        "max_avg_early_reloads": 3.5,
        "max_avg_empty_ammo_steps": 420.0,
        "max_avg_no_los_shot_rate": 0.28,
        "max_avg_no_los_shot_attempts": 6.0,
        "max_avg_blind_rotation_steps": 35.0,
        "max_avg_too_close_steps": 120.0,
        "max_avg_danger_close_steps": 45.0,
        "min_episodes": 220,
        "required_eval_checks": 3,
    },
)




class StageTrainingCallback(BaseCallback):
    """PPO monitor, validator, checkpoint writer, and curriculum gatekeeper.

    Stable-Baselines3 performs PPO rollout collection and policy updates.  This
    callback owns the thesis-specific curriculum logic: it logs episode metrics,
    evaluates on validation scenarios, saves best models only through validation,
    and unlocks the next stage only after consecutive validation success plus
    previous-skill retention checks.
    """

    def __init__(
        self,
        stage_id: str,
        eval_env,
        skill_table: dict[str, Any],
        training_config: dict[str, Any],
        eval_every_steps: int = 4000,
        eval_episodes_per_check: int = 10,
        success_window_size: int | None = None,
        success_threshold: float | None = None,
        required_consecutive_checks: int | None = None,
        save_every_episodes: int = 50,
        checkpoint_dir: str | Path | None = None,
        resume_state: dict[str, Any] | None = None,
        append_logs: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.stage = get_stage(stage_id)
        self.eval_env = eval_env
        self.skill_table = skill_table
        self.training_config = training_config
        self.eval_every_steps = int(eval_every_steps)
        self.eval_episodes_per_check = int(eval_episodes_per_check)
        if self.stage.stage_id == "stage3" and self.eval_episodes_per_check < 25:
            self.eval_episodes_per_check = 25
        if self.stage.stage_id == "stage6" and self.eval_episodes_per_check < 15:
            # Stage 6 validation was too seed-sensitive with very small batches.
            # Use a slightly larger batch to reduce noisy good-check resets.
            self.eval_episodes_per_check = 15
        if self.stage.stage_id == "stage7" and self.eval_episodes_per_check < 15:
            # Stage 7 validation is also seed-sensitive because obstacle layouts
            # change line of sight, pathing, and blocked-shot opportunities.
            self.eval_episodes_per_check = 15
        self.success_window_size = int(success_window_size or self.stage.success_window_size)
        self.success_threshold = float(success_threshold or self.stage.success_threshold)
        self.required_consecutive_checks = int(required_consecutive_checks or self.stage.required_consecutive_checks)
        self.save_every_episodes = int(save_every_episodes)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.append_logs = bool(append_logs)

        self.episode_counter = 0
        self.eval_check_counter = 0
        self.total_eval_episodes = 0
        self.train_success_window: deque[int] = deque(maxlen=self.success_window_size)
        self.eval_success_window: deque[int] = deque(maxlen=self.success_window_size)
        self.best_eval_success_rate = 0.0
        self.best_validation_score = 0.0
        self.last_eval_success_rate = 0.0
        self.last_validation_score = 0.0
        self.last_eval_timestep = 0
        self.consecutive_good_eval_checks = 0
        self.success_reached = False
        self.last_checkpoint_path = ""
        self.stop_reason = "running"
        self.session_started = False
        self.last_validation_report: dict[str, Any] = {}
        raw_seed = training_config.get("seed")
        self.eval_seed_base = int(raw_seed) + 10_000 if raw_seed is not None else 10_000

        self.stage4_projectile_speed_level = 0
        self.stage4_episodes_at_projectile_level = 0
        self.stage4_curriculum_rollbacks = 0
        self.stage4_level_success_window: deque[int] = deque(maxlen=STAGE4_CURRICULUM_WINDOW)
        self.stage4_level_metric_window: deque[float] = deque(maxlen=STAGE4_CURRICULUM_WINDOW)
        self.stage4_level_damage_window: deque[float] = deque(maxlen=STAGE4_CURRICULUM_WINDOW)
        self.stage4_level_death_window: deque[int] = deque(maxlen=STAGE4_CURRICULUM_WINDOW)
        self.stage4_best_level_scores: dict[str, float] = {}

        self.stage5_curriculum_phase = 0
        self.stage5_episodes_at_phase = 0
        self.stage5_phase_success_window: deque[int] = deque(maxlen=STAGE5_CURRICULUM_WINDOW)
        self.stage5_phase_metric_window: deque[float] = deque(maxlen=STAGE5_CURRICULUM_WINDOW)
        self.stage5_phase_death_window: deque[int] = deque(maxlen=STAGE5_CURRICULUM_WINDOW)
        self.stage5_phase_damage_dealt_window: deque[float] = deque(maxlen=STAGE5_CURRICULUM_WINDOW)
        self.stage5_best_phase_scores: dict[str, float] = {}

        self.stage6_curriculum_phase = 0
        self.stage6_episodes_at_phase = 0
        self.stage6_phase_success_window: deque[int] = deque(maxlen=STAGE6_CURRICULUM_WINDOW)
        self.stage6_phase_metric_window: deque[float] = deque(maxlen=STAGE6_CURRICULUM_WINDOW)
        self.stage6_phase_death_window: deque[int] = deque(maxlen=STAGE6_CURRICULUM_WINDOW)
        self.stage6_phase_damage_taken_window: deque[float] = deque(maxlen=STAGE6_CURRICULUM_WINDOW)
        self.stage6_best_phase_scores: dict[str, float] = {}
        self.stage6_candidate_score = 0.0

        self.stage7_curriculum_phase = 0
        self.stage7_episodes_at_phase = 0
        self.stage7_phase_success_window: deque[int] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_phase_metric_window: deque[float] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_phase_death_window: deque[int] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_phase_blocked_rate_window: deque[float] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_phase_blocked_shots_window: deque[float] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_phase_damage_taken_window: deque[float] = deque(maxlen=STAGE7_CURRICULUM_WINDOW)
        self.stage7_best_phase_scores: dict[str, float] = {}
        self.stage7_eval_gate_phase = 0
        self.stage7_eval_gate_good_checks = 0
        self.stage7_last_eval_gate_summary: dict[str, Any] = {}
        self._stage7_next_progress_print_timestep = 0
        self._stage7_progress_interval = 300

        self.stage8_curriculum_phase = 0
        self.stage8_episodes_at_phase = 0
        self.stage8_phase_success_window: deque[int] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_phase_metric_window: deque[float] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_phase_death_window: deque[int] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_phase_reload_quality_window: deque[float] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_phase_early_reload_window: deque[float] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_phase_empty_ammo_window: deque[float] = deque(maxlen=STAGE8_CURRICULUM_WINDOW)
        self.stage8_best_phase_scores: dict[str, float] = {}
        self.stage8_eval_gate_phase = 0
        self.stage8_eval_gate_good_checks = 0
        self.stage8_last_eval_gate_summary: dict[str, Any] = {}
        self.stage8_last_promotion_event: str = ""

        if resume_state:
            self._restore_state(resume_state)

        self.train_log_file = None
        self.eval_log_file = None
        self.train_writer = None
        self.eval_writer = None

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.episode_counter = int(state.get("episode_counter", 0))
        self.eval_check_counter = int(state.get("eval_check_counter", 0))
        self.total_eval_episodes = int(state.get("total_eval_episodes", 0))
        self.train_success_window = deque([int(v) for v in state.get("train_success_window", [])], maxlen=self.success_window_size)
        self.eval_success_window = deque([int(v) for v in state.get("eval_success_window", [])], maxlen=self.success_window_size)
        self.best_eval_success_rate = float(state.get("best_eval_success_rate", 0.0))
        self.best_validation_score = float(state.get("best_validation_score", 0.0))
        self.last_eval_success_rate = float(state.get("last_eval_success_rate", 0.0))
        self.last_validation_score = float(state.get("last_validation_score", 0.0))
        self.last_eval_timestep = int(state.get("last_eval_timestep", 0))
        self.consecutive_good_eval_checks = int(state.get("consecutive_good_eval_checks", 0))
        self.last_checkpoint_path = str(state.get("last_checkpoint_path", ""))
        self.last_validation_report = dict(state.get("last_validation_report", {}))
        self.eval_seed_base = int(state.get("eval_seed_base", self.eval_seed_base))
        self.stage4_projectile_speed_level = int(state.get("stage4_projectile_speed_level", self.stage4_projectile_speed_level))
        self.stage4_projectile_speed_level = max(0, min(self.stage4_projectile_speed_level, len(STAGE4_PROJECTILE_SPEED_LEVELS) - 1))
        self.stage4_episodes_at_projectile_level = int(state.get("stage4_episodes_at_projectile_level", self.stage4_episodes_at_projectile_level))
        self.stage4_curriculum_rollbacks = int(state.get("stage4_curriculum_rollbacks", self.stage4_curriculum_rollbacks))

        self.stage4_level_success_window.clear()
        for value in state.get("stage4_level_success_window", []):
            try:
                self.stage4_level_success_window.append(int(value))
            except (TypeError, ValueError):
                continue

        self.stage4_level_metric_window.clear()
        metric_values = state.get("stage4_level_metric_window", state.get("stage4_metric_window", []))
        for value in metric_values:
            try:
                self.stage4_level_metric_window.append(float(value))
            except (TypeError, ValueError):
                continue

        self.stage4_level_damage_window.clear()
        for value in state.get("stage4_level_damage_window", []):
            try:
                self.stage4_level_damage_window.append(float(value))
            except (TypeError, ValueError):
                continue

        self.stage4_level_death_window.clear()
        for value in state.get("stage4_level_death_window", []):
            try:
                self.stage4_level_death_window.append(int(value))
            except (TypeError, ValueError):
                continue

        self.stage4_best_level_scores = {str(k): float(v) for k, v in dict(state.get("stage4_best_level_scores", {})).items()}

        self.stage5_curriculum_phase = int(state.get("stage5_curriculum_phase", self.stage5_curriculum_phase))
        self.stage5_curriculum_phase = max(0, min(self.stage5_curriculum_phase, len(STAGE5_CURRICULUM_PHASES) - 1))
        self.stage5_episodes_at_phase = int(state.get("stage5_episodes_at_phase", self.stage5_episodes_at_phase))
        self.stage5_phase_success_window.clear()
        for value in state.get("stage5_phase_success_window", []):
            try:
                self.stage5_phase_success_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage5_phase_metric_window.clear()
        for value in state.get("stage5_phase_metric_window", []):
            try:
                self.stage5_phase_metric_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage5_phase_death_window.clear()
        for value in state.get("stage5_phase_death_window", []):
            try:
                self.stage5_phase_death_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage5_phase_damage_dealt_window.clear()
        for value in state.get("stage5_phase_damage_dealt_window", []):
            try:
                self.stage5_phase_damage_dealt_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage5_best_phase_scores = {str(k): float(v) for k, v in dict(state.get("stage5_best_phase_scores", {})).items()}

        self.stage6_curriculum_phase = int(state.get("stage6_curriculum_phase", self.stage6_curriculum_phase))
        self.stage6_curriculum_phase = max(0, min(self.stage6_curriculum_phase, len(STAGE6_CURRICULUM_PHASES) - 1))
        self.stage6_episodes_at_phase = int(state.get("stage6_episodes_at_phase", self.stage6_episodes_at_phase))
        self.stage6_phase_success_window.clear()
        for value in state.get("stage6_phase_success_window", []):
            try:
                self.stage6_phase_success_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage6_phase_metric_window.clear()
        for value in state.get("stage6_phase_metric_window", []):
            try:
                self.stage6_phase_metric_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage6_phase_death_window.clear()
        for value in state.get("stage6_phase_death_window", []):
            try:
                self.stage6_phase_death_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage6_phase_damage_taken_window.clear()
        for value in state.get("stage6_phase_damage_taken_window", []):
            try:
                self.stage6_phase_damage_taken_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage6_best_phase_scores = {str(k): float(v) for k, v in dict(state.get("stage6_best_phase_scores", {})).items()}
        self.stage6_candidate_score = float(state.get("stage6_candidate_score", self.stage6_candidate_score))

        self.stage7_curriculum_phase = int(state.get("stage7_curriculum_phase", self.stage7_curriculum_phase))
        self.stage7_curriculum_phase = max(0, min(self.stage7_curriculum_phase, len(STAGE7_CURRICULUM_PHASES) - 1))
        self.stage7_episodes_at_phase = int(state.get("stage7_episodes_at_phase", self.stage7_episodes_at_phase))
        self.stage7_phase_success_window.clear()
        for value in state.get("stage7_phase_success_window", []):
            try:
                self.stage7_phase_success_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage7_phase_metric_window.clear()
        for value in state.get("stage7_phase_metric_window", []):
            try:
                self.stage7_phase_metric_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage7_phase_death_window.clear()
        for value in state.get("stage7_phase_death_window", []):
            try:
                self.stage7_phase_death_window.append(int(value))
            except (TypeError, ValueError):
                continue
        self.stage7_phase_blocked_rate_window.clear()
        for value in state.get("stage7_phase_blocked_rate_window", []):
            try:
                self.stage7_phase_blocked_rate_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage7_phase_blocked_shots_window.clear()
        for value in state.get("stage7_phase_blocked_shots_window", []):
            try:
                self.stage7_phase_blocked_shots_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage7_phase_damage_taken_window.clear()
        for value in state.get("stage7_phase_damage_taken_window", []):
            try:
                self.stage7_phase_damage_taken_window.append(float(value))
            except (TypeError, ValueError):
                continue
        self.stage7_best_phase_scores = {str(k): float(v) for k, v in dict(state.get("stage7_best_phase_scores", {})).items()}
        self.stage7_eval_gate_phase = int(state.get("stage7_eval_gate_phase", self.stage7_curriculum_phase))
        self.stage7_eval_gate_good_checks = int(state.get("stage7_eval_gate_good_checks", 0))
        self.stage7_last_eval_gate_summary = dict(state.get("stage7_last_eval_gate_summary", {}))

        self.stage8_curriculum_phase = int(state.get("stage8_curriculum_phase", self.stage8_curriculum_phase))
        self.stage8_curriculum_phase = max(0, min(self.stage8_curriculum_phase, len(STAGE8_CURRICULUM_PHASES) - 1))
        self.stage8_episodes_at_phase = int(state.get("stage8_episodes_at_phase", self.stage8_episodes_at_phase))
        self.stage8_phase_success_window.clear()
        for value in state.get("stage8_phase_success_window", []):
            try: self.stage8_phase_success_window.append(int(value))
            except (TypeError, ValueError): continue
        self.stage8_phase_metric_window.clear()
        for value in state.get("stage8_phase_metric_window", []):
            try: self.stage8_phase_metric_window.append(float(value))
            except (TypeError, ValueError): continue
        self.stage8_phase_death_window.clear()
        for value in state.get("stage8_phase_death_window", []):
            try: self.stage8_phase_death_window.append(int(value))
            except (TypeError, ValueError): continue
        self.stage8_phase_reload_quality_window.clear()
        for value in state.get("stage8_phase_reload_quality_window", []):
            try: self.stage8_phase_reload_quality_window.append(float(value))
            except (TypeError, ValueError): continue
        self.stage8_phase_early_reload_window.clear()
        for value in state.get("stage8_phase_early_reload_window", []):
            try: self.stage8_phase_early_reload_window.append(float(value))
            except (TypeError, ValueError): continue
        self.stage8_phase_empty_ammo_window.clear()
        for value in state.get("stage8_phase_empty_ammo_window", []):
            try: self.stage8_phase_empty_ammo_window.append(float(value))
            except (TypeError, ValueError): continue
        self.stage8_best_phase_scores = {str(k): float(v) for k, v in dict(state.get("stage8_best_phase_scores", {})).items()}
        self.stage8_eval_gate_phase = int(state.get("stage8_eval_gate_phase", self.stage8_curriculum_phase))
        self.stage8_eval_gate_good_checks = int(state.get("stage8_eval_gate_good_checks", 0))
        self.stage8_last_eval_gate_summary = dict(state.get("stage8_last_eval_gate_summary", {}))
        self.stage8_last_promotion_event = str(state.get("stage8_last_promotion_event", ""))

    def get_state(self) -> dict[str, Any]:
        return {
            "episode_counter": self.episode_counter,
            "eval_check_counter": self.eval_check_counter,
            "total_eval_episodes": self.total_eval_episodes,
            "train_success_window": list(self.train_success_window),
            "eval_success_window": list(self.eval_success_window),
            "best_eval_success_rate": self.best_eval_success_rate,
            "best_validation_score": self.best_validation_score,
            "last_eval_success_rate": self.last_eval_success_rate,
            "last_validation_score": self.last_validation_score,
            "last_eval_timestep": self.last_eval_timestep,
            "consecutive_good_eval_checks": self.consecutive_good_eval_checks,
            "success_reached": self.success_reached,
            "last_checkpoint_path": self.last_checkpoint_path,
            "last_validation_report": self.last_validation_report,
            "eval_seed_base": self.eval_seed_base,
            "stage4_projectile_speed_level": self.stage4_projectile_speed_level,
            "stage4_episodes_at_projectile_level": self.stage4_episodes_at_projectile_level,
            "stage4_curriculum_rollbacks": self.stage4_curriculum_rollbacks,
            "stage4_level_success_window": list(self.stage4_level_success_window),
            "stage4_level_metric_window": list(self.stage4_level_metric_window),
            "stage4_level_damage_window": list(self.stage4_level_damage_window),
            "stage4_level_death_window": list(self.stage4_level_death_window),
            "stage4_best_level_scores": dict(self.stage4_best_level_scores),
            "stage5_curriculum_phase": self.stage5_curriculum_phase,
            "stage5_episodes_at_phase": self.stage5_episodes_at_phase,
            "stage5_phase_success_window": list(self.stage5_phase_success_window),
            "stage5_phase_metric_window": list(self.stage5_phase_metric_window),
            "stage5_phase_death_window": list(self.stage5_phase_death_window),
            "stage5_phase_damage_dealt_window": list(self.stage5_phase_damage_dealt_window),
            "stage5_best_phase_scores": dict(self.stage5_best_phase_scores),
            "stage6_curriculum_phase": self.stage6_curriculum_phase,
            "stage6_episodes_at_phase": self.stage6_episodes_at_phase,
            "stage6_phase_success_window": list(self.stage6_phase_success_window),
            "stage6_phase_metric_window": list(self.stage6_phase_metric_window),
            "stage6_phase_death_window": list(self.stage6_phase_death_window),
            "stage6_phase_damage_taken_window": list(self.stage6_phase_damage_taken_window),
            "stage6_best_phase_scores": dict(self.stage6_best_phase_scores),
            "stage6_candidate_score": self.stage6_candidate_score,
            "stage7_curriculum_phase": self.stage7_curriculum_phase,
            "stage7_episodes_at_phase": self.stage7_episodes_at_phase,
            "stage7_phase_success_window": list(self.stage7_phase_success_window),
            "stage7_phase_metric_window": list(self.stage7_phase_metric_window),
            "stage7_phase_death_window": list(self.stage7_phase_death_window),
            "stage7_phase_blocked_rate_window": list(self.stage7_phase_blocked_rate_window),
            "stage7_phase_blocked_shots_window": list(self.stage7_phase_blocked_shots_window),
            "stage7_phase_damage_taken_window": list(self.stage7_phase_damage_taken_window),
            "stage7_best_phase_scores": dict(self.stage7_best_phase_scores),
            "stage7_eval_gate_phase": self.stage7_eval_gate_phase,
            "stage7_eval_gate_good_checks": self.stage7_eval_gate_good_checks,
            "stage7_last_eval_gate_summary": dict(self.stage7_last_eval_gate_summary),
            "stage8_curriculum_phase": self.stage8_curriculum_phase,
            "stage8_episodes_at_phase": self.stage8_episodes_at_phase,
            "stage8_phase_success_window": list(self.stage8_phase_success_window),
            "stage8_phase_metric_window": list(self.stage8_phase_metric_window),
            "stage8_phase_death_window": list(self.stage8_phase_death_window),
            "stage8_phase_reload_quality_window": list(self.stage8_phase_reload_quality_window),
            "stage8_phase_early_reload_window": list(self.stage8_phase_early_reload_window),
            "stage8_phase_empty_ammo_window": list(self.stage8_phase_empty_ammo_window),
            "stage8_best_phase_scores": dict(self.stage8_best_phase_scores),
            "stage8_eval_gate_phase": self.stage8_eval_gate_phase,
            "stage8_eval_gate_good_checks": self.stage8_eval_gate_good_checks,
            "stage8_last_eval_gate_summary": dict(self.stage8_last_eval_gate_summary),
            "stage8_last_promotion_event": self.stage8_last_promotion_event,
            "stop_reason": self.stop_reason,
        }

    def _on_training_start(self) -> None:
        append = self.append_logs or self.session_started
        self._open_csv_writers(append=append)
        self.session_started = True
        save_json_snapshot(self.stage.config_snapshot_path, self.training_config)
        save_json_snapshot(self.stage.skill_table_snapshot_path, self.skill_table)
        self._apply_stage4_projectile_speed_scale()
        self._apply_stage5_curriculum_phase()
        self._apply_stage6_curriculum_phase()
        self._apply_stage7_curriculum_phase()
        self._apply_stage8_curriculum_phase()

        # When a new stage is initialized from a previous-stage PPO checkpoint,
        # the model can already carry a large absolute num_timesteps value.
        # Without rebasing this counter, the eval condition below can fire after
        # the first visible env step, making the Pygame window look frozen while
        # headless validation silently runs.
        model_timesteps = int(getattr(self.model, "num_timesteps", getattr(self, "num_timesteps", 0)) or 0)
        if self.last_eval_timestep <= 0 and model_timesteps > 0:
            self.last_eval_timestep = model_timesteps

        if self.stage.stage_id == "stage7":
            phase = self._stage7_current_phase()
            self._stage7_next_progress_print_timestep = model_timesteps + self._stage7_progress_interval
            print(
                f"[START stage7] Phase {phase.get('phase', '7A')} | "
                f"max_steps {int(phase.get('max_steps', 1200))} | "
                f"eval_every {self.eval_every_steps} steps | "
                f"eval_episodes {self.eval_episodes_per_check} | "
                f"next_eval_at_timestep {self.last_eval_timestep + self.eval_every_steps}"
            )
        if self.stage.stage_id == "stage8":
            phase = self._stage8_current_phase()
            print(
                f"[START stage8] Phase {phase.get('phase', '8A')} | "
                f"max_steps {int(phase.get('max_steps', 900))} | "
                f"eval_every {self.eval_every_steps} steps | "
                f"eval_episodes {self.eval_episodes_per_check} | "
                f"next_eval_at_timestep {self.last_eval_timestep + self.eval_every_steps}"
            )

    def _stage4_current_projectile_speed_scale(self) -> float:
        return float(STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]["scale"])

    def _apply_stage4_projectile_speed_scale(self) -> None:
        if self.stage.stage_id != "stage4":
            return
        level = STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]
        scale = float(level["scale"])
        fire_rate_scale = float(level.get("fire_rate_scale", 1.0))
        first_shot_delay_steps = int(level.get("first_shot_delay_steps", 0))
        max_active_projectiles = int(level.get("max_active_projectiles", 99))
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_stage4_projectile_speed_scale", scale)
            self.training_env.env_method(
                "set_stage4_projectile_pressure",
                fire_rate_scale,
                first_shot_delay_steps,
                max_active_projectiles,
            )
            self.training_env.env_method(
                "set_stage4_curriculum_level",
                self.stage4_projectile_speed_level + 1,
                len(STAGE4_PROJECTILE_SPEED_LEVELS),
            )
            return
        try:
            self.training_env.unwrapped.set_stage4_projectile_speed_scale(scale)
            self.training_env.unwrapped.set_stage4_projectile_pressure(
                fire_rate_scale,
                first_shot_delay_steps,
                max_active_projectiles,
            )
            self.training_env.unwrapped.set_stage4_curriculum_level(
                self.stage4_projectile_speed_level + 1,
                len(STAGE4_PROJECTILE_SPEED_LEVELS),
            )
        except AttributeError:
            self.training_env.set_stage4_projectile_speed_scale(scale)
            self.training_env.set_stage4_projectile_pressure(
                fire_rate_scale,
                first_shot_delay_steps,
                max_active_projectiles,
            )
            self.training_env.set_stage4_curriculum_level(
                self.stage4_projectile_speed_level + 1,
                len(STAGE4_PROJECTILE_SPEED_LEVELS),
            )

    def _stage4_reset_level_windows(self) -> None:
        self.stage4_level_success_window.clear()
        self.stage4_level_metric_window.clear()
        self.stage4_level_damage_window.clear()
        self.stage4_level_death_window.clear()

    def _stage4_record_level_stats(self, info: dict[str, Any], success: int) -> dict[str, float]:
        self.stage4_level_success_window.append(int(success))
        try:
            self.stage4_level_metric_window.append(float(info.get("stage_success_metric", 0.0)))
        except (TypeError, ValueError):
            self.stage4_level_metric_window.append(0.0)
        try:
            self.stage4_level_damage_window.append(float(info.get("damage_taken", 0.0)))
        except (TypeError, ValueError):
            self.stage4_level_damage_window.append(0.0)
        self.stage4_level_death_window.append(1 if float(info.get("deaths", 0.0) or 0.0) > 0.0 else 0)
        return self._stage4_level_stats()

    def _stage4_level_stats(self) -> dict[str, float]:
        success_count = len(self.stage4_level_success_window)
        metric_count = len(self.stage4_level_metric_window)
        damage_count = len(self.stage4_level_damage_window)
        death_count = len(self.stage4_level_death_window)
        return {
            "window_count": float(min(success_count, metric_count, damage_count, death_count)),
            "success_rate": sum(self.stage4_level_success_window) / max(1, success_count),
            "metric": sum(self.stage4_level_metric_window) / max(1, metric_count),
            "avg_damage": sum(self.stage4_level_damage_window) / max(1, damage_count),
            "death_rate": sum(self.stage4_level_death_window) / max(1, death_count),
        }

    def _stage4_level_score(self, stats: dict[str, float]) -> float:
        damage_score = max(0.0, min(1.0, 1.0 - stats.get("avg_damage", 105.0) / 105.0))
        survival_score = max(0.0, min(1.0, 1.0 - stats.get("death_rate", 1.0)))
        return (
            2.0 * stats.get("success_rate", 0.0)
            + stats.get("metric", 0.0)
            + 0.35 * survival_score
            + 0.25 * damage_score
        )

    def _stage4_level_checkpoint_paths(self, level_index: int) -> tuple[Path, Path]:
        base_dir = Path(self.checkpoint_dir) / self.stage.stage_id if self.checkpoint_dir is not None else self.stage.checkpoint_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        level_number = level_index + 1
        return (
            base_dir / f"best_stage4_level_{level_number:02d}.pt",
            base_dir / f"best_stage4_level_{level_number:02d}.zip",
        )

    def _save_stage4_level_best_if_needed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage4":
            return
        if stats.get("window_count", 0.0) < min(10, STAGE4_CURRICULUM_WINDOW):
            return
        level_key = str(self.stage4_projectile_speed_level)
        score = self._stage4_level_score(stats)
        best_score = float(self.stage4_best_level_scores.get(level_key, -1.0))
        if score <= best_score + 1e-4:
            return
        self.stage4_best_level_scores[level_key] = score
        state_path, model_path = self._stage4_level_checkpoint_paths(self.stage4_projectile_speed_level)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=f"stage4_level_{self.stage4_projectile_speed_level + 1:02d}_best",
            )
        )
        print(
            f"[CURRICULUM stage4] Saved best model for level {self.stage4_projectile_speed_level + 1} "
            f"(score {score:.3f}, success {stats.get('success_rate', 0.0):.2%}, "
            f"metric {stats.get('metric', 0.0):.3f}, death {stats.get('death_rate', 0.0):.2%}, "
            f"damage {stats.get('avg_damage', 0.0):.1f})."
        )

    def _maybe_rollback_stage4_projectile_speed(self, stats: dict[str, float]) -> bool:
        if self.stage.stage_id != "stage4":
            return False
        if self.stage4_projectile_speed_level <= 0:
            return False
        if self.stage4_episodes_at_projectile_level < STAGE4_ROLLBACK_MIN_EPISODES:
            return False
        if stats.get("window_count", 0.0) < STAGE4_CURRICULUM_WINDOW:
            return False

        current = STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]
        rollback_death_rate = float(current.get("rollback_death_rate", 0.80))
        rollback_max_success = float(current.get("rollback_max_success", 0.03))
        rollback_min_damage = float(current.get("rollback_min_damage", 95.0))

        collapse = (
            stats.get("death_rate", 0.0) + 1e-9 >= rollback_death_rate
            and stats.get("success_rate", 0.0) <= rollback_max_success + 1e-9
            and stats.get("avg_damage", 0.0) + 1e-9 >= rollback_min_damage
        )
        if not collapse:
            return False

        old_level_number = self.stage4_projectile_speed_level + 1
        self.stage4_projectile_speed_level -= 1
        self.stage4_episodes_at_projectile_level = 0
        self.stage4_curriculum_rollbacks += 1
        self._stage4_reset_level_windows()
        self._apply_stage4_projectile_speed_scale()
        new_level = STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]
        print(
            f"[CURRICULUM stage4] Rollback level {old_level_number} -> {self.stage4_projectile_speed_level + 1} "
            f"because recent window collapsed "
            f"(success {stats.get('success_rate', 0.0):.2%}, death {stats.get('death_rate', 0.0):.2%}, "
            f"damage {stats.get('avg_damage', 0.0):.1f}). "
            f"New speed {float(new_level.get('scale', 1.0)):.0%}, "
            f"fire-rate {float(new_level.get('fire_rate_scale', 1.0)):.0%}."
        )
        return True

    def _maybe_increase_stage4_projectile_speed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage4":
            return
        if self.stage4_projectile_speed_level >= len(STAGE4_PROJECTILE_SPEED_LEVELS) - 1:
            return

        current = STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]
        if self.stage4_episodes_at_projectile_level < int(current["min_episodes"]):
            return
        if stats.get("window_count", 0.0) < STAGE4_CURRICULUM_WINDOW:
            return

        required_success = float(current.get("required_success", 1.01))
        required_metric = float(current.get("required_metric", 1.01))
        max_death_rate = float(current.get("max_death_rate", 1.0))
        max_avg_damage = float(current.get("max_avg_damage", 105.0))

        enough_success = stats.get("success_rate", 0.0) + 1e-9 >= required_success
        enough_metric = stats.get("metric", 0.0) + 1e-9 >= required_metric
        safe_enough = stats.get("death_rate", 1.0) <= max_death_rate + 1e-9
        damage_ok = stats.get("avg_damage", 105.0) <= max_avg_damage + 1e-9
        if not (enough_success and enough_metric and safe_enough and damage_ok):
            return

        old_scale = self._stage4_current_projectile_speed_scale()
        old_level_number = self.stage4_projectile_speed_level + 1
        self.stage4_projectile_speed_level += 1
        self.stage4_episodes_at_projectile_level = 0
        self._stage4_reset_level_windows()
        new_scale = self._stage4_current_projectile_speed_scale()
        new_level = STAGE4_PROJECTILE_SPEED_LEVELS[self.stage4_projectile_speed_level]
        self._apply_stage4_projectile_speed_scale()
        print(
            f"[CURRICULUM stage4] Projectile pressure increased "
            f"level {old_level_number} -> {self.stage4_projectile_speed_level + 1}, "
            f"speed {old_scale:.0%} -> {new_scale:.0%}, "
            f"fire-rate {float(new_level.get('fire_rate_scale', 1.0)):.0%}, "
            f"active-projectiles {int(new_level.get('max_active_projectiles', 99))}; "
            f"stable window success {stats.get('success_rate', 0.0):.2%}, "
            f"metric {stats.get('metric', 0.0):.3f}, "
            f"death {stats.get('death_rate', 0.0):.2%}, "
            f"damage {stats.get('avg_damage', 0.0):.1f}."
        )

    def _maybe_update_stage4_curriculum(self, stats: dict[str, float]) -> None:
        self._save_stage4_level_best_if_needed(stats)
        if self._maybe_rollback_stage4_projectile_speed(stats):
            return
        self._maybe_increase_stage4_projectile_speed(stats)
    def _stage5_current_phase(self) -> dict[str, Any]:
        return STAGE5_CURRICULUM_PHASES[self.stage5_curriculum_phase]

    def _apply_stage5_curriculum_phase(self) -> None:
        if self.stage.stage_id != "stage5":
            return
        phase = self._stage5_current_phase()
        kwargs = dict(
            level=self.stage5_curriculum_phase + 1,
            total_levels=len(STAGE5_CURRICULUM_PHASES),
            phase_index=int(phase.get("phase_index", self.stage5_curriculum_phase + 1)),
            enemy_can_move=bool(phase.get("enemy_can_move", True)),
            enemy_can_shoot=bool(phase.get("enemy_can_shoot", True)),
            fire_rate_scale=float(phase.get("fire_rate_scale", 1.0)),
            first_shot_delay_steps=int(phase.get("first_shot_delay_steps", 0)),
            max_active_enemy_projectiles=int(phase.get("max_active_projectiles", 99)),
        )
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_stage5_curriculum_phase", **kwargs)
        else:
            try:
                self.training_env.unwrapped.set_stage5_curriculum_phase(**kwargs)
            except AttributeError:
                self.training_env.set_stage5_curriculum_phase(**kwargs)
        if self.eval_env is not None and hasattr(self.eval_env, "set_stage5_curriculum_phase"):
            self.eval_env.set_stage5_curriculum_phase(**kwargs)

    def _stage5_reset_phase_windows(self) -> None:
        self.stage5_phase_success_window.clear()
        self.stage5_phase_metric_window.clear()
        self.stage5_phase_death_window.clear()
        self.stage5_phase_damage_dealt_window.clear()

    def _stage5_record_phase_stats(self, info: dict[str, Any], success: int) -> dict[str, float]:
        self.stage5_phase_success_window.append(int(success))
        try:
            self.stage5_phase_metric_window.append(float(info.get("stage_success_metric", 0.0)))
        except (TypeError, ValueError):
            self.stage5_phase_metric_window.append(0.0)
        self.stage5_phase_death_window.append(1 if float(info.get("deaths", 0.0) or 0.0) > 0.0 else 0)
        try:
            self.stage5_phase_damage_dealt_window.append(float(info.get("damage_dealt", 0.0)))
        except (TypeError, ValueError):
            self.stage5_phase_damage_dealt_window.append(0.0)
        return self._stage5_phase_stats()

    def _stage5_phase_stats(self) -> dict[str, float]:
        success_count = len(self.stage5_phase_success_window)
        metric_count = len(self.stage5_phase_metric_window)
        death_count = len(self.stage5_phase_death_window)
        damage_count = len(self.stage5_phase_damage_dealt_window)
        return {
            "window_count": float(min(success_count, metric_count, death_count, damage_count)),
            "success_rate": sum(self.stage5_phase_success_window) / max(1, success_count),
            "metric": sum(self.stage5_phase_metric_window) / max(1, metric_count),
            "death_rate": sum(self.stage5_phase_death_window) / max(1, death_count),
            "avg_damage_dealt": sum(self.stage5_phase_damage_dealt_window) / max(1, damage_count),
        }

    def _stage5_phase_score(self, stats: dict[str, float]) -> float:
        return (
            2.0 * stats.get("success_rate", 0.0)
            + stats.get("metric", 0.0)
            + 0.40 * min(1.0, stats.get("avg_damage_dealt", 0.0) / 90.0)
            + 0.25 * max(0.0, 1.0 - stats.get("death_rate", 1.0))
        )

    def _stage5_phase_checkpoint_paths(self, phase_index: int) -> tuple[Path, Path]:
        base_dir = Path(self.checkpoint_dir) / self.stage.stage_id if self.checkpoint_dir is not None else self.stage.checkpoint_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        phase = str(STAGE5_CURRICULUM_PHASES[phase_index].get("phase", f"level_{phase_index + 1}")).lower()
        level_number = phase_index + 1
        return (
            base_dir / f"best_stage5_{phase}_level_{level_number:02d}.pt",
            base_dir / f"best_stage5_{phase}_level_{level_number:02d}.zip",
        )

    def _save_stage5_phase_best_if_needed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage5":
            return
        if stats.get("window_count", 0.0) < min(10, STAGE5_CURRICULUM_WINDOW):
            return
        phase_key = str(self.stage5_curriculum_phase)
        score = self._stage5_phase_score(stats)
        best_score = float(self.stage5_best_phase_scores.get(phase_key, -1.0))
        if score <= best_score + 1e-4:
            return
        self.stage5_best_phase_scores[phase_key] = score
        state_path, model_path = self._stage5_phase_checkpoint_paths(self.stage5_curriculum_phase)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=f"stage5_phase_{self.stage5_curriculum_phase + 1:02d}_best",
            )
        )
        phase = self._stage5_current_phase()
        print(
            f"[CURRICULUM stage5 {phase.get('phase', '5C')}] Saved best model for level {self.stage5_curriculum_phase + 1} "
            f"(score {score:.3f}, success {stats.get('success_rate', 0.0):.2%}, "
            f"metric {stats.get('metric', 0.0):.3f}, death {stats.get('death_rate', 0.0):.2%}, "
            f"damage_dealt {stats.get('avg_damage_dealt', 0.0):.1f})."
        )

    def _maybe_increase_stage5_phase(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage5":
            return
        self._save_stage5_phase_best_if_needed(stats)
        if self.stage5_curriculum_phase >= len(STAGE5_CURRICULUM_PHASES) - 1:
            return
        current = self._stage5_current_phase()
        if self.stage5_episodes_at_phase < int(current.get("min_episodes", 0)):
            return
        if stats.get("window_count", 0.0) < STAGE5_CURRICULUM_WINDOW:
            return
        if stats.get("success_rate", 0.0) + 1e-9 < float(current.get("required_success", 1.01)):
            return
        if stats.get("metric", 0.0) + 1e-9 < float(current.get("required_metric", 1.01)):
            return
        if stats.get("avg_damage_dealt", 0.0) + 1e-9 < float(current.get("min_avg_damage_dealt", 0.0)):
            return
        if stats.get("death_rate", 1.0) > float(current.get("max_death_rate", 1.0)) + 1e-9:
            return

        old_phase = str(current.get("phase", "5A"))
        old_level = self.stage5_curriculum_phase + 1
        self.stage5_curriculum_phase += 1
        self.stage5_episodes_at_phase = 0
        self._stage5_reset_phase_windows()
        self._apply_stage5_curriculum_phase()
        new_phase = self._stage5_current_phase()
        print(
            f"[CURRICULUM stage5] Promotion {old_phase} level {old_level} -> "
            f"{new_phase.get('phase', '5C')} level {self.stage5_curriculum_phase + 1}; "
            f"window success {stats.get('success_rate', 0.0):.2%}, metric {stats.get('metric', 0.0):.3f}, "
            f"damage_dealt {stats.get('avg_damage_dealt', 0.0):.1f}, death {stats.get('death_rate', 0.0):.2%}."
        )

    def _stage6_current_phase(self) -> dict[str, Any]:
        return STAGE6_CURRICULUM_PHASES[self.stage6_curriculum_phase]

    def _apply_stage6_curriculum_phase(self) -> None:
        if self.stage.stage_id != "stage6":
            return
        phase = self._stage6_current_phase()
        kwargs = dict(
            level=self.stage6_curriculum_phase + 1,
            total_levels=len(STAGE6_CURRICULUM_PHASES),
            phase_index=int(phase.get("phase_index", self.stage6_curriculum_phase + 1)),
            enemy_can_move=bool(phase.get("enemy_can_move", True)),
            enemy_can_shoot=bool(phase.get("enemy_can_shoot", True)),
            fire_rate_scale=float(phase.get("fire_rate_scale", 1.0)),
            first_shot_delay_steps=int(phase.get("first_shot_delay_steps", 0)),
            max_active_enemy_projectiles=int(phase.get("max_active_projectiles", 99)),
        )
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_stage6_curriculum_phase", **kwargs)
        else:
            try:
                self.training_env.unwrapped.set_stage6_curriculum_phase(**kwargs)
            except AttributeError:
                self.training_env.set_stage6_curriculum_phase(**kwargs)
        if self.eval_env is not None and hasattr(self.eval_env, "set_stage6_curriculum_phase"):
            self.eval_env.set_stage6_curriculum_phase(**kwargs)

    def _stage6_reset_phase_windows(self) -> None:
        self.stage6_phase_success_window.clear()
        self.stage6_phase_metric_window.clear()
        self.stage6_phase_death_window.clear()
        self.stage6_phase_damage_taken_window.clear()

    def _stage6_record_phase_stats(self, info: dict[str, Any], success: int) -> dict[str, float]:
        self.stage6_phase_success_window.append(int(success))
        try:
            self.stage6_phase_metric_window.append(float(info.get("stage_success_metric", 0.0)))
        except (TypeError, ValueError):
            self.stage6_phase_metric_window.append(0.0)
        self.stage6_phase_death_window.append(1 if float(info.get("deaths", 0.0) or 0.0) > 0.0 else 0)
        try:
            self.stage6_phase_damage_taken_window.append(float(info.get("damage_taken", 105.0)))
        except (TypeError, ValueError):
            self.stage6_phase_damage_taken_window.append(105.0)
        return self._stage6_phase_stats()

    def _stage6_phase_stats(self) -> dict[str, float]:
        success_count = len(self.stage6_phase_success_window)
        metric_count = len(self.stage6_phase_metric_window)
        death_count = len(self.stage6_phase_death_window)
        damage_count = len(self.stage6_phase_damage_taken_window)
        return {
            "window_count": float(min(success_count, metric_count, death_count, damage_count)),
            "success_rate": sum(self.stage6_phase_success_window) / max(1, success_count),
            "metric": sum(self.stage6_phase_metric_window) / max(1, metric_count),
            "death_rate": sum(self.stage6_phase_death_window) / max(1, death_count),
            "avg_damage_taken": sum(self.stage6_phase_damage_taken_window) / max(1, damage_count),
        }

    def _stage6_phase_score(self, stats: dict[str, float]) -> float:
        damage_score = max(0.0, min(1.0, 1.0 - stats.get("avg_damage_taken", 105.0) / 105.0))
        survival_score = max(0.0, min(1.0, 1.0 - stats.get("death_rate", 1.0)))
        return (
            2.0 * stats.get("success_rate", 0.0)
            + stats.get("metric", 0.0)
            + 0.40 * survival_score
            + 0.30 * damage_score
        )

    def _stage6_phase_checkpoint_paths(self, phase_index: int) -> tuple[Path, Path]:
        base_dir = Path(self.checkpoint_dir) / self.stage.stage_id if self.checkpoint_dir is not None else self.stage.checkpoint_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        phase = str(STAGE6_CURRICULUM_PHASES[phase_index].get("phase", f"level_{phase_index + 1}")).lower()
        level_number = phase_index + 1
        return (
            base_dir / f"best_stage6_{phase}_level_{level_number:02d}.pt",
            base_dir / f"best_stage6_{phase}_level_{level_number:02d}.zip",
        )

    def _save_stage6_phase_best_if_needed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage6":
            return
        if stats.get("window_count", 0.0) < min(10, STAGE6_CURRICULUM_WINDOW):
            return
        phase_key = str(self.stage6_curriculum_phase)
        score = self._stage6_phase_score(stats)
        best_score = float(self.stage6_best_phase_scores.get(phase_key, -1.0))
        if score <= best_score + 1e-4:
            return
        self.stage6_best_phase_scores[phase_key] = score
        state_path, model_path = self._stage6_phase_checkpoint_paths(self.stage6_curriculum_phase)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=f"stage6_phase_{self.stage6_curriculum_phase + 1:02d}_best",
            )
        )
        phase = self._stage6_current_phase()
        print(
            f"[CURRICULUM stage6 {phase.get('phase', '6C')}] Saved best model for level {self.stage6_curriculum_phase + 1} "
            f"(score {score:.3f}, success {stats.get('success_rate', 0.0):.2%}, "
            f"metric {stats.get('metric', 0.0):.3f}, death {stats.get('death_rate', 0.0):.2%}, "
            f"damage_taken {stats.get('avg_damage_taken', 0.0):.1f})."
        )

    def _maybe_save_stage6_candidate_checkpoint(self, *, curriculum_final_phase: bool) -> None:
        if self.stage.stage_id != "stage6" or not curriculum_final_phase:
            return
        if self.stage6_curriculum_phase < len(STAGE6_CURRICULUM_PHASES) - 1:
            return

        # Candidate checkpoints are not official best models. They preserve a
        # near-pass policy when Stage 6 repeatedly reaches the validation border
        # but misses certification because of one noisy evaluation batch.
        near_pass = (
            self.consecutive_good_eval_checks >= max(1, self.required_consecutive_checks - 1)
            or self.last_validation_score >= 0.85
            or self.last_eval_success_rate >= max(0.80, self.success_threshold + 0.10)
        )
        if not near_pass:
            return

        candidate_score = self.last_validation_score + 0.02 * float(self.consecutive_good_eval_checks)
        if candidate_score <= self.stage6_candidate_score + 1e-9:
            return

        self.stage6_candidate_score = candidate_score
        self._save_checkpoint(reason="candidate")
        print(
            f"Stage 6 near-pass candidate saved: {self.last_checkpoint_path} "
            f"(validation_score={self.last_validation_score:.3f}, "
            f"rolling_success={self.last_eval_success_rate:.2%}, "
            f"good_checks={self.consecutive_good_eval_checks}/{self.required_consecutive_checks})."
        )

    def _maybe_increase_stage6_phase(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage6":
            return
        self._save_stage6_phase_best_if_needed(stats)
        if self.stage6_curriculum_phase >= len(STAGE6_CURRICULUM_PHASES) - 1:
            return
        current = self._stage6_current_phase()
        if self.stage6_episodes_at_phase < int(current.get("min_episodes", 0)):
            return
        if stats.get("window_count", 0.0) < STAGE6_CURRICULUM_WINDOW:
            return
        if stats.get("success_rate", 0.0) + 1e-9 < float(current.get("required_success", 1.01)):
            return
        if stats.get("metric", 0.0) + 1e-9 < float(current.get("required_metric", 1.01)):
            return
        if stats.get("death_rate", 1.0) > float(current.get("max_death_rate", 1.0)) + 1e-9:
            return
        if stats.get("avg_damage_taken", 105.0) > float(current.get("max_avg_damage_taken", 105.0)) + 1e-9:
            return

        old_phase = str(current.get("phase", "6A"))
        old_level = self.stage6_curriculum_phase + 1
        self.stage6_curriculum_phase += 1
        self.stage6_episodes_at_phase = 0
        self._stage6_reset_phase_windows()
        self._apply_stage6_curriculum_phase()
        new_phase = self._stage6_current_phase()
        print(
            f"[CURRICULUM stage6] Promotion {old_phase} level {old_level} -> "
            f"{new_phase.get('phase', '6C')} level {self.stage6_curriculum_phase + 1}; "
            f"window success {stats.get('success_rate', 0.0):.2%}, metric {stats.get('metric', 0.0):.3f}, "
            f"damage_taken {stats.get('avg_damage_taken', 0.0):.1f}, death {stats.get('death_rate', 0.0):.2%}."
        )

    def _stage7_current_phase(self) -> dict[str, Any]:
        return STAGE7_CURRICULUM_PHASES[self.stage7_curriculum_phase]

    def _apply_stage7_curriculum_phase(self) -> None:
        if self.stage.stage_id != "stage7":
            return
        phase = self._stage7_current_phase()
        kwargs = dict(
            level=self.stage7_curriculum_phase + 1,
            total_levels=len(STAGE7_CURRICULUM_PHASES),
            phase_index=int(phase.get("phase_index", self.stage7_curriculum_phase + 1)),
            max_steps=int(phase.get("max_steps", 1200)),
            obstacle_count=int(phase.get("obstacle_count", self.stage7_curriculum_phase + 1)),
            enemy_can_move=bool(phase.get("enemy_can_move", True)),
            enemy_can_shoot=bool(phase.get("enemy_can_shoot", True)),
            fire_rate_scale=float(phase.get("fire_rate_scale", 1.0)),
            first_shot_delay_steps=int(phase.get("first_shot_delay_steps", 0)),
            max_active_enemy_projectiles=int(phase.get("max_active_projectiles", 99)),
        )
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_stage7_curriculum_phase", **kwargs)
        else:
            try:
                self.training_env.unwrapped.set_stage7_curriculum_phase(**kwargs)
            except AttributeError:
                self.training_env.set_stage7_curriculum_phase(**kwargs)
        if self.eval_env is not None and hasattr(self.eval_env, "set_stage7_curriculum_phase"):
            self.eval_env.set_stage7_curriculum_phase(**kwargs)

    def _stage7_reset_phase_windows(self) -> None:
        self.stage7_phase_success_window.clear()
        self.stage7_phase_metric_window.clear()
        self.stage7_phase_death_window.clear()
        self.stage7_phase_blocked_rate_window.clear()
        self.stage7_phase_blocked_shots_window.clear()
        self.stage7_phase_damage_taken_window.clear()

    def _stage7_record_phase_stats(self, info: dict[str, Any], success: int) -> dict[str, float]:
        self.stage7_phase_success_window.append(int(success))
        try:
            self.stage7_phase_metric_window.append(float(info.get("stage_success_metric", 0.0)))
        except (TypeError, ValueError):
            self.stage7_phase_metric_window.append(0.0)
        self.stage7_phase_death_window.append(1 if float(info.get("deaths", 0.0) or 0.0) > 0.0 else 0)
        try:
            self.stage7_phase_blocked_rate_window.append(float(info.get("blocked_move_rate", 0.0)))
        except (TypeError, ValueError):
            self.stage7_phase_blocked_rate_window.append(0.0)
        try:
            self.stage7_phase_blocked_shots_window.append(float(info.get("shots_blocked", 0.0)))
        except (TypeError, ValueError):
            self.stage7_phase_blocked_shots_window.append(0.0)
        try:
            self.stage7_phase_damage_taken_window.append(float(info.get("damage_taken", 105.0)))
        except (TypeError, ValueError):
            self.stage7_phase_damage_taken_window.append(105.0)
        return self._stage7_phase_stats()

    def _stage7_phase_stats(self) -> dict[str, float]:
        success_count = len(self.stage7_phase_success_window)
        metric_count = len(self.stage7_phase_metric_window)
        death_count = len(self.stage7_phase_death_window)
        blocked_rate_count = len(self.stage7_phase_blocked_rate_window)
        blocked_shots_count = len(self.stage7_phase_blocked_shots_window)
        damage_count = len(self.stage7_phase_damage_taken_window)
        return {
            "window_count": float(success_count),
            "success_rate": sum(self.stage7_phase_success_window) / max(1, success_count),
            "metric": sum(self.stage7_phase_metric_window) / max(1, metric_count),
            "death_rate": sum(self.stage7_phase_death_window) / max(1, death_count),
            "avg_blocked_rate": sum(self.stage7_phase_blocked_rate_window) / max(1, blocked_rate_count),
            "avg_blocked_shots": sum(self.stage7_phase_blocked_shots_window) / max(1, blocked_shots_count),
            "avg_damage_taken": sum(self.stage7_phase_damage_taken_window) / max(1, damage_count),
        }

    def _stage7_phase_score(self, stats: dict[str, float]) -> float:
        blocked_rate_score = max(0.0, 1.0 - stats.get("avg_blocked_rate", 0.0) * 8.0)
        blocked_shot_score = max(0.0, 1.0 - stats.get("avg_blocked_shots", 0.0) / 14.0)
        survival_score = max(0.0, 1.0 - stats.get("death_rate", 1.0))
        damage_score = max(0.0, 1.0 - stats.get("avg_damage_taken", 105.0) / 105.0)
        return (
            0.35 * stats.get("success_rate", 0.0)
            + 0.25 * stats.get("metric", 0.0)
            + 0.15 * blocked_rate_score
            + 0.10 * blocked_shot_score
            + 0.10 * survival_score
            + 0.05 * damage_score
        )

    def _stage7_phase_checkpoint_paths(self, phase_index: int) -> tuple[Path, Path]:
        base_dir = self.stage.checkpoint_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        phase = str(STAGE7_CURRICULUM_PHASES[phase_index].get("phase", f"level_{phase_index + 1}")).lower()
        level_number = phase_index + 1
        return (
            base_dir / f"best_stage7_{phase}_level_{level_number:02d}.pt",
            base_dir / f"best_stage7_{phase}_level_{level_number:02d}.zip",
        )

    def _save_stage7_phase_best_if_needed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage7":
            return
        if self.model is None:
            return
        phase_key = str(self.stage7_curriculum_phase)
        score = self._stage7_phase_score(stats)
        best_score = float(self.stage7_best_phase_scores.get(phase_key, -1.0))
        if score <= best_score + 1e-9:
            return
        self.stage7_best_phase_scores[phase_key] = score
        state_path, model_path = self._stage7_phase_checkpoint_paths(self.stage7_curriculum_phase)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=f"stage7_phase_{self.stage7_curriculum_phase + 1:02d}_best",
            )
        )
        phase = self._stage7_current_phase()
        print(
            f"[CURRICULUM stage7 {phase.get('phase', '7E')}] Saved best model for level {self.stage7_curriculum_phase + 1} "
            f"with phase score {score:.3f}: {model_path}"
        )

    def _stage7_eval_gate_ready(self, phase: dict[str, Any]) -> bool:
        if self.stage.stage_id != "stage7":
            return True
        if self.stage7_eval_gate_phase != self.stage7_curriculum_phase:
            return False
        required_checks = int(phase.get("required_eval_checks", 1))
        return self.stage7_eval_gate_good_checks >= required_checks

    def _update_stage7_eval_gate(self, summary: dict[str, Any]) -> None:
        if self.stage.stage_id != "stage7":
            return
        if self.stage7_curriculum_phase >= len(STAGE7_CURRICULUM_PHASES) - 1:
            return

        phase = self._stage7_current_phase()
        if self.stage7_eval_gate_phase != self.stage7_curriculum_phase:
            self.stage7_eval_gate_phase = self.stage7_curriculum_phase
            self.stage7_eval_gate_good_checks = 0

        mean_metrics = dict(summary.get("mean_metrics", {}))
        success_rate = float(self.last_eval_success_rate)
        validation_score = float(summary.get("validation_score", self.last_validation_score))
        death_rate = float(mean_metrics.get("deaths", 1.0) or 0.0)
        blocked_rate = float(mean_metrics.get("blocked_move_rate", 1.0) or 0.0)
        blocked_shots = float(mean_metrics.get("shots_blocked", 999.0) or 0.0)
        movement = float(mean_metrics.get("movement_distance_total", 0.0) or 0.0)
        visible_ratio = float(mean_metrics.get("enemy_visible_ratio", 0.0) or 0.0)
        clear_shot_ratio = float(mean_metrics.get("enemy_clear_shot_ratio", 0.0) or 0.0)
        time_on_target = float(mean_metrics.get("time_on_target_ratio", 0.0) or 0.0)
        nav_progress = float(mean_metrics.get("nav_progress_ratio", 0.0) or 0.0)

        required_success = float(phase.get("required_eval_success", phase.get("required_success", 1.01)))
        required_metric = float(phase.get("required_eval_metric", phase.get("required_metric", 1.01)))
        max_death_rate = float(phase.get("max_eval_death_rate", 1.0))
        max_blocked_rate = float(phase.get("max_eval_blocked_rate", 1.0))
        max_blocked_shots = float(phase.get("max_avg_blocked_shots", 999.0))
        min_movement = float(phase.get("min_eval_movement", 0.0))
        min_visible_ratio = float(phase.get("min_eval_visible_ratio", 0.0))
        min_clear_shot_ratio = float(phase.get("min_eval_clear_shot_ratio", 0.0))
        min_time_on_target = float(phase.get("min_eval_time_on_target", 0.0))
        min_nav_progress = float(phase.get("min_eval_nav_progress", 0.0))

        ok = (
            success_rate + 1e-9 >= required_success
            and validation_score + 1e-9 >= required_metric
            and death_rate <= max_death_rate + 1e-9
            and blocked_rate <= max_blocked_rate + 1e-9
            and blocked_shots <= max_blocked_shots + 1e-9
            and movement + 1e-9 >= min_movement
            and visible_ratio + 1e-9 >= min_visible_ratio
            and clear_shot_ratio + 1e-9 >= min_clear_shot_ratio
            and time_on_target + 1e-9 >= min_time_on_target
            and nav_progress + 1e-9 >= min_nav_progress
        )
        if ok:
            self.stage7_eval_gate_good_checks += 1
        else:
            self.stage7_eval_gate_good_checks = 0

        self.stage7_last_eval_gate_summary = {
            "phase": phase.get("phase", "7A"),
            "ok": bool(ok),
            "checks": self.stage7_eval_gate_good_checks,
            "required_checks": int(phase.get("required_eval_checks", 1)),
            "success_rate": success_rate,
            "required_success": required_success,
            "validation_score": validation_score,
            "required_metric": required_metric,
            "death_rate": death_rate,
            "max_death_rate": max_death_rate,
            "blocked_rate": blocked_rate,
            "max_blocked_rate": max_blocked_rate,
            "blocked_shots": blocked_shots,
            "max_blocked_shots": max_blocked_shots,
            "movement": movement,
            "min_movement": min_movement,
            "visible_ratio": visible_ratio,
            "min_visible_ratio": min_visible_ratio,
            "clear_shot_ratio": clear_shot_ratio,
            "min_clear_shot_ratio": min_clear_shot_ratio,
            "time_on_target": time_on_target,
            "min_time_on_target": min_time_on_target,
            "nav_progress": nav_progress,
            "min_nav_progress": min_nav_progress,
        }
        summary["stage7_eval_gate"] = dict(self.stage7_last_eval_gate_summary)
        status = "passed" if ok else "blocked"
        print(
            f"[CURRICULUM stage7 {phase.get('phase', '7A')}] Eval gate {status}: "
            f"checks {self.stage7_eval_gate_good_checks}/{int(phase.get('required_eval_checks', 1))}, "
            f"success {success_rate:.2%}/{required_success:.2%}, score {validation_score:.3f}/{required_metric:.3f}, "
            f"death {death_rate:.2%}/{max_death_rate:.2%}, blocked {blocked_rate:.2%}/{max_blocked_rate:.2%}, "
            f"blocked_shots {blocked_shots:.1f}/{max_blocked_shots:.1f}, "
            f"move {movement:.1f}/{min_movement:.1f}, visible {visible_ratio:.2%}/{min_visible_ratio:.2%}, "
            f"clear {clear_shot_ratio:.2%}/{min_clear_shot_ratio:.2%}, aim {time_on_target:.2%}/{min_time_on_target:.2%}, "
            f"nav {nav_progress:.2%}/{min_nav_progress:.2%}."
        )

    def _maybe_increase_stage7_phase(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage7":
            return
        self._save_stage7_phase_best_if_needed(stats)
        if self.stage7_curriculum_phase >= len(STAGE7_CURRICULUM_PHASES) - 1:
            return
        current = self._stage7_current_phase()
        if self.stage7_episodes_at_phase < int(current.get("min_episodes", 0)):
            return
        if stats.get("window_count", 0.0) < STAGE7_CURRICULUM_WINDOW:
            return
        if stats.get("success_rate", 0.0) + 1e-9 < float(current.get("required_success", 1.01)):
            return
        if stats.get("metric", 0.0) + 1e-9 < float(current.get("required_metric", 1.01)):
            return
        if stats.get("death_rate", 1.0) > float(current.get("max_death_rate", 1.0)) + 1e-9:
            return
        if stats.get("avg_blocked_rate", 1.0) > float(current.get("max_avg_blocked_rate", 1.0)) + 1e-9:
            return
        if stats.get("avg_blocked_shots", 999.0) > float(current.get("max_avg_blocked_shots", 999.0)) + 1e-9:
            return
        if stats.get("avg_damage_taken", 105.0) > float(current.get("max_avg_damage_taken", 105.0)) + 1e-9:
            return
        if not self._stage7_eval_gate_ready(current):
            gate = self.stage7_last_eval_gate_summary or {}
            print(
                f"[CURRICULUM stage7 {current.get('phase', '7A')}] Training gate passed but eval gate is not ready; "
                f"eval checks {int(gate.get('checks', 0))}/{int(current.get('required_eval_checks', 1))}."
            )
            return

        old_phase = str(current.get("phase", "7A"))
        old_level = self.stage7_curriculum_phase + 1
        self.stage7_curriculum_phase += 1
        self.stage7_episodes_at_phase = 0
        self.stage7_eval_gate_phase = self.stage7_curriculum_phase
        self.stage7_eval_gate_good_checks = 0
        self.stage7_last_eval_gate_summary = {}
        self._stage7_reset_phase_windows()
        self._apply_stage7_curriculum_phase()
        new_phase = self._stage7_current_phase()
        print(
            f"[CURRICULUM stage7] Promotion {old_phase} level {old_level} -> "
            f"{new_phase.get('phase', '7E')} level {self.stage7_curriculum_phase + 1}; "
            f"window success {stats.get('success_rate', 0.0):.2%}, metric {stats.get('metric', 0.0):.3f}, "
            f"blocked_rate {stats.get('avg_blocked_rate', 0.0):.2%}, blocked_shots {stats.get('avg_blocked_shots', 0.0):.1f}, "
            f"damage_taken {stats.get('avg_damage_taken', 0.0):.1f}, death {stats.get('death_rate', 0.0):.2%}."
        )


    def _stage8_current_phase(self) -> dict[str, Any]:
        return STAGE8_CURRICULUM_PHASES[self.stage8_curriculum_phase]

    def _apply_stage8_curriculum_phase(self) -> None:
        if self.stage.stage_id != "stage8":
            return
        phase = self._stage8_current_phase()
        kwargs = dict(
            level=self.stage8_curriculum_phase + 1,
            total_levels=len(STAGE8_CURRICULUM_PHASES),
            phase_index=int(phase.get("phase_index", self.stage8_curriculum_phase + 1)),
            max_steps=int(phase.get("max_steps", 900)),
            obstacle_count=int(phase.get("obstacle_count", 0)),
            enemy_can_move=bool(phase.get("enemy_can_move", False)),
            enemy_can_shoot=bool(phase.get("enemy_can_shoot", False)),
            fire_rate_scale=float(phase.get("fire_rate_scale", 0.0)),
            first_shot_delay_steps=int(phase.get("first_shot_delay_steps", 9999)),
            max_active_enemy_projectiles=int(phase.get("max_active_projectiles", 0)),
            enemy_waves=int(phase.get("enemy_waves", 5)),
        )
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_stage8_curriculum_phase", **kwargs)
        else:
            try:
                self.training_env.unwrapped.set_stage8_curriculum_phase(**kwargs)
            except AttributeError:
                self.training_env.set_stage8_curriculum_phase(**kwargs)
        if self.eval_env is not None and hasattr(self.eval_env, "set_stage8_curriculum_phase"):
            self.eval_env.set_stage8_curriculum_phase(**kwargs)

    def _stage8_reset_phase_windows(self) -> None:
        self.stage8_phase_success_window.clear()
        self.stage8_phase_metric_window.clear()
        self.stage8_phase_death_window.clear()
        self.stage8_phase_reload_quality_window.clear()
        self.stage8_phase_early_reload_window.clear()
        self.stage8_phase_empty_ammo_window.clear()

    def _stage8_record_phase_stats(self, info: dict[str, Any], success: int) -> dict[str, float]:
        self.stage8_phase_success_window.append(int(success))
        try: self.stage8_phase_metric_window.append(float(info.get("stage_success_metric", 0.0)))
        except (TypeError, ValueError): self.stage8_phase_metric_window.append(0.0)
        self.stage8_phase_death_window.append(1 if float(info.get("deaths", 0.0) or 0.0) > 0.0 else 0)
        try: self.stage8_phase_reload_quality_window.append(float(info.get("stage8_reload_quality", 0.0)))
        except (TypeError, ValueError): self.stage8_phase_reload_quality_window.append(0.0)
        try: self.stage8_phase_early_reload_window.append(float(info.get("stage8_early_reloads", 0.0)))
        except (TypeError, ValueError): self.stage8_phase_early_reload_window.append(0.0)
        try: self.stage8_phase_empty_ammo_window.append(float(info.get("stage8_empty_ammo_steps", 0.0)))
        except (TypeError, ValueError): self.stage8_phase_empty_ammo_window.append(0.0)
        return self._stage8_phase_stats()

    def _stage8_phase_stats(self) -> dict[str, float]:
        count = len(self.stage8_phase_success_window)
        return {
            "window_count": float(count),
            "success_rate": sum(self.stage8_phase_success_window) / max(1, count),
            "metric": sum(self.stage8_phase_metric_window) / max(1, len(self.stage8_phase_metric_window)),
            "death_rate": sum(self.stage8_phase_death_window) / max(1, len(self.stage8_phase_death_window)),
            "reload_quality": sum(self.stage8_phase_reload_quality_window) / max(1, len(self.stage8_phase_reload_quality_window)),
            "avg_early_reloads": sum(self.stage8_phase_early_reload_window) / max(1, len(self.stage8_phase_early_reload_window)),
            "avg_empty_ammo_steps": sum(self.stage8_phase_empty_ammo_window) / max(1, len(self.stage8_phase_empty_ammo_window)),
        }

    def _stage8_phase_score(self, stats: dict[str, float]) -> float:
        survival_score = max(0.0, 1.0 - stats.get("death_rate", 1.0))
        early_score = max(0.0, 1.0 - stats.get("avg_early_reloads", 0.0) / 2.0)
        empty_score = max(0.0, 1.0 - stats.get("avg_empty_ammo_steps", 0.0) / 200.0)
        return (
            0.30 * stats.get("success_rate", 0.0)
            + 0.25 * stats.get("metric", 0.0)
            + 0.20 * stats.get("reload_quality", 0.0)
            + 0.10 * survival_score
            + 0.08 * early_score
            + 0.07 * empty_score
        )

    def _stage8_phase_checkpoint_paths(self, phase_index: int) -> tuple[Path, Path]:
        base_dir = self.stage.checkpoint_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        phase = str(STAGE8_CURRICULUM_PHASES[phase_index].get("phase", f"level_{phase_index + 1}")).lower()
        level_number = phase_index + 1
        return (
            base_dir / f"best_stage8_{phase}_level_{level_number:02d}.pt",
            base_dir / f"best_stage8_{phase}_level_{level_number:02d}.zip",
        )

    def _save_stage8_phase_best_if_needed(self, stats: dict[str, float]) -> None:
        if self.stage.stage_id != "stage8" or self.model is None:
            return
        phase_key = str(self.stage8_curriculum_phase)
        score = self._stage8_phase_score(stats)
        best_score = float(self.stage8_best_phase_scores.get(phase_key, -1.0))
        if score <= best_score + 1e-9:
            return
        self.stage8_best_phase_scores[phase_key] = score
        state_path, model_path = self._stage8_phase_checkpoint_paths(self.stage8_curriculum_phase)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=f"stage8_phase_{self.stage8_curriculum_phase + 1:02d}_best",
            )
        )
        phase = self._stage8_current_phase()
        print(
            f"[CURRICULUM stage8 {phase.get('phase', '8E')}] Saved best model for level {self.stage8_curriculum_phase + 1} "
            f"with phase score {score:.3f}: {model_path}"
        )

    def _stage8_eval_gate_ready(self, phase: dict[str, Any]) -> bool:
        if self.stage.stage_id != "stage8":
            return True
        if self.stage8_eval_gate_phase != self.stage8_curriculum_phase:
            return False
        required_checks = int(phase.get("required_eval_checks", self.required_consecutive_checks))
        return self.stage8_eval_gate_good_checks >= required_checks

    def _update_stage8_eval_gate(self, summary: dict[str, Any]) -> None:
        if self.stage.stage_id != "stage8":
            return

        phase = self._stage8_current_phase()
        if self.stage8_eval_gate_phase != self.stage8_curriculum_phase:
            self.stage8_eval_gate_phase = self.stage8_curriculum_phase
            self.stage8_eval_gate_good_checks = 0

        decision = evaluate_stage8_phase(phase, summary, default_required_checks=self.required_consecutive_checks)
        if decision.passed:
            self.stage8_eval_gate_good_checks += 1
        else:
            self.stage8_eval_gate_good_checks = 0

        gate_summary = decision.to_dict()
        gate_summary["phase_name"] = decision.phase
        gate_summary["phase_index"] = decision.phase_index
        gate_summary["phase_passed"] = decision.passed
        gate_summary["eval_reason"] = "; ".join(decision.fail_reasons if decision.fail_reasons else decision.pass_reasons)
        gate_summary["checks"] = self.stage8_eval_gate_good_checks
        gate_summary["required_checks"] = decision.checks_required
        gate_summary["summary_text"] = decision.summary_text
        gate_summary["last_promotion_event"] = self.stage8_last_promotion_event
        self.stage8_last_eval_gate_summary = gate_summary
        summary["stage8_eval_gate"] = dict(gate_summary)

        status = "passed" if decision.passed else "blocked"
        reasons = gate_summary["eval_reason"]
        print(
            f"[CURRICULUM stage8 {decision.phase}] Eval gate {status}: "
            f"passed={decision.passed} | checks {self.stage8_eval_gate_good_checks}/{decision.checks_required} | "
            f"score {decision.validation_score:.3f}/{decision.details.get('required_metric', 0.0):.3f} | "
            f"success {decision.success_rate:.2%}/{decision.details.get('required_success', 0.0):.2%} | "
            f"reloadQ {decision.details.get('reload_quality', 0.0):.3f}/{decision.details.get('min_reload_quality', 0.0):.3f} | "
            f"early {decision.details.get('avg_early_reloads', 0.0):.2f}/{decision.details.get('max_avg_early_reloads', 0.0):.2f} | "
            f"empty {decision.details.get('avg_empty_ammo_steps', 0.0):.1f}/{decision.details.get('max_avg_empty_ammo_steps', 0.0):.1f} | "
            f"noLOS {decision.details.get('avg_no_los_shot_rate', 0.0):.2%}/{decision.details.get('max_avg_no_los_shot_rate', 1.0):.2%} | "
            f"blind {decision.details.get('avg_blind_rotation_steps', 0.0):.1f}/{decision.details.get('max_avg_blind_rotation_steps', 9999.0):.1f} | "
            f"close {decision.details.get('avg_too_close_steps', 0.0):.1f}/{decision.details.get('max_avg_too_close_steps', 9999.0):.1f} | "
            f"death {decision.details.get('death_rate', 0.0):.2%}/{decision.details.get('max_death_rate', 0.0):.2%} | "
            f"reasons: {reasons}"
        )

    def _maybe_promote_stage8_phase_after_eval(self) -> None:
        if self.stage.stage_id != "stage8":
            return
        if self.stage8_curriculum_phase >= len(STAGE8_CURRICULUM_PHASES) - 1:
            return

        current = self._stage8_current_phase()
        if self.stage8_episodes_at_phase < int(current.get("min_episodes", 0)):
            return
        if not self._stage8_eval_gate_ready(current):
            return

        old_phase_index = self.stage8_curriculum_phase
        old_phase = str(current.get("phase", "8A"))
        old_level = self.stage8_curriculum_phase + 1
        gate_summary = dict(self.stage8_last_eval_gate_summary)

        self.stage8_curriculum_phase += 1
        self.stage8_episodes_at_phase = 0
        self._stage8_reset_phase_windows()
        self.stage8_eval_gate_phase = self.stage8_curriculum_phase
        self.stage8_eval_gate_good_checks = 0
        self.stage8_last_eval_gate_summary = {}
        self._apply_stage8_curriculum_phase()
        new_phase = self._stage8_current_phase()
        self.stage8_last_promotion_event = (
            f"PROMOTE {old_phase} -> {new_phase.get('phase', '8E')} at eval_check {self.eval_check_counter}"
        )

        if self.stage8_curriculum_phase == old_phase_index:
            raise RuntimeError(
                f"PROMOTION BUG: Stage 8 phase {old_phase} satisfied promotion rule but phase did not advance."
            )

        print(
            f"[CURRICULUM stage8] Promotion {old_phase} level {old_level} -> "
            f"{new_phase.get('phase', '8E')} level {self.stage8_curriculum_phase + 1}; "
            f"checks {gate_summary.get('checks', 0)}/{gate_summary.get('required_checks', 0)} | "
            f"score {gate_summary.get('validation_score', 0.0):.3f} | success {gate_summary.get('success_rate', 0.0):.2%}."
        )

    def _save_stage8_phase_best(self, stats: dict[str, float]) -> None:
        self._save_stage8_phase_best_if_needed(stats)

    def _open_csv_writers(self, append: bool) -> None:
        self.stage.log_dir.mkdir(parents=True, exist_ok=True)
        train_exists = self.stage.train_log_path.exists()
        eval_exists = self.stage.eval_log_path.exists()
        mode = "a" if append else "w"
        self.train_log_file = self.stage.train_log_path.open(mode, newline="", encoding="utf-8")
        self.eval_log_file = self.stage.eval_log_path.open(mode, newline="", encoding="utf-8")

        self.train_writer = csv.DictWriter(self.train_log_file, fieldnames=TRAIN_LOG_FIELDS)
        self.eval_writer = csv.DictWriter(self.eval_log_file, fieldnames=EVAL_LOG_FIELDS)

        if not append or not train_exists:
            self.train_writer.writeheader()
        if not append or not eval_exists:
            self.eval_writer.writeheader()

    def _on_step(self) -> bool:
        if self.success_reached:
            return False

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos and dones is not None:
            for done, info in zip(dones, infos):
                if done and isinstance(info, dict):
                    self._log_training_episode(info)

        if self._consume_visual_save_request():
            self._save_checkpoint(reason="manual_visual")
            print(f"Manual visual checkpoint saved: {self.last_checkpoint_path}")

        if self.stage.stage_id == "stage7":
            self._maybe_print_stage7_progress()

        if self.num_timesteps > 0 and (self.num_timesteps - self.last_eval_timestep) >= self.eval_every_steps:
            self.last_eval_timestep = int(self.num_timesteps)
            return self._run_evaluation_check()

        return True

    def _maybe_print_stage7_progress(self) -> None:
        if self.stage.stage_id != "stage7":
            return
        if int(self.num_timesteps) < int(self._stage7_next_progress_print_timestep):
            return
        phase = self._stage7_current_phase()
        current_timestep = int(self.num_timesteps)
        next_eval_in = max(0, int(self.eval_every_steps) - (current_timestep - int(self.last_eval_timestep)))
        episode_step = "?"
        envs = getattr(self.training_env, "envs", None)
        if envs:
            try:
                raw_step = getattr(envs[0].unwrapped, "_step_count", "?")
                raw_limit = envs[0].unwrapped._effective_max_steps()
                episode_step = f"{raw_step}/{raw_limit}"
            except Exception:
                episode_step = "?"
        else:
            try:
                raw_step = getattr(self.training_env.unwrapped, "_step_count", "?")
                raw_limit = self.training_env.unwrapped._effective_max_steps()
                episode_step = f"{raw_step}/{raw_limit}"
            except Exception:
                episode_step = "?"
        print(
            f"[PROGRESS stage7] timestep {current_timestep} | "
            f"phase {phase.get('phase', '7A')} | "
            f"episode_step {episode_step} | "
            f"episodes {self.episode_counter} | "
            f"next_eval_in {next_eval_in} steps"
        )
        while int(self._stage7_next_progress_print_timestep) <= current_timestep:
            self._stage7_next_progress_print_timestep += int(self._stage7_progress_interval)

    def _consume_visual_save_request(self) -> bool:
        envs = getattr(self.training_env, "envs", [])
        for env in envs:
            try:
                if env.unwrapped.consume_save_request():
                    return True
            except AttributeError:
                continue
        return False

    def _log_training_episode(self, info: dict[str, Any]) -> None:
        self.episode_counter += 1
        success = 1 if info.get("success", False) else 0
        self.train_success_window.append(success)
        rolling_success = sum(self.train_success_window) / len(self.train_success_window)

        stage4_stats: dict[str, float] = {}
        if self.stage.stage_id == "stage4":
            self.stage4_episodes_at_projectile_level += 1
            stage4_stats = self._stage4_record_level_stats(info, success)
            info["stage4_curriculum_level"] = self.stage4_projectile_speed_level + 1
            info["stage4_curriculum_total_levels"] = len(STAGE4_PROJECTILE_SPEED_LEVELS)
            info["stage4_episodes_at_curriculum_level"] = self.stage4_episodes_at_projectile_level
            info["stage4_curriculum_rolling_success"] = stage4_stats.get("success_rate", 0.0)
            info["stage4_curriculum_rolling_metric"] = stage4_stats.get("metric", 0.0)
            info["stage4_curriculum_rolling_death_rate"] = stage4_stats.get("death_rate", 0.0)
            info["stage4_curriculum_rolling_damage"] = stage4_stats.get("avg_damage", 0.0)
            info["stage4_curriculum_rollbacks"] = self.stage4_curriculum_rollbacks

        if self.stage.stage_id == "stage5":
            self.stage5_episodes_at_phase += 1
            stage5_stats = self._stage5_record_phase_stats(info, success)
            phase = self._stage5_current_phase()
            info["stage5_curriculum_level"] = self.stage5_curriculum_phase + 1
            info["stage5_curriculum_total_levels"] = len(STAGE5_CURRICULUM_PHASES)
            info["stage5_curriculum_phase_index"] = int(phase.get("phase_index", self.stage5_curriculum_phase + 1))
            info["stage5_episodes_at_curriculum_level"] = self.stage5_episodes_at_phase
            info["stage5_enemy_fire_rate_scale"] = float(phase.get("fire_rate_scale", 1.0))
            info["stage5_first_shot_delay_steps"] = int(phase.get("first_shot_delay_steps", 0))
            info["stage5_max_active_enemy_projectiles"] = int(phase.get("max_active_projectiles", 99))
            info["stage5_enemy_can_move"] = 1 if bool(phase.get("enemy_can_move", True)) else 0
            info["stage5_enemy_can_shoot"] = 1 if bool(phase.get("enemy_can_shoot", True)) else 0
            info["stage5_curriculum_rolling_success"] = stage5_stats.get("success_rate", 0.0)
            info["stage5_curriculum_rolling_metric"] = stage5_stats.get("metric", 0.0)
            info["stage5_curriculum_rolling_death_rate"] = stage5_stats.get("death_rate", 0.0)
            info["stage5_curriculum_rolling_damage_dealt"] = stage5_stats.get("avg_damage_dealt", 0.0)

        if self.stage.stage_id == "stage6":
            self.stage6_episodes_at_phase += 1
            stage6_stats = self._stage6_record_phase_stats(info, success)
            phase = self._stage6_current_phase()
            info["stage6_curriculum_level"] = self.stage6_curriculum_phase + 1
            info["stage6_curriculum_total_levels"] = len(STAGE6_CURRICULUM_PHASES)
            info["stage6_curriculum_phase_index"] = int(phase.get("phase_index", self.stage6_curriculum_phase + 1))
            info["stage6_episodes_at_curriculum_level"] = self.stage6_episodes_at_phase
            info["stage6_enemy_fire_rate_scale"] = float(phase.get("fire_rate_scale", 1.0))
            info["stage6_first_shot_delay_steps"] = int(phase.get("first_shot_delay_steps", 0))
            info["stage6_max_active_enemy_projectiles"] = int(phase.get("max_active_projectiles", 99))
            info["stage6_enemy_can_move"] = 1 if bool(phase.get("enemy_can_move", True)) else 0
            info["stage6_enemy_can_shoot"] = 1 if bool(phase.get("enemy_can_shoot", True)) else 0
            info["stage6_curriculum_rolling_success"] = stage6_stats.get("success_rate", 0.0)
            info["stage6_curriculum_rolling_metric"] = stage6_stats.get("metric", 0.0)
            info["stage6_curriculum_rolling_death_rate"] = stage6_stats.get("death_rate", 0.0)
            info["stage6_curriculum_rolling_damage_taken"] = stage6_stats.get("avg_damage_taken", 0.0)

        if self.stage.stage_id == "stage7":
            self.stage7_episodes_at_phase += 1
            stage7_stats = self._stage7_record_phase_stats(info, success)
            phase = self._stage7_current_phase()
            info["stage7_curriculum_level"] = self.stage7_curriculum_phase + 1
            info["stage7_curriculum_total_levels"] = len(STAGE7_CURRICULUM_PHASES)
            info["stage7_curriculum_phase_index"] = int(phase.get("phase_index", self.stage7_curriculum_phase + 1))
            info["stage7_max_steps"] = int(phase.get("max_steps", 1200))
            info["stage7_episodes_at_curriculum_level"] = self.stage7_episodes_at_phase
            info["stage7_obstacle_count"] = int(phase.get("obstacle_count", 1))
            info["stage7_enemy_fire_rate_scale"] = float(phase.get("fire_rate_scale", 1.0))
            info["stage7_first_shot_delay_steps"] = int(phase.get("first_shot_delay_steps", 0))
            info["stage7_max_active_enemy_projectiles"] = int(phase.get("max_active_projectiles", 99))
            info["stage7_enemy_can_move"] = 1 if bool(phase.get("enemy_can_move", True)) else 0
            info["stage7_enemy_can_shoot"] = 1 if bool(phase.get("enemy_can_shoot", True)) else 0
            info["stage7_curriculum_rolling_success"] = stage7_stats.get("success_rate", 0.0)
            info["stage7_curriculum_rolling_metric"] = stage7_stats.get("metric", 0.0)
            info["stage7_curriculum_rolling_death_rate"] = stage7_stats.get("death_rate", 0.0)
            info["stage7_curriculum_rolling_blocked_rate"] = stage7_stats.get("avg_blocked_rate", 0.0)
            info["stage7_curriculum_rolling_blocked_shots"] = stage7_stats.get("avg_blocked_shots", 0.0)
            info["stage7_curriculum_rolling_damage_taken"] = stage7_stats.get("avg_damage_taken", 0.0)

        if self.stage.stage_id == "stage8":
            self.stage8_episodes_at_phase += 1
            stage8_stats = self._stage8_record_phase_stats(info, success)
            phase = self._stage8_current_phase()
            info["stage8_curriculum_level"] = self.stage8_curriculum_phase + 1
            info["stage8_curriculum_total_levels"] = len(STAGE8_CURRICULUM_PHASES)
            info["stage8_curriculum_phase_index"] = int(phase.get("phase_index", self.stage8_curriculum_phase + 1))
            info["stage8_max_steps"] = int(phase.get("max_steps", 900))
            info["stage8_episodes_at_curriculum_level"] = self.stage8_episodes_at_phase
            info["stage8_obstacle_count"] = int(phase.get("obstacle_count", 0))
            info["stage8_enemy_fire_rate_scale"] = float(phase.get("fire_rate_scale", 0.0))
            info["stage8_first_shot_delay_steps"] = int(phase.get("first_shot_delay_steps", 9999))
            info["stage8_max_active_enemy_projectiles"] = int(phase.get("max_active_projectiles", 0))
            info["stage8_enemy_can_move"] = 1 if bool(phase.get("enemy_can_move", False)) else 0
            info["stage8_enemy_can_shoot"] = 1 if bool(phase.get("enemy_can_shoot", False)) else 0
            info["stage8_curriculum_rolling_success"] = stage8_stats.get("success_rate", 0.0)
            info["stage8_curriculum_rolling_metric"] = stage8_stats.get("metric", 0.0)
            info["stage8_curriculum_rolling_death_rate"] = stage8_stats.get("death_rate", 0.0)
            info["stage8_curriculum_rolling_reload_quality"] = stage8_stats.get("reload_quality", 0.0)
            info["stage8_curriculum_rolling_early_reloads"] = stage8_stats.get("avg_early_reloads", 0.0)
            info["stage8_curriculum_rolling_empty_ammo_steps"] = stage8_stats.get("avg_empty_ammo_steps", 0.0)

        ppo_stats = self._get_ppo_stats()

        row = self._make_episode_row(info, success, rolling_success, ppo_stats)
        self.train_writer.writerow(row)
        self.train_log_file.flush()

        print(
            f"[TRAIN {self.stage.stage_id}] "
            f"Episode {self.episode_counter:5d} | "
            f"Reward {float(info.get('total_reward', 0.0)):8.2f} | "
            f"Steps {int(info.get('steps', 0)):4d} | "
            f"Success {success} | "
            f"Metric {float(info.get('stage_success_metric', 0.0)):5.2f} | "
            f"Aim {float(info.get('aim_error_mean', 0.0)):5.3f} | "
            f"OnTarget {float(info.get('time_on_target_ratio', 0.0)):5.1%} | "
            f"MaxAligned {int(info.get('max_aligned_steps', 0)):3d} | "
            f"Invalid {int(info.get('invalid_actions', 0)):2d} | "
            f"Rolling train {rolling_success:5.2%}"
        )

        if self.stage.stage_id == "stage3":
            print(
                f"           NavProgress {float(info.get('nav_progress_ratio', 0.0)):5.1%} | "
                f"WaypointHits {int(info.get('nav_waypoint_reached_count', 0)):3d} | "
                f"PathDev {float(info.get('nav_avg_path_deviation', 0.0)):5.1f} | "
                f"DesiredDist {float(info.get('nav_avg_desired_distance', 0.0)):5.1f} | "
                f"Range {float(info.get('useful_range_steps', 0)) / max(1, int(info.get('steps', 1))):5.1%} | "
                f"BoundarySafe {float(info.get('boundary_safe_ratio', 0.0)):5.1%}"
            )

        if self.stage.stage_id == "stage4":
            print(
                f"           ProjectileSpeed {float(info.get('stage4_projectile_speed_scale', 1.0)):5.0%} | "
                f"FireRate {float(info.get('stage4_enemy_fire_rate_scale', 1.0)):5.0%} | "
                f"Delay {int(info.get('stage4_first_shot_delay_steps', 0)):3d} | "
                f"MaxActive {int(info.get('stage4_max_active_enemy_projectiles', 99)):2d} | "
                f"Level {self.stage4_projectile_speed_level + 1}/{len(STAGE4_PROJECTILE_SPEED_LEVELS)} | "
                f"EpisodesAtLevel {self.stage4_episodes_at_projectile_level:3d} | "
                f"WindowSuccess {stage4_stats.get('success_rate', 0.0):5.1%} | "
                f"WindowMetric {stage4_stats.get('metric', 0.0):5.3f} | "
                f"DeathRate {stage4_stats.get('death_rate', 0.0):5.1%} | "
                f"Damage {stage4_stats.get('avg_damage', 0.0):5.1f} | "
                f"Avoidance {float(info.get('projectile_avoidance_rate', 0.0)):5.1%} | "
                f"DangerSteps {int(info.get('projectile_danger_steps', 0)):3d} | "
                f"BoundarySafe {float(info.get('boundary_safe_ratio', 0.0)):5.1%}"
            )
            self._maybe_update_stage4_curriculum(stage4_stats)

        if self.stage.stage_id == "stage5":
            phase = self._stage5_current_phase()
            stage5_stats = self._stage5_phase_stats()
            print(
                f"           Phase {phase.get('phase', '5C')} | "
                f"Level {self.stage5_curriculum_phase + 1}/{len(STAGE5_CURRICULUM_PHASES)} | "
                f"EpisodesAtLevel {self.stage5_episodes_at_phase:3d} | "
                f"WindowSuccess {stage5_stats.get('success_rate', 0.0):5.1%} | "
                f"WindowMetric {stage5_stats.get('metric', 0.0):5.3f} | "
                f"DeathRate {stage5_stats.get('death_rate', 0.0):5.1%} | "
                f"DamageDealt {stage5_stats.get('avg_damage_dealt', 0.0):5.1f} | "
                f"MovingHits {int(info.get('moving_shots_hit', 0)):2d} | "
                f"MovingHitRate {float(info.get('moving_hit_rate', 0.0)):5.1%} | "
                f"FireRate {float(phase.get('fire_rate_scale', 1.0)):5.0%} | "
                f"Delay {int(phase.get('first_shot_delay_steps', 0)):3d} | "
                f"MaxActive {int(phase.get('max_active_projectiles', 99)):2d}"
            )
            self._maybe_increase_stage5_phase(stage5_stats)

        if self.stage.stage_id == "stage6":
            phase = self._stage6_current_phase()
            stage6_stats = self._stage6_phase_stats()
            print(
                f"           Phase {phase.get('phase', '6C')} | "
                f"Level {self.stage6_curriculum_phase + 1}/{len(STAGE6_CURRICULUM_PHASES)} | "
                f"EpisodesAtLevel {self.stage6_episodes_at_phase:3d} | "
                f"WindowSuccess {stage6_stats.get('success_rate', 0.0):5.1%} | "
                f"WindowMetric {stage6_stats.get('metric', 0.0):5.3f} | "
                f"DeathRate {stage6_stats.get('death_rate', 0.0):5.1%} | "
                f"DamageTaken {stage6_stats.get('avg_damage_taken', 0.0):5.1f} | "
                f"Kills {int(info.get('kills', 0)):1d} | "
                f"DmgRatio {float(info.get('damage_dealt_taken_ratio', 0.0)):5.2f} | "
                f"HitRate {float(info.get('hit_rate', 0.0)):5.1%} | "
                f"ReloadFail {int(info.get('reload_failures', 0)):2d} | "
                f"FireRate {float(phase.get('fire_rate_scale', 1.0)):5.0%} | "
                f"Delay {int(phase.get('first_shot_delay_steps', 0)):3d} | "
                f"MaxActive {int(phase.get('max_active_projectiles', 99)):2d}"
            )
            self._maybe_increase_stage6_phase(stage6_stats)

        if self.stage.stage_id == "stage7":
            phase = self._stage7_current_phase()
            stage7_stats = self._stage7_phase_stats()
            print(
                f"           Phase {phase.get('phase', '7E')} | "
                f"Level {self.stage7_curriculum_phase + 1}/{len(STAGE7_CURRICULUM_PHASES)} | "
                f"Obstacles {int(phase.get('obstacle_count', 1)):1d} | "
                f"EpisodesAtLevel {self.stage7_episodes_at_phase:3d} | "
                f"WindowSuccess {stage7_stats.get('success_rate', 0.0):5.1%} | "
                f"WindowMetric {stage7_stats.get('metric', 0.0):5.3f} | "
                f"DeathRate {stage7_stats.get('death_rate', 0.0):5.1%} | "
                f"BlockedRate {stage7_stats.get('avg_blocked_rate', 0.0):5.1%} | "
                f"BlockedShots {stage7_stats.get('avg_blocked_shots', 0.0):4.1f} | "
                f"DamageTaken {stage7_stats.get('avg_damage_taken', 0.0):5.1f} | "
                f"FireRate {float(phase.get('fire_rate_scale', 1.0)):5.0%} | "
                f"Delay {int(phase.get('first_shot_delay_steps', 0)):3d} | "
                f"MaxActive {int(phase.get('max_active_projectiles', 99)):2d}"
            )
            self._maybe_increase_stage7_phase(stage7_stats)

        if self.stage.stage_id == "stage8":
            phase = self._stage8_current_phase()
            stage8_stats = self._stage8_phase_stats()
            print(
                f"           Phase {phase.get('phase', '8E')} | "
                f"Level {self.stage8_curriculum_phase + 1}/{len(STAGE8_CURRICULUM_PHASES)} | "
                f"Obstacles {int(phase.get('obstacle_count', 0)):1d} | "
                f"EpisodesAtLevel {self.stage8_episodes_at_phase:3d} | "
                f"WindowSuccess {stage8_stats.get('success_rate', 0.0):5.1%} | "
                f"WindowMetric {stage8_stats.get('metric', 0.0):5.3f} | "
                f"DeathRate {stage8_stats.get('death_rate', 0.0):5.1%} | "
                f"ReloadQ {stage8_stats.get('reload_quality', 0.0):5.3f} | "
                f"EarlyReloads {stage8_stats.get('avg_early_reloads', 0.0):4.2f} | "
                f"EmptyAmmoSteps {stage8_stats.get('avg_empty_ammo_steps', 0.0):5.1f} | "
                f"FireRate {float(phase.get('fire_rate_scale', 0.0)):5.0%} | "
                f"Delay {int(phase.get('first_shot_delay_steps', 9999)):4d} | "
                f"MaxActive {int(phase.get('max_active_projectiles', 0)):2d}"
            )
            self._save_stage8_phase_best(stage8_stats)

        if self.save_every_episodes > 0 and self.episode_counter % self.save_every_episodes == 0:
            self._save_checkpoint(reason="periodic", episode=self.episode_counter)

        update_training_stats(
            stage_id=self.stage.stage_id,
            best_success_rate=self.best_eval_success_rate,
            last_rolling_success_rate=self.last_eval_success_rate,
            total_episodes=self.episode_counter,
            skill_table=self.skill_table,
            last_checkpoint_path=self.last_checkpoint_path,
            promotion_reason=info.get("stage_promotion_reason", ""),
            passed_validation=False,
            official_best_model=False,
            validation_success_rate=self.last_eval_success_rate,
            best_validation_score=self.best_validation_score,
            last_validation_report=self.last_validation_report,
        )

    def _make_episode_row(self, info: dict[str, Any], success: int, rolling_success: float, ppo_stats: dict[str, Any]) -> dict[str, Any]:
        row = {
            "episode": self.episode_counter,
            "stage": self.stage.stage_id,
            "stage_name": self.stage.name,
            "timesteps": self.num_timesteps,
            "rolling_success_rate": self._round(rolling_success),
            "stage_promotion_reason": info.get("stage_promotion_reason", NA),
            "stage8_phase_name": info.get("stage8_phase_name", NA) if self.stage.stage_id == "stage8" else NA,
            "stage8_phase_passed": NA,
            "stage8_eval_reason": NA,
            "stage8_last_promotion_event": self.stage8_last_promotion_event or NA,
            "PPO_policy_loss": self._round(ppo_stats.get("policy_loss")),
            "PPO_value_loss": self._round(ppo_stats.get("value_loss")),
            "PPO_entropy": self._round(ppo_stats.get("entropy")),
            "PPO_approx_kl": self._round(ppo_stats.get("approx_kl")),
            "PPO_clip_fraction": self._round(ppo_stats.get("clip_fraction")),
            "learning_rate": self._round(ppo_stats.get("learning_rate")),
            "checkpoint_path": self.last_checkpoint_path or NA,
        }
        for field in METRIC_FIELDS:
            value = success if field == "success" else info.get(field)
            row[field] = self._format_metric(value)
        return row

    def _run_evaluation_check(self) -> bool:
        self.eval_check_counter += 1
        episode_infos: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []

        if self.stage.stage_id == "stage7":
            phase = self._stage7_current_phase()
            print(
                f"[EVAL START stage7] Check {self.eval_check_counter} | "
                f"phase {phase.get('phase', '7A')} | "
                f"episodes {self.eval_episodes_per_check} | "
                f"max_steps {int(phase.get('max_steps', 1200))}"
            )
        if self.stage.stage_id == "stage8":
            phase = self._stage8_current_phase()
            print(
                f"[EVAL START stage8] Check {self.eval_check_counter} | "
                f"phase {phase.get('phase', '8A')} | "
                f"episodes {self.eval_episodes_per_check} | "
                f"max_steps {int(phase.get('max_steps', 900))} | "
                f"waves {int(phase.get('enemy_waves', 5))}"
            )

        for episode_index in range(self.eval_episodes_per_check):
            self.total_eval_episodes += 1
            eval_seed = self.eval_seed_base + self.eval_check_counter * 1009 + episode_index * 37
            info = self._run_single_eval_episode(self.eval_env, seed=eval_seed)
            info["eval_seed"] = eval_seed
            if self.stage.stage_id == "stage7":
                print(
                    f"[EVAL EP stage7] {episode_index + 1}/{self.eval_episodes_per_check} | "
                    f"seed {eval_seed} | "
                    f"success {1 if info.get('success', False) else 0} | "
                    f"steps {int(info.get('steps', info.get('episode_steps', 0)) or 0)} | "
                    f"LOS {float(info.get('enemy_visible_ratio', 0.0) or 0.0):.1%} | "
                    f"clear {float(info.get('enemy_clear_shot_ratio', 0.0) or 0.0):.1%} | "
                    f"shots {int(info.get('shots_fired', 0) or 0)} | "
                    f"damage {float(info.get('damage_dealt', 0.0) or 0.0):.1f}"
                )
            if self.stage.stage_id == "stage8":
                print(
                    f"[EVAL EP stage8] {episode_index + 1}/{self.eval_episodes_per_check} | "
                    f"seed {eval_seed} | success {1 if info.get('success', False) else 0} | "
                    f"waves {float(info.get('stage8_waves_cleared', 0.0) or 0.0):.0f}/{float(info.get('stage8_enemy_waves', 0.0) or 0.0):.0f} | "
                    f"reloads {float(info.get('stage8_reload_started', 0.0) or 0.0):.0f} | "
                    f"good {float(info.get('stage8_good_reloads', 0.0) or 0.0):.0f} | "
                    f"early {float(info.get('stage8_early_reloads', 0.0) or 0.0):.0f} | "
                    f"shots_after_reload {float(info.get('stage8_shots_after_reload', 0.0) or 0.0):.0f} | "
                    f"damage {float(info.get('damage_dealt', 0.0) or 0.0):.1f}"
                )
            episode_infos.append(info)
            success = 1 if info.get("success", False) else 0
            self.eval_success_window.append(success)
            self.last_eval_success_rate = sum(self.eval_success_window) / len(self.eval_success_window)

            row = {
                "eval_check": self.eval_check_counter,
                "eval_episode": self.total_eval_episodes,
                "eval_seed": info.get("eval_seed", NA),
                "stage": self.stage.stage_id,
                "stage_name": self.stage.name,
                "timesteps": self.num_timesteps,
                "rolling_success_rate": self._round(self.last_eval_success_rate),
                "validation_score": NA,
                "good_checks_count": (self.stage8_eval_gate_good_checks if self.stage.stage_id == "stage8" and self.stage8_curriculum_phase < len(STAGE8_CURRICULUM_PHASES) - 1 else self.consecutive_good_eval_checks),
                "required_good_checks": (int(self._stage8_current_phase().get("required_eval_checks", self.required_consecutive_checks)) if self.stage.stage_id == "stage8" and self.stage8_curriculum_phase < len(STAGE8_CURRICULUM_PHASES) - 1 else self.required_consecutive_checks),
                "stage_promotion_reason": info.get("stage_promotion_reason", NA),
                "stage8_phase_name": info.get("stage8_phase_name", NA) if self.stage.stage_id == "stage8" else NA,
                "stage8_phase_passed": NA,
                "stage8_eval_reason": NA,
                "stage8_last_promotion_event": self.stage8_last_promotion_event or NA,
            }
            for field in METRIC_FIELDS:
                row[field] = self._format_metric(info.get(field))
            eval_rows.append(row)

        summary = self._summarize_eval_infos(episode_infos)
        self.last_validation_report = summary
        self.last_validation_score = float(summary["validation_score"])
        self.best_eval_success_rate = max(self.best_eval_success_rate, self.last_eval_success_rate)
        if self.stage.stage_id == "stage7":
            self._update_stage7_eval_gate(summary)
        if self.stage.stage_id == "stage8":
            self._update_stage8_eval_gate(summary)
            self._maybe_promote_stage8_phase_after_eval()

        self._save_checkpoint(reason="latest")

        stage5_final_phase = self.stage.stage_id != "stage5" or self.stage5_curriculum_phase >= len(STAGE5_CURRICULUM_PHASES) - 1
        stage6_final_phase = self.stage.stage_id != "stage6" or self.stage6_curriculum_phase >= len(STAGE6_CURRICULUM_PHASES) - 1
        stage7_final_phase = self.stage.stage_id != "stage7" or self.stage7_curriculum_phase >= len(STAGE7_CURRICULUM_PHASES) - 1
        stage8_final_phase = self.stage.stage_id != "stage8" or self.stage8_curriculum_phase >= len(STAGE8_CURRICULUM_PHASES) - 1
        curriculum_final_phase = stage5_final_phase and stage6_final_phase and stage7_final_phase and stage8_final_phase
        if self.last_validation_score > self.best_validation_score + 1e-9:
            self.best_validation_score = self.last_validation_score
            if curriculum_final_phase:
                self._save_checkpoint(reason="best")
                print(f"Validation improved. Saved candidate best model: {self.stage.best_model_path}")
            elif self.stage.stage_id == "stage5":
                phase_stats = self._stage5_phase_stats()
                self._save_stage5_phase_best_if_needed(phase_stats)
                print(
                    f"Stage 5 curriculum eval is phase-aware ({self._stage5_current_phase().get('phase', '5C')}, "
                    f"level {self.stage5_curriculum_phase + 1}/{len(STAGE5_CURRICULUM_PHASES)}); "
                    "official best Stage 5 checkpoint is saved only at final 5C."
                )
            elif self.stage.stage_id == "stage6":
                phase_stats = self._stage6_phase_stats()
                self._save_stage6_phase_best_if_needed(phase_stats)
                print(
                    f"Stage 6 curriculum eval is phase-aware ({self._stage6_current_phase().get('phase', '6C')}, "
                    f"level {self.stage6_curriculum_phase + 1}/{len(STAGE6_CURRICULUM_PHASES)}); "
                    "official best Stage 6 checkpoint is saved only at final 6D."
                )
            elif self.stage.stage_id == "stage7":
                phase_stats = self._stage7_phase_stats()
                self._save_stage7_phase_best_if_needed(phase_stats)
                print(
                    f"Stage 7 curriculum eval is phase-aware ({self._stage7_current_phase().get('phase', '7E')}, "
                    f"level {self.stage7_curriculum_phase + 1}/{len(STAGE7_CURRICULUM_PHASES)}); "
                    "official best Stage 7 checkpoint is saved only at final 7E."
                )
            elif self.stage.stage_id == "stage8":
                phase_stats = self._stage8_phase_stats()
                self._save_stage8_phase_best_if_needed(phase_stats)
                print(
                    f"Stage 8 curriculum eval is phase-aware ({self._stage8_current_phase().get('phase', '8E')}, "
                    f"level {self.stage8_curriculum_phase + 1}/{len(STAGE8_CURRICULUM_PHASES)}); "
                    "official best Stage 8 checkpoint is saved only at final 8E."
                )

        enough_history = len(self.eval_success_window) >= self.success_window_size
        good_check = enough_history and curriculum_final_phase and self.last_eval_success_rate >= self.success_threshold
        if good_check:
            self.consecutive_good_eval_checks += 1
        elif self.stage.stage_id == "stage6" and enough_history and curriculum_final_phase:
            # Stage 6 has high seed variance. Do not throw away a 3/5 or 4/5
            # validation streak because of one near-miss batch, but still reset
            # on genuinely poor evaluations.
            near_miss = self.last_eval_success_rate >= 0.60 and self.last_validation_score >= 0.65
            if near_miss:
                self.consecutive_good_eval_checks = max(0, self.consecutive_good_eval_checks - 1)
            else:
                self.consecutive_good_eval_checks = 0
        elif self.stage.stage_id == "stage7" and enough_history and curriculum_final_phase:
            # Final obstacle layouts are seed-sensitive because LOS, blocked shots,
            # and pathing failures interact. Decay near misses, but reset bad checks.
            near_miss = self.last_eval_success_rate >= 0.55 and self.last_validation_score >= 0.60
            if near_miss:
                self.consecutive_good_eval_checks = max(0, self.consecutive_good_eval_checks - 1)
            else:
                self.consecutive_good_eval_checks = 0
        else:
            self.consecutive_good_eval_checks = 0

        self._maybe_save_stage6_candidate_checkpoint(curriculum_final_phase=curriculum_final_phase)

        summary["rolling_eval_success_rate"] = self.last_eval_success_rate
        if self.stage.stage_id == "stage8" and self.stage8_curriculum_phase < len(STAGE8_CURRICULUM_PHASES) - 1:
            summary["good_checks"] = self.stage8_eval_gate_good_checks
            summary["required_good_checks"] = int(self._stage8_current_phase().get("required_eval_checks", self.required_consecutive_checks))
        else:
            summary["good_checks"] = self.consecutive_good_eval_checks
            summary["required_good_checks"] = self.required_consecutive_checks

        retention_ok = True
        retention_report: dict[str, Any] = {}
        if self.consecutive_good_eval_checks >= self.required_consecutive_checks:
            retention_ok, retention_report = self._evaluate_previous_skill_retention()
            summary["previous_skill_retention"] = retention_report
            if not retention_ok:
                self.consecutive_good_eval_checks = 0
                print("Previous-skill retention failed; stage is not unlocked yet.")
                for prev_stage_id, report in retention_report.items():
                    print(
                        f"  {prev_stage_id}: success={float(report.get('success_rate', 0.0)):.2%}, "
                        f"required={float(report.get('required_success_rate', 0.0)):.2%}"
                    )

        if self.stage.stage_id == "stage8" and self.stage8_curriculum_phase < len(STAGE8_CURRICULUM_PHASES) - 1:
            summary["good_checks"] = self.stage8_eval_gate_good_checks
            summary["required_good_checks"] = int(self._stage8_current_phase().get("required_eval_checks", self.required_consecutive_checks))
        else:
            summary["good_checks"] = self.consecutive_good_eval_checks
            summary["required_good_checks"] = self.required_consecutive_checks

        # Log validation score and good-check state after the whole evaluation
        # batch is summarized; otherwise each row would only have a placeholder.
        for row in eval_rows:
            row["validation_score"] = self._round(self.last_validation_score)
            if self.stage.stage_id == "stage8":
                gate = summary.get("stage8_eval_gate", {})
                row["stage8_phase_name"] = gate.get("phase_name", row.get("stage8_phase_name", NA))
                row["stage8_phase_passed"] = gate.get("phase_passed", NA)
                row["stage8_eval_reason"] = gate.get("eval_reason", NA)
                row["stage8_last_promotion_event"] = self.stage8_last_promotion_event or row.get("stage8_last_promotion_event", NA)
                row["stage_promotion_reason"] = gate.get("summary_text", row.get("stage_promotion_reason", NA))
                gate_phase_index = int(gate.get("phase_index", 0) or 0)
                is_stage8_intermediate_phase = 0 < gate_phase_index < len(STAGE8_CURRICULUM_PHASES)
                if is_stage8_intermediate_phase and gate_phase_index < len(STAGE8_CURRICULUM_PHASES):
                    row["good_checks_count"] = self.stage8_eval_gate_good_checks
                    row["required_good_checks"] = int(gate.get("required_checks", self.required_consecutive_checks))
                else:
                    row["good_checks_count"] = self.consecutive_good_eval_checks
                    row["required_good_checks"] = self.required_consecutive_checks
            else:
                row["good_checks_count"] = self.consecutive_good_eval_checks
                row["required_good_checks"] = self.required_consecutive_checks
            self.eval_writer.writerow(row)
        self.eval_log_file.flush()

        update_training_stats(
            stage_id=self.stage.stage_id,
            best_success_rate=self.best_eval_success_rate,
            last_rolling_success_rate=self.last_eval_success_rate,
            total_episodes=self.episode_counter,
            skill_table=self.skill_table,
            last_checkpoint_path=self.last_checkpoint_path,
            promotion_reason=summary.get("criteria_summary_text", "evaluation_check_completed"),
            passed_validation=False,
            official_best_model=False,
            validation_success_rate=self.last_eval_success_rate,
            best_validation_score=self.best_validation_score,
            last_validation_report=summary,
        )

        mean_metrics = summary.get("mean_metrics", {})
        print(
            f"[EVAL  {self.stage.stage_id}] "
            f"Check {self.eval_check_counter:3d} | "
            f"Rolling success {self.last_eval_success_rate:5.2%} | "
            f"Validation score {self.last_validation_score:5.3f} | "
            f"Best score {self.best_validation_score:5.3f} | "
            f"Aim {float(mean_metrics.get('aim_error_mean', 0.0)):5.3f} | "
            f"OnTarget {float(mean_metrics.get('time_on_target_ratio', 0.0)):5.1%} | "
            f"Invalid {float(mean_metrics.get('invalid_actions', 0.0)):4.1f} | "
            f"Good checks {self.consecutive_good_eval_checks}/{self.required_consecutive_checks}"
        )
        if self.stage.stage_id == "stage3":
            print(
                f"           NavProgress {float(mean_metrics.get('nav_progress_ratio', 0.0)):5.1%} | "
                f"PathDev {float(mean_metrics.get('nav_avg_path_deviation', 0.0)):5.1f} | "
                f"DesiredDist {float(mean_metrics.get('nav_avg_desired_distance', 0.0)):5.1f} | "
                f"UsableAim {float(mean_metrics.get('stage3_usable_aim_ratio', 0.0)):5.1%} | "
                f"BoundarySafe {float(mean_metrics.get('boundary_safe_ratio', 0.0)):5.1%}"
            )
        if self.stage.stage_id == "stage4":
            print(
                f"           Damage {float(mean_metrics.get('damage_taken', 0.0)):5.1f} | "
                f"Avoidance {float(mean_metrics.get('projectile_avoidance_rate', 0.0)):5.1%} | "
                f"Dodges {float(mean_metrics.get('projectile_dodges', 0.0)):4.1f} | "
                f"HitsTaken {float(mean_metrics.get('projectile_hits_taken', 0.0)):4.1f} | "
                f"DangerSteps {float(mean_metrics.get('projectile_danger_steps', 0.0)):5.1f} | "
                f"BoundarySafe {float(mean_metrics.get('boundary_safe_ratio', 0.0)):5.1%}"
            )
        if self.stage.stage_id == "stage5":
            print(
                f"           Phase {self._stage5_current_phase().get('phase', '5C')} | "
                f"DamageDealt {float(mean_metrics.get('damage_dealt', 0.0)):5.1f} | "
                f"MovingHits {float(mean_metrics.get('moving_shots_hit', 0.0)):4.1f} | "
                f"MovingHitRate {float(mean_metrics.get('moving_hit_rate', 0.0)):5.1%} | "
                f"HitRate {float(mean_metrics.get('hit_rate', 0.0)):5.1%} | "
                f"DamageTaken {float(mean_metrics.get('damage_taken', 0.0)):5.1f} | "
                f"Unavailable {float(mean_metrics.get('shot_unavailable_count', 0.0)):4.1f}"
            )
        if self.stage.stage_id == "stage6":
            print(
                f"           Phase {self._stage6_current_phase().get('phase', '6C')} | "
                f"Kills {float(mean_metrics.get('kills', 0.0)):4.1f} | "
                f"Deaths {float(mean_metrics.get('deaths', 0.0)):4.1f} | "
                f"DamageDealt {float(mean_metrics.get('damage_dealt', 0.0)):5.1f} | "
                f"DamageTaken {float(mean_metrics.get('damage_taken', 0.0)):5.1f} | "
                f"DmgRatio {float(mean_metrics.get('damage_dealt_taken_ratio', 0.0)):5.2f} | "
                f"HitRate {float(mean_metrics.get('hit_rate', 0.0)):5.1%} | "
                f"ProjHits {float(mean_metrics.get('projectile_hits_taken', 0.0)):4.1f} | "
                f"ReloadFail {float(mean_metrics.get('reload_failures', 0.0)):4.1f}"
            )
        if self.stage.stage_id == "stage7":
            print(
                f"           Phase {self._stage7_current_phase().get('phase', '7E')} | "
                f"Obstacles {int(self._stage7_current_phase().get('obstacle_count', 4)):1d} | "
                f"Kills {float(mean_metrics.get('kills', 0.0)):4.1f} | "
                f"Deaths {float(mean_metrics.get('deaths', 0.0)):4.1f} | "
                f"DamageDealt {float(mean_metrics.get('damage_dealt', 0.0)):5.1f} | "
                f"DamageTaken {float(mean_metrics.get('damage_taken', 0.0)):5.1f} | "
                f"DmgRatio {float(mean_metrics.get('damage_dealt_taken_ratio', 0.0)):5.2f} | "
                f"BlockedRate {float(mean_metrics.get('blocked_move_rate', 0.0)):5.1%} | "
                f"BlockedShots {float(mean_metrics.get('shots_blocked', 0.0)):4.1f} | "
                f"LOS {float(mean_metrics.get('enemy_visible_ratio', 0.0)):5.1%}"
            )
        if self.stage.stage_id == "stage8":
            print(
                f"           Phase {self._stage8_current_phase().get('phase', '8E')} | "
                f"Obstacles {int(self._stage8_current_phase().get('obstacle_count', 4)):1d} | "
                f"Waves {float(mean_metrics.get('stage8_waves_cleared', 0.0)):4.1f}/{float(mean_metrics.get('stage8_enemy_waves', 0.0)):4.1f} | "
                f"Kills {float(mean_metrics.get('kills', 0.0)):4.1f} | "
                f"Deaths {float(mean_metrics.get('deaths', 0.0)):4.1f} | "
                f"DamageDealt {float(mean_metrics.get('damage_dealt', 0.0)):5.1f} | "
                f"HitRate {float(mean_metrics.get('hit_rate', 0.0)):5.1%} | "
                f"Reloads {float(mean_metrics.get('stage8_reload_started', 0.0)):4.1f} | "
                f"GoodReloads {float(mean_metrics.get('stage8_good_reloads', 0.0)):4.1f} | "
                f"EarlyReloads {float(mean_metrics.get('stage8_early_reloads', 0.0)):4.1f} | "
                f"EmptyAmmo {float(mean_metrics.get('stage8_empty_ammo_steps', 0.0)):5.1f} | "
                f"ShotsAfterReload {float(mean_metrics.get('stage8_shots_after_reload', 0.0)):4.1f}"
            )
            if self.stage8_last_eval_gate_summary:
                gate = self.stage8_last_eval_gate_summary
                print(
                    f"           PhaseGate {gate.get('phase_name', gate.get('phase', '8A'))} | passed {gate.get('phase_passed', gate.get('passed', False))} | "
                    f"checks {gate.get('checks', 0)}/{gate.get('required_checks', 0)} | {gate.get('summary_text', '')}"
                )
                if self.stage8_last_promotion_event:
                    print(f"           LastPromotion {self.stage8_last_promotion_event}")
        print(summary.get("criteria_summary_text", "No criteria summary available."))

        if self.consecutive_good_eval_checks >= self.required_consecutive_checks and retention_ok:
            self.success_reached = True
            self.stop_reason = (
                f"validated rolling success {self.last_eval_success_rate:.2%} >= {self.success_threshold:.2%} "
                f"for {self.required_consecutive_checks} checks; previous skills retained"
            )
            self.skill_table = update_skill_table_after_stage(
                self.skill_table,
                self.stage.stage_id,
                self.last_eval_success_rate,
                str(self.stage.best_model_path),
                learned=True,
            )
            self._save_checkpoint(reason="best")
            save_json_snapshot(self.stage.skill_table_snapshot_path, self.skill_table)
            print(f"Stage passed validation. Official best MaskablePPO model saved to {self.stage.best_model_path}")

            update_training_stats(
                stage_id=self.stage.stage_id,
                best_success_rate=self.best_eval_success_rate,
                last_rolling_success_rate=self.last_eval_success_rate,
                total_episodes=self.episode_counter,
                skill_table=self.skill_table,
                last_checkpoint_path=self.last_checkpoint_path,
                promotion_reason=self.stop_reason,
                passed_validation=True,
                official_best_model=True,
                validation_success_rate=self.last_eval_success_rate,
                best_validation_score=self.best_validation_score,
                last_validation_report=summary,
            )
            return False

        return True

    def _run_single_eval_episode(self, env, seed: int | None = None) -> dict[str, Any]:
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        info: dict[str, Any] = {}
        while not done:
            action, _ = predict_with_action_masks(self.model, obs, env, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        info = dict(info)
        info["total_reward"] = float(total_reward)
        return info

    def _summarize_eval_infos(self, infos: list[dict[str, Any]]) -> dict[str, Any]:
        count = max(1, len(infos))
        means: dict[str, float] = {}
        totals: dict[str, float] = {}
        for field in METRIC_FIELDS:
            numeric_values: list[float] = []
            for info in infos:
                value = info.get(field)
                if isinstance(value, bool):
                    numeric_values.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    numeric_values.append(float(value))
            if numeric_values:
                means[field] = sum(numeric_values) / len(numeric_values)
                totals[field] = sum(numeric_values)

        success_rate = means.get("success", 0.0)
        mean_stage_metric = means.get("stage_success_metric", 0.0)
        # The rolling success rate is the hard gate. The validation score is only
        # used to decide which candidate checkpoint is the best so far.
        validation_score = 0.70 * success_rate + 0.30 * min(1.0, mean_stage_metric)
        criteria_summary, criteria_text = self._criteria_summary(infos)
        return {
            "episodes": len(infos),
            "success_rate": success_rate,
            "mean_stage_success_metric": mean_stage_metric,
            "validation_score": validation_score,
            "mean_metrics": means,
            "total_metrics": totals,
            "criteria_summary": criteria_summary,
            "criteria_summary_text": criteria_text,
            "eval_seeds": [info.get("eval_seed") for info in infos if info.get("eval_seed") is not None],
        }

    def _criteria_summary(self, infos: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        latest_detail: dict[str, str] = {}
        for info in infos:
            report = info.get("criteria_report") or {}
            for name, payload in report.items():
                try:
                    ok, detail = payload
                except Exception:
                    ok, detail = False, str(payload)
                totals[name] += 1
                if ok:
                    counts[name] += 1
                latest_detail[name] = str(detail)

        if not totals:
            return {}, "Stage not passed: no criteria report available."

        summary: dict[str, Any] = {}
        parts: list[str] = []
        all_ok = True
        for name in sorted(totals.keys()):
            passed = counts[name]
            total = totals[name]
            ok = passed == total
            all_ok = all_ok and ok
            status = "passed" if ok else "failed"
            detail = latest_detail.get(name, "")
            parts.append(f"{name} {status}: {passed}/{total} episodes; latest {detail}")
            summary[name] = {"passed": int(passed), "total": int(total), "latest_detail": detail, "ok": ok}

        prefix = f"{self.stage.name} validation passed:" if all_ok else f"{self.stage.name} not passed:"
        return summary, prefix + "\n" + "\n".join(parts)

    def _evaluate_previous_skill_retention(self) -> tuple[bool, dict[str, Any]]:
        if not self.stage.previous_skill_eval_stage_ids:
            return True, {}

        from rl.env import ShooterEnv

        report: dict[str, Any] = {}
        all_ok = True
        episodes = max(4, min(8, self.eval_episodes_per_check))
        for previous_stage_id in self.stage.previous_skill_eval_stage_ids:
            previous_skill_table = build_skill_table(previous_stage_id, previous_table=self.skill_table, evaluation_mode=True)
            previous_env = ShooterEnv(stage_id=previous_stage_id, render_mode=None, skill_table=previous_skill_table, validation_mode=True, observation_stage_id=self.stage.stage_id)
            successes = 0
            try:
                for _ in range(episodes):
                    eval_seed = self.eval_seed_base + 50_000 + self.eval_check_counter * 997 + episodes * 17 + _
                    info = self._run_single_eval_episode(previous_env, seed=eval_seed)
                    if info.get("success", False):
                        successes += 1
            finally:
                previous_env.close()
            success_rate = successes / episodes
            ok = success_rate + 1e-9 >= self.stage.previous_skill_min_success
            all_ok = all_ok and ok
            report[previous_stage_id] = {
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                "required_success_rate": self.stage.previous_skill_min_success,
                "ok": ok,
            }
        return all_ok, report

    def _save_checkpoint(self, reason: str, episode: int | None = None) -> None:
        state_path, model_path = checkpoint_paths_for_reason(self.stage.stage_id, reason, self.checkpoint_dir, episode)
        self.last_checkpoint_path = str(state_path)
        self.last_checkpoint_path = str(
            save_training_checkpoint(
                model=self.model,
                stage_id=self.stage.stage_id,
                skill_table=self.skill_table,
                callback_state=self.get_state(),
                config=self.training_config,
                checkpoint_path=state_path,
                model_path=model_path,
                env=self.training_env,
                reason=reason,
            )
        )

    def _get_ppo_stats(self) -> dict[str, Any]:
        values = getattr(getattr(self.model, "logger", None), "name_to_value", {}) or {}
        return {
            "policy_loss": values.get("train/policy_gradient_loss"),
            "value_loss": values.get("train/value_loss"),
            "entropy": values.get("train/entropy_loss"),
            "approx_kl": values.get("train/approx_kl"),
            "clip_fraction": values.get("train/clip_fraction"),
            "learning_rate": values.get("train/learning_rate"),
        }

    @staticmethod
    def _round(value: Any) -> Any:
        if value is None:
            return NA
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return NA

    def _format_metric(self, value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        return self._round(value)

    def _on_training_end(self) -> None:
        self.close_files()

    def close_files(self) -> None:
        if self.train_log_file is not None:
            self.train_log_file.flush()
            self.train_log_file.close()
            self.train_log_file = None
        if self.eval_log_file is not None:
            self.eval_log_file.flush()
            self.eval_log_file.close()
            self.eval_log_file = None

    def close(self) -> None:
        self.close_files()
        if self.eval_env is not None:
            self.eval_env.close()
