from __future__ import annotations

import math
from typing import Any

from config.game_constants import (
    PLAYER_DIAMETER_PX,
    PLAYER_RADIUS_PX,
    STAGE3_WALL_CLEARANCE_PX,
)
from game import detection
from game.agent import Agent, wrap_angle
from game.world import Event, Vec2, World
from navigation.path_reference import NavigationReference


class RewardManager:
    """Stage-aware reward and metric manager for PPO.

    The main objective remains to win the duel while surviving.  Shaping rewards
    are intentionally smaller than kill/death and damage terms so the policy does
    not learn to farm auxiliary signals such as line-of-sight toggles or pickups.
    """

    KILL_REWARD = 22.0
    DEATH_PENALTY = -24.0
    HIT_REWARD = 0.85
    DAMAGE_DEALT_COEF = 0.030
    DAMAGE_TAKEN_COEF = -0.065
    SHOT_PENALTY = -0.050
    MISS_PENALTY = -0.160
    BAD_SHOT_PENALTY = -0.380
    SHOT_BLOCKED_PENALTY = -0.350
    INVALID_ACTION_PENALTY = -0.250
    SHOT_UNAVAILABLE_PENALTY = -0.120
    TIME_PENALTY = -0.006

    AIM_IMPROVEMENT_REWARD = 0.045
    AIM_WORSE_PENALTY = -0.018
    ALIGNED_HOLD_REWARD = 0.025
    STAGE1_ALIGNMENT_REWARD = 0.040
    STAGE1_NO_TURN_WHEN_OFF_TARGET_PENALTY = -0.001
    STAGE1_WRONG_TURN_PENALTY = -0.014
    STAGE1_JITTER_PENALTY = -0.006
    ALIGNED_SHOT_REWARD = 0.120

    POSITION_IMPROVE_REWARD = 0.025
    POSITION_WORSE_PENALTY = -0.012
    BLOCKED_MOVE_PENALTY = -0.110
    REPEATED_BLOCKED_MOVE_PENALTY = -0.080
    REGAIN_LOS_REWARD = 0.340
    LOS_LOSS_PENALTY = -0.030
    PASSIVE_HIDING_PENALTY = -0.018
    PROJECTILE_DODGE_REWARD = 0.180
    DANGER_MOVEMENT_REWARD = 0.018
    STAGE4_FAKE_SURVIVAL_PENALTY = -1.20
    STAGE4_BOUNDARY_PENALTY = -0.018
    STAGE4_DANGER_BOUNDARY_PENALTY = -0.032
    USEFUL_PICKUP_REWARD = 1.10
    USELESS_PICKUP_PENALTY = -0.350

    IDEAL_COMBAT_RANGE = 245.0

    STAGE1_ALIGN_THRESHOLD = 0.12
    STAGE1_REQUIRED_CONSEC = 60
    STAGE1_TARGET_RATIO = 0.60
    STAGE1_MAX_MEAN_ERROR = 0.35
    STAGE1_REQUIRED_STEPS = 300

    STAGE2_ALIGN_THRESHOLD = 0.16
    # Stage 2 must learn shot timing, not just hold the crosshair near the target.
    # The wider align threshold is retained for aim-retention metrics, while the
    # tighter fire thresholds decide whether a shot was actually well timed.
    STAGE2_FIRE_THRESHOLD = 0.075
    STAGE2_STRONG_FIRE_THRESHOLD = 0.045
    STAGE2_REQUIRED_HITS = 4
    STAGE2_REQUIRED_HIT_RATE = 0.45
    STAGE2_MAX_BAD_SHOT_RATE = 0.35
    STAGE2_MAX_MEAN_AIM_ERROR = 0.70
    STAGE2_MIN_TIME_ON_TARGET = 0.25
    STAGE2_HESITATION_PENALTY = -0.008
    STAGE2_PRECISE_SHOT_REWARD = 0.060
    STAGE2_WEAK_ALIGNED_SHOT_PENALTY = -0.080
    STAGE2_EXTRA_HIT_REWARD = 1.200
    STAGE2_EXTRA_MISS_PENALTY = -0.300
    STAGE2_EXTRA_BLOCKED_SHOT_PENALTY = -0.450
    STAGE2_PATIENCE_REWARD = 0.002
    STAGE2_SOFT_SHOT_BUDGET = 11
    STAGE2_EXTRA_SHOT_BUDGET_PENALTY = -0.055
    STAGE2_MAX_BLOCKED_SHOT_RATE = 0.25

    STAGE3_RANGE_TOLERANCE = 85.0
    STAGE3_AIM_THRESHOLD = 0.24
    STAGE3_REQUIRED_GOOD_STEPS = 105
    STAGE3_REQUIRED_MOVEMENT = 190.0
    STAGE3_AIM_SHAPING_SCALE = 0.38
    # Moderate combined Stage 3 bonus. This is intentionally not the main
    # reward; it only reinforces the full desired state together:
    # good aim + visibility + correct distance.
    STAGE3_COMBINED_REWARD_SCALE = 0.110
    STAGE3_AIM_QUALITY_DECAY = 4.5
    STAGE3_VISIBILITY_DISCOVERY_FLOOR = 0.15
    STAGE3_IN_RANGE_REWARD = 0.060
    STAGE3_RANGE_STREAK_REWARD = 0.035
    STAGE3_GOOD_TRACKING_REWARD = 0.085
    STAGE3_RANGE_CORRECTION_REWARD = 0.018
    STAGE3_RANGE_DRIFT_PENALTY = -0.030
    STAGE3_IDEAL_BAND_REWARD = 0.090
    STAGE3_DANGER_CLOSE_PENALTY = -0.120
    STAGE3_SMOOTH_DIRECTION_REWARD = 0.007
    STAGE3_SHARP_TURN_PENALTY = -0.010
    STAGE3_OUT_OF_RANGE_PENALTY = -0.045
    STAGE3_MOVEMENT_ACTIVITY_REWARD = 0.002
    STAGE3_MOVEMENT_JITTER_PENALTY = -0.055

    # Stage 3 closed-loop combat-distance control. These terms teach the
    # desired behavior directly: chase when too far, withdraw when too close,
    # and stabilize when the ideal combat distance is reached.
    STAGE3_RANGE_QUALITY_REWARD = 0.050
    STAGE3_AIM_QUALITY_REWARD = 0.045
    STAGE3_VISIBILITY_REWARD = 0.020
    STAGE3_COMBINED_TRACKING_BONUS = 0.070
    STAGE3_MOVING_TRACKING_BONUS = 0.018
    STAGE3_RANGE_CONTROL_REWARD = 0.032
    STAGE3_RANGE_CONTROL_PENALTY = -0.050
    STAGE3_STABLE_RANGE_REWARD = 0.105
    STAGE3_RANGE_STABILITY_DRIFT_PENALTY = -0.020
    STAGE3_RANGE_CONTROL_DEADZONE = 2.0
    STAGE3_STABLE_RANGE_DELTA = 8.0
    STAGE3_NAV_MIN_SCALE = 0.03
    STAGE3_NAV_MAX_SCALE = 0.42
    STAGE3_VERY_BAD_AIM_THRESHOLD = 0.65
    STAGE3_MOVING_BAD_AIM_PENALTY = -0.045
    # Anti-freeze movement terms for Stage 3. These are active only when
    # distance correction is needed, so they do not push the agent to wander
    # after it reaches the ideal combat band.
    STAGE3_NEEDED_MOVEMENT_REWARD = 0.035
    STAGE3_NEEDED_STATIONARY_PENALTY = -0.035
    STAGE3_STATIONARY_GRACE_STEPS = 20
    # Checkpoint-compatible Stage 3 refinement. These terms do not change
    # observations, actions, or metrics; they only make the existing reward
    # less likely to punish correct tactical slowing/stabilization.
    STAGE3_TACTICAL_POCKET_REWARD = 0.030
    STAGE3_TACTICAL_POCKET_SLOW_REWARD = 0.018
    STAGE3_SMOOTH_CORRECTION_REWARD = 0.012
    STAGE3_NAV_AIM_GATE_FLOOR = 0.40

    STAGE3_MIN_USEFUL_RANGE_RATIO = 0.35
    STAGE3_MIN_VISIBLE_RATIO = 0.45
    STAGE3_MIN_USABLE_AIM_RATIO = 0.32
    STAGE3_MIN_STRICT_AIM_STREAK = 20
    STAGE3_MIN_NAV_PROGRESS_RATIO = 0.45
    STAGE3_MAX_AVG_PATH_DEVIATION = 70.0
    STAGE3_MIN_BOUNDARY_SAFE_RATIO = 0.88
    STAGE3_MAX_JITTER_RATE = 0.08
    STAGE3_BLOCKED_RATE_LIMIT = 0.045
    STAGE3_MAX_MEAN_AIM_ERROR = 0.55
    STAGE3_MIN_RANGE_CONTROL_RATIO = 0.52

    # Stage 5 is centered on successful moving combat output. These terms are
    # stronger than passive aim/range shaping, but still below kill/death events.
    STAGE5_MOVING_HIT_BONUS = 1.650
    STAGE5_STATIONARY_SHOT_PENALTY = -0.220
    STAGE5_STATIONARY_HIT_PENALTY = -0.450
    STAGE5_ALIGNED_MOVING_SHOT_REWARD = 0.090
    STAGE5_COMBAT_AIM_SUPPORT = 0.014
    STAGE5_COMBAT_RANGE_SUPPORT = 0.006
    STAGE5_HESITATION_PENALTY = -0.006
    STAGE5_MIN_MOVING_SPEED_RATIO = 0.12

    # Stage 6 reintroduces full open-arena combat.  The shaping is still below
    # kill/death and damage events, but it pushes the Stage 5 kill-race policy
    # toward cleaner wins: less damage taken, fewer projectile hits, and fewer
    # useless reload/shoot attempts.
    STAGE6_CLEAN_DAMAGE_BONUS = 0.006
    STAGE6_LOW_DAMAGE_PENALTY = -0.008
    STAGE6_RELOAD_FAILURE_EXTRA_PENALTY = -0.060
    STAGE6_EMPTY_TRIGGER_EXTRA_PENALTY = -0.080
    STAGE6_PROJECTILE_HIT_EXTRA_PENALTY = -0.140
    STAGE6_ALIGNED_MOVING_PRESSURE_REWARD = 0.006

    # Stage 7 must not be solvable by waiting for the enemy to walk around
    # cover.  These terms specifically reward active line-of-sight search and
    # punish standing still while the enemy is hidden.
    STAGE7_SEARCH_MOVEMENT_REWARD = 0.006
    STAGE7_NO_LOS_STATIONARY_PENALTY = -0.110
    STAGE7_ACTIVE_LOS_REGAIN_BONUS = 0.420
    STAGE7_LONG_NO_LOS_EXTRA_PENALTY = -0.055
    STAGE7_NO_LOS_GRACE_STEPS = 20
    STAGE7_LOS_NAV_PROGRESS_REWARD = 0.045
    STAGE7_LOS_NAV_REGRESS_PENALTY = -0.075
    STAGE7_MOVING_WITHOUT_NAV_PROGRESS_PENALTY = -0.026
    STAGE7_NAV_CORRIDOR_BONUS = 0.020
    STAGE7_VISIBLE_AIM_IMPROVE_REWARD = 0.040
    STAGE7_VISIBLE_AIM_WORSE_PENALTY = -0.035
    STAGE7_VISIBLE_ALIGNED_REWARD = 0.055
    STAGE7_VISIBLE_BAD_AIM_PENALTY = -0.065
    STAGE7_READY_VISIBLE_NO_SHOT_PENALTY = -0.014
    STAGE7_READY_ALIGNED_NO_SHOT_PENALTY = -0.040
    STAGE7_CLEAR_SHOT_REWARD = 0.090
    STAGE7_CLEAR_SHOT_GAIN_BONUS = 0.360
    STAGE7_LOS_NO_CLEAR_SHOT_PENALTY = -0.028
    STAGE7_READY_CLEAR_NO_SHOT_PENALTY = -0.070
    STAGE7_BLOCKED_SHOT_EXTRA_PENALTY = -0.850
    STAGE7_UNCLEAN_SHOT_ATTEMPT_PENALTY = -0.420
    STAGE7_CLEAN_HIT_BONUS = 0.450

    # Stage 8: final tactical reload management with retained movement/aiming.
    # Tactical mistakes are not action-masked in the final setup; they are
    # learned through reward penalties and validation metrics.
    STAGE8_TARGET_RELOAD_RATIO = 1.0 / 3.0
    STAGE8_GOOD_RELOAD_BAND = 0.001  # Stage 8 now uses integer ammo checks; kept for compatibility.
    STAGE8_ACCEPTABLE_RELOAD_BAND = 0.112  # 2/9 is acceptable-late; 4/9 is early.
    STAGE8_GOOD_RELOAD_REWARD = 2.000
    STAGE8_ACCEPTABLE_RELOAD_REWARD = 0.120
    STAGE8_EARLY_RELOAD_PENALTY = -1.600
    STAGE8_LATE_RELOAD_PENALTY = -0.900
    STAGE8_EMPTY_RELOAD_PENALTY = -1.150
    STAGE8_EMPTY_TRIGGER_EXTRA_PENALTY = -0.550
    STAGE8_RELOAD_FAILURE_EXTRA_PENALTY = -0.350
    STAGE8_RELOAD_WHILE_KILLABLE_PENALTY = -0.650
    STAGE8_LOW_AMMO_NO_RELOAD_PENALTY = -0.045
    STAGE8_CRITICAL_AMMO_NO_RELOAD_PENALTY = -0.110
    STAGE8_CLEAN_ALIGNED_SHOT_REWARD = 0.145
    STAGE8_UNALIGNED_SHOT_ATTEMPT_PENALTY = -0.520
    STAGE8_NO_LOS_SHOT_ATTEMPT_PENALTY = -0.650
    STAGE8_AIM_IMPROVE_REWARD = 0.070
    STAGE8_AIM_WORSE_PENALTY = -0.070
    STAGE8_AIM_HOLD_REWARD = 0.055
    STAGE8_VISIBLE_BAD_AIM_PENALTY = -0.095
    STAGE8_ROTATION_FLIP_PENALTY = -0.020
    STAGE8_LARGE_TURN_PENALTY = -0.014
    STAGE8_BLOCKED_SPIN_PENALTY = -0.050
    STAGE8_MOVEMENT_FLIP_PENALTY = -0.045
    STAGE8_SMOOTH_MOVE_REWARD = 0.010
    STAGE8_BLOCKED_MOVE_EXTRA_PENALTY = -0.190
    STAGE8_REPEATED_BLOCK_EXTRA_PENALTY = -0.120
    STAGE8_BLOCKED_TURN_FLIP_EXTRA_PENALTY = -0.050
    STAGE8_CLEAR_SHOT_GAIN_BONUS = 0.420
    STAGE8_CLEAR_ALIGNED_NO_SHOT_PENALTY = -0.075
    STAGE8_SHOT_AFTER_RELOAD_REWARD = 0.240
    STAGE8_NO_POST_RELOAD_COMBAT_PENALTY = -0.018
    STAGE8_MIN_SHOTS_BEFORE_RELOAD = 1
    STAGE8_PREMATURE_RELOAD_PENALTY = -1.500
    # Stage 8A unmasked reload-timing tutor. These terms are deliberately
    # phase-limited so later combat phases are not hard-scripted. The policy
    # may still choose early/late reloads, but 8A makes the desired sequence
    # much clearer: spawn at 4/9 -> shoot once -> reach 3/9 -> reload -> shoot.
    STAGE8_8A_REACH_TARGET_AMMO_REWARD = 0.850
    STAGE8_8A_TARGET_AMMO_HOLD_REWARD = 0.018
    STAGE8_8A_TARGET_NO_RELOAD_PENALTY = -0.160
    STAGE8_8A_SHOOT_PAST_TARGET_PENALTY = -2.000
    STAGE8_8A_EARLY_RELOAD_EXTRA_PENALTY = -2.200
    STAGE8_8A_EMPTY_AMMO_STEP_PENALTY = -0.320
    STAGE8_8A_EMPTY_SHOT_EXTRA_PENALTY = -1.600
    STAGE8_8A_GOOD_RELOAD_EXTRA_REWARD = 3.000
    STAGE8_8A_POST_RELOAD_SHOT_EXTRA_REWARD = 1.000
    STAGE8_8A_NO_FIRST_SHOT_GRACE_STEPS = 80
    STAGE8_8A_NO_FIRST_SHOT_BASE_PENALTY = -0.035
    STAGE8_8A_ALIGNED_NO_FIRST_SHOT_PENALTY = -0.120
    STAGE8_8A_FIRST_SHOT_REWARD = 1.150
    STAGE8_8A_AIM_REWARD_SCALE_BEFORE_FIRST_SHOT = 0.15
    STAGE8_8A_TARGET_NO_RELOAD_TIMEOUT_STEPS = 55
    STAGE8_8A_TARGET_NO_RELOAD_TIMEOUT_PENALTY = -0.320
    STAGE8_ALIGNMENT_THRESHOLD = 0.14
    STAGE8_STRICT_ALIGNMENT_THRESHOLD = 0.10
    STAGE8_MAX_BLOCKED_RATE = 0.060
    STAGE8_MAX_MEAN_AIM_ERROR = 0.42
    STAGE8_MIN_HIT_RATE_FOR_AIM_OVERRIDE = 0.75
    STAGE8_MAX_TURN_FLIP_RATE = 0.300
    STAGE8_MAX_MOVEMENT_FLIP_RATE = 0.150
    STAGE8_MAX_BLOCKED_SPIN_STEPS = 30
    STAGE8_MIN_TIME_ON_TARGET = 0.18
    STAGE8_MIN_HIT_RATE = 0.70
    STAGE8_MIN_CLEAR_SHOT_RATIO = 0.060
    STAGE8_MIN_VISIBLE_RATIO = 0.30
    STAGE8_MAX_UNAVAILABLE_SHOTS = 3
    STAGE8_MAX_RELOAD_FAILURES = 1

    NAV_PROGRESS_REWARD = 0.004
    NAV_REGRESS_PENALTY = -0.004
    NAV_WAYPOINT_REWARD = 0.050
    NAV_CORRIDOR_REWARD = 0.010
    NAV_DEVIATION_PENALTY = -0.020
    NAV_DIRECTION_REWARD = 0.018
    NAV_WRONG_DIRECTION_PENALTY = -0.018
    NAV_STUCK_PENALTY = -0.020
    BOUNDARY_SAFE_DISTANCE = STAGE3_WALL_CLEARANCE_PX
    BOUNDARY_HUGGING_PENALTY = -0.022
    TOO_CLOSE_RANGE_PENALTY = -0.035

    def __init__(self, stage_id: str):
        self.stage_id = stage_id
        self.reset(None, None, None)

    def reset(self, learner: Agent | None, enemy: Agent | None, world: World | None) -> None:
        self.kills = 0
        self.deaths = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.shots_missed = 0
        self.shots_blocked = 0
        self.valid_shots = 0
        self.bad_shots = 0
        self.moving_shots_fired = 0
        self.moving_shots_hit = 0
        self.moving_damage_dealt = 0.0
        self.stage5_curriculum_level = 0
        self.stage5_curriculum_total_levels = 0
        self.stage5_curriculum_phase_index = 3
        self.stage6_curriculum_level = 0
        self.stage6_curriculum_total_levels = 0
        self.stage6_curriculum_phase_index = 4
        self.stage7_curriculum_level = 0
        self.stage7_curriculum_total_levels = 0
        self.stage7_curriculum_phase_index = 5
        self.invalid_actions = 0
        self.shot_unavailable_count = 0
        self.reload_failures = 0
        self.blocked_moves = 0
        self.consecutive_blocked_moves = 0
        self.max_consecutive_blocked_moves = 0
        self.line_of_sight_gained_count = 0
        self.line_of_sight_lost_count = 0
        self.clear_shot_gained_count = 0
        self.clear_shot_lost_count = 0
        self.steps_with_enemy_clear_shot = 0
        self.stage7_unclean_shot_attempts = 0
        self.enemy_visible_events = 0
        self.enemy_lost_events = 0
        self.steps_with_enemy_visible = 0
        self.useful_pickups = 0
        self.useless_pickups = 0
        self.pickups_collected = 0
        self.stage8_reload_started = 0
        self.stage8_good_reloads = 0
        self.stage8_acceptable_reloads = 0
        self.stage8_early_reloads = 0
        self.stage8_late_reloads = 0
        self.stage8_empty_reloads = 0
        self.stage8_reload_while_killable = 0
        self.stage8_low_ammo_steps = 0
        self.stage8_critical_ammo_steps = 0
        self.stage8_empty_ammo_steps = 0
        self.stage8_clean_aligned_shots = 0
        self.stage8_shots_after_reload = 0
        self.stage8_unaligned_shot_attempts = 0
        self.stage8_no_los_shot_attempts = 0
        self.stage8_aim_hold_steps = 0
        self.stage8_aim_improve_steps = 0
        self.stage8_aim_worse_steps = 0
        self.stage8_large_turn_count = 0
        self.stage8_turn_flip_count = 0
        self.stage8_blocked_spin_steps = 0
        self.stage8_movement_flip_count = 0
        self.stage8_smooth_move_steps = 0
        self.stage8_reload_quality_sum = 0.0
        self.stage8_target_ammo_reached = False
        self.stage8_target_ammo_steps = 0
        self.stage8_shot_past_target_before_reload = 0
        self.stage8_first_shot_rewarded = False
        self.stage8_no_first_shot_pressure_steps = 0
        self.stage8_target_no_reload_timeout_steps = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.hits_taken = 0
        self.total_reward = 0.0

        self.projectile_danger_steps = 0
        self.projectile_dodges = 0
        self.projectile_hits_taken = 0
        self.prev_projectile_danger = False

        self.prev_distance = 0.0
        self.prev_line_of_sight = False
        self.prev_angle_error = math.pi
        self.prev_learner_position: Vec2 | None = None
        self.los_lost_streak = 0
        self.los_recovery_cooldown = 0
        self.passive_steps = 0
        self.stationary_steps = 0

        self.aligned_consecutive_steps = 0
        self.max_aligned_consecutive_steps = 0
        self.good_tracking_steps = 0
        self.useful_range_steps = 0
        self.range_streak_steps = 0
        self.max_range_streak_steps = 0
        self.movement_jitter_count = 0
        self.movement_jitter_streak = 0
        self.movement_distance_total = 0.0
        self.prev_velocity = Vec2(0.0, 0.0)

        self.stage3_usable_aim_steps = 0
        self.stage3_range_control_opportunities = 0
        self.stage3_correct_range_control_steps = 0
        self.stage3_wrong_range_control_steps = 0
        self.stage3_stable_range_steps = 0
        self.stage3_range_drift_sum = 0.0
        self.boundary_safe_steps = 0
        self.boundary_hugging_steps = 0
        self.nav_valid_steps = 0
        self.nav_progress_positive_steps = 0
        self.nav_regress_steps = 0
        self.nav_waypoint_reached_count = 0
        self.nav_path_deviation_sum = 0.0
        self.nav_path_deviation_bad_steps = 0
        self.nav_desired_distance_sum = 0.0
        self.nav_path_progress_sum = 0.0
        self.nav_desired_progress_sum = 0.0
        self.nav_direction_good_steps = 0
        self.nav_recompute_count = 0

        self.aim_error_sum = 0.0
        self.aim_error_count = 0
        self.aim_error_min = math.pi
        self.aim_error_final = math.pi
        self.time_on_target_steps = 0
        self.heading_delta_total = 0.0
        self.heading_jitter_count = 0
        self.prev_heading = 0.0
        self.prev_rotation_sign = 0
        self.survival_steps = 0
        self.last_criteria: dict[str, tuple[bool, str]] = {}

        if learner is not None and enemy is not None:
            self.prev_distance = learner.position.distance_to(enemy.position)
            self.prev_angle_error = self._angle_error_to_enemy(learner, enemy)
            self.prev_heading = learner.heading
            self.prev_learner_position = Vec2(learner.position.x, learner.position.y)

        if learner is not None and enemy is not None and world is not None:
            self.prev_line_of_sight = world.has_line_of_sight(learner.position, enemy.position)
            self.prev_clear_shot = self._stage2_shot_ray_hits_enemy(learner, enemy, world) if self.prev_line_of_sight else False
            self.prev_projectile_danger = detection.projectile_threat(world, learner).danger
        else:
            self.prev_clear_shot = False

    def compute(self, events: list[Event], learner: Agent, enemy: Agent, world: World, step_count: int, nav_reference: NavigationReference | None = None) -> float:
        current_distance = learner.position.distance_to(enemy.position)
        current_line_of_sight = enemy.is_alive and world.has_line_of_sight(learner.position, enemy.position)
        current_clear_shot = current_line_of_sight and self._stage2_shot_ray_hits_enemy(learner, enemy, world)
        current_angle_error = self._angle_error_to_enemy(learner, enemy) if enemy.is_alive else math.pi
        heading_delta = wrap_angle(learner.heading - self.prev_heading)
        moving_this_step = learner.velocity.length() > max(5.0, learner.move_speed * 0.08)
        threat = detection.projectile_threat(world, learner)
        damage_taken_before = self.damage_taken

        self._update_continuous_metrics(learner, current_line_of_sight, current_clear_shot, current_angle_error, moving_this_step)
        boundary_clearance = self._boundary_clearance(learner, world)
        if boundary_clearance >= self.BOUNDARY_SAFE_DISTANCE:
            self.boundary_safe_steps += 1
        else:
            self.boundary_hugging_steps += 1

        reward = self.TIME_PENALTY
        reward += self._base_event_rewards(events, learner)
        damage_taken_this_step = self.damage_taken > damage_taken_before

        resolved_dodge_events = sum(
            1
            for event in events
            if event.agent_id == learner.agent_id and event.kind in {"projectile_passed_agent", "projectile_resolved_safe"}
        )

        if threat.danger:
            self.projectile_danger_steps += 1
            if moving_this_step:
                reward += self.DANGER_MOVEMENT_REWARD
        if resolved_dodge_events > 0:
            self.projectile_dodges += resolved_dodge_events
            reward += self.PROJECTILE_DODGE_REWARD * resolved_dodge_events
            if self.projectile_dodges > 1:
                reward += min(0.060, 0.015 * resolved_dodge_events)
        elif self.prev_projectile_danger and not threat.danger and not damage_taken_this_step:
            self.projectile_dodges += 1
            reward += self.PROJECTILE_DODGE_REWARD
        if damage_taken_this_step:
            self.projectile_hits_taken += 1

        if self.stage_id == "stage1":
            reward += self._stage1_reward(current_line_of_sight, current_angle_error, heading_delta)
        elif self.stage_id == "stage2":
            reward += self._stage2_reward(events, learner, enemy, world, current_line_of_sight, current_angle_error)
        elif self.stage_id == "stage3":
            reward += self._stage3_reward(learner, current_distance, current_line_of_sight, current_angle_error, moving_this_step, nav_reference, boundary_clearance)
        elif self.stage_id == "stage4":
            reward += self._stage4_reward(
                learner=learner,
                current_distance=current_distance,
                current_line_of_sight=current_line_of_sight,
                moving=moving_this_step,
                threat=threat,
                step_count=step_count,
                boundary_clearance=boundary_clearance,
            )
        elif self.stage_id == "stage5":
            reward += self._stage5_reward(events, learner, current_distance, current_line_of_sight, current_angle_error, moving_this_step, nav_reference)
        elif self.stage_id == "stage6":
            reward += self._combat_reward(learner, current_distance, current_line_of_sight, current_angle_error, moving_this_step, obstacles=False, nav_reference=nav_reference)
            reward += self._stage6_combat_refinement(events, learner, current_line_of_sight, current_angle_error, moving_this_step)
        elif self.stage_id == "stage7":
            reward += self._combat_reward(learner, current_distance, current_line_of_sight, current_angle_error, moving_this_step, obstacles=True, nav_reference=nav_reference)
            reward += self._obstacle_visibility_reward(current_line_of_sight, moving_this_step)
            reward += self._stage7_active_obstacle_reward(events, learner, enemy, world, current_line_of_sight, current_clear_shot, current_angle_error, moving_this_step, nav_reference)
        elif self.stage_id == "stage8":
            reward += self._combat_reward(learner, current_distance, current_line_of_sight, current_angle_error, moving_this_step, obstacles=True, nav_reference=nav_reference)
            reward += self._obstacle_visibility_reward(current_line_of_sight, moving_this_step)
            reward += self._stage8_tactical_reload_reward(events, learner, enemy, world, current_line_of_sight, current_clear_shot, current_angle_error, moving_this_step, heading_delta, threat.danger, nav_reference)

        self.prev_distance = current_distance
        self.prev_line_of_sight = current_line_of_sight
        self.prev_clear_shot = current_clear_shot
        self.prev_angle_error = current_angle_error
        self.prev_heading = learner.heading
        self.prev_velocity = Vec2(learner.velocity.x, learner.velocity.y)
        self.prev_projectile_danger = threat.danger
        self.total_reward += reward
        return reward

    def _update_continuous_metrics(self, learner: Agent, visible: bool, clear_shot: bool, angle_error: float, moving: bool) -> None:
        self.survival_steps += 1
        self.aim_error_sum += angle_error
        self.aim_error_count += 1
        self.aim_error_min = min(self.aim_error_min, angle_error)
        self.aim_error_final = angle_error
        if visible:
            self.steps_with_enemy_visible += 1
            self.los_lost_streak = 0
        else:
            self.los_lost_streak += 1
        if clear_shot:
            self.steps_with_enemy_clear_shot += 1
        if clear_shot and not self.prev_clear_shot:
            self.clear_shot_gained_count += 1
        elif self.prev_clear_shot and not clear_shot:
            self.clear_shot_lost_count += 1
        if visible and angle_error < self.STAGE1_ALIGN_THRESHOLD:
            self.time_on_target_steps += 1
        if moving:
            self.stationary_steps = 0
        else:
            self.stationary_steps += 1
        if self.prev_learner_position is not None:
            self.movement_distance_total += learner.position.distance_to(self.prev_learner_position)
        self.prev_learner_position = Vec2(learner.position.x, learner.position.y)
        if self.los_recovery_cooldown > 0:
            self.los_recovery_cooldown -= 1

    def _base_event_rewards(self, events: list[Event], learner: Agent) -> float:
        reward = 0.0
        blocked_this_step = False
        for event in events:
            is_learner = event.agent_id == learner.agent_id

            if event.kind == "shot_fired" and is_learner:
                self.shots_fired += 1
                if event.data.get("moving_shot"):
                    self.moving_shots_fired += 1
                reward += self.SHOT_PENALTY
            elif event.kind == "shot_hit" and is_learner:
                self.shots_hit += 1
                self.damage_dealt += float(event.value)
                if event.data.get("moving_shot"):
                    self.moving_shots_hit += 1
                    self.moving_damage_dealt += float(event.value)
                reward += self.HIT_REWARD + self.DAMAGE_DEALT_COEF * float(event.value)
            elif event.kind == "damage_taken" and is_learner:
                self.damage_taken += float(event.value)
                self.hits_taken += 1
                reward += self.DAMAGE_TAKEN_COEF * float(event.value)
            elif event.kind == "shot_missed" and is_learner:
                self.shots_missed += 1
                reward += self.MISS_PENALTY
            elif event.kind == "shot_blocked" and is_learner:
                self.shots_blocked += 1
                reward += self.SHOT_BLOCKED_PENALTY
            elif event.kind == "kill" and is_learner:
                self.kills += 1
                reward += self.KILL_REWARD
            elif event.kind == "death" and is_learner:
                self.deaths += 1
                reward += self.DEATH_PENALTY
            elif event.kind == "blocked_move" and is_learner:
                self.blocked_moves += 1
                blocked_this_step = True
                reward += self.BLOCKED_MOVE_PENALTY
            elif event.kind == "invalid_action" and is_learner:
                self.invalid_actions += 1
                ability = event.data.get("ability")
                if ability == "shoot_without_clear_shot":
                    self.stage7_unclean_shot_attempts += 1
                    if self.stage_id == "stage7":
                        reward += self.STAGE7_UNCLEAN_SHOT_ATTEMPT_PENALTY
                    elif self.stage_id == "stage8":
                        self.stage8_no_los_shot_attempts += 1
                        reward += self.STAGE8_NO_LOS_SHOT_ATTEMPT_PENALTY
                    else:
                        reward += self.INVALID_ACTION_PENALTY
                elif ability == "shoot_without_aim_alignment" and self.stage_id == "stage8":
                    self.stage8_unaligned_shot_attempts += 1
                    reward += self.STAGE8_UNALIGNED_SHOT_ATTEMPT_PENALTY
                else:
                    reward += self.INVALID_ACTION_PENALTY
            elif event.kind in {"shot_failed_reloading", "shot_failed_cooldown", "shot_failed_no_ammo"} and is_learner:
                self.shot_unavailable_count += 1
                reward += self.SHOT_UNAVAILABLE_PENALTY
            elif event.kind in {"reload_failed_already_reloading", "reload_failed_full_ammo", "reload_failed_no_reserve"} and is_learner:
                self.reload_failures += 1
                reward += self.SHOT_UNAVAILABLE_PENALTY * 0.5
                if self.stage_id == "stage8":
                    reward += self.STAGE8_RELOAD_FAILURE_EXTRA_PENALTY
            elif event.kind == "line_of_sight_gained" and is_learner:
                self.line_of_sight_gained_count += 1
            elif event.kind == "line_of_sight_lost" and is_learner:
                self.line_of_sight_lost_count += 1
            elif event.kind == "enemy_visible" and is_learner:
                self.enemy_visible_events += 1
            elif event.kind == "enemy_lost" and is_learner:
                self.enemy_lost_events += 1
            elif event.kind == "pickup_collected" and is_learner:
                self.pickups_collected += 1
            elif event.kind == "useful_pickup_collected" and is_learner:
                self.useful_pickups += 1
            elif event.kind == "useless_pickup_collected" and is_learner:
                self.useless_pickups += 1

        if blocked_this_step:
            self.consecutive_blocked_moves += 1
            self.max_consecutive_blocked_moves = max(self.max_consecutive_blocked_moves, self.consecutive_blocked_moves)
            if self.consecutive_blocked_moves >= 2:
                reward += self.REPEATED_BLOCKED_MOVE_PENALTY * (self.consecutive_blocked_moves - 1)
        else:
            self.consecutive_blocked_moves = 0
        return reward

    def _stage1_reward(self, current_line_of_sight: bool, current_angle_error: float, heading_delta: float) -> float:
        reward = 0.0
        normalized_error = min(1.0, current_angle_error / math.pi)
        improvement = self.prev_angle_error - current_angle_error
        heading_delta_abs = abs(heading_delta)
        rotation_sign = 1 if heading_delta > 1e-5 else -1 if heading_delta < -1e-5 else 0

        # Stage 1 must learn stable tracking over the whole episode. Dense error
        # shaping makes doing nothing and random spinning worse than reducing error.
        if current_line_of_sight:
            reward += 0.060 * (1.0 - normalized_error) - 0.030 * normalized_error
            if improvement > 0.010:
                reward += min(0.080, improvement * 0.90)
            elif improvement < -0.010:
                reward -= min(0.080, (-improvement) * 0.90)
        else:
            reward -= 0.030

        aligned = current_line_of_sight and current_angle_error < self.STAGE1_ALIGN_THRESHOLD
        if aligned:
            self.aligned_consecutive_steps += 1
            streak_bonus = min(0.030, self.aligned_consecutive_steps / max(1.0, self.STAGE1_REQUIRED_CONSEC) * 0.030)
            reward += self.STAGE1_ALIGNMENT_REWARD + streak_bonus
        else:
            self.aligned_consecutive_steps = 0
            if current_line_of_sight and heading_delta_abs <= 1e-5 and current_angle_error > self.STAGE1_ALIGN_THRESHOLD * 1.5:
                reward += self.STAGE1_NO_TURN_WHEN_OFF_TARGET_PENALTY

        if (
            current_line_of_sight
            and heading_delta_abs > 1e-5
            and current_angle_error > self.STAGE1_ALIGN_THRESHOLD
            and improvement < -0.006
        ):
            reward += self.STAGE1_WRONG_TURN_PENALTY

        if rotation_sign != 0 and self.prev_rotation_sign != 0 and rotation_sign != self.prev_rotation_sign:
            self.heading_jitter_count += 1
            reward += self.STAGE1_JITTER_PENALTY
        if rotation_sign != 0:
            self.prev_rotation_sign = rotation_sign
        self.heading_delta_total += heading_delta_abs

        self.max_aligned_consecutive_steps = max(self.max_aligned_consecutive_steps, self.aligned_consecutive_steps)
        return reward

    def _stage2_reward(
        self,
        events: list[Event],
        learner: Agent,
        enemy: Agent,
        world: World,
        current_line_of_sight: bool,
        current_angle_error: float,
    ) -> float:
        # Stage 2 is a shooting-timing stage. Aim retention remains useful, but
        # the policy must learn trigger discipline: fire only when the current
        # ray is actually a clean Glock line toward the enemy, not merely when the
        # enemy is visible somewhere in the arena.
        reward = 0.45 * self._aim_tracking_reward(current_line_of_sight, current_angle_error, self.STAGE2_ALIGN_THRESHOLD)
        normalized_error = min(1.0, current_angle_error / math.pi)
        if current_line_of_sight:
            reward += 0.012 * (1.0 - normalized_error)
        else:
            reward -= 0.020

        shot_fired_this_step = False
        precise_now = current_line_of_sight and current_angle_error < self.STAGE2_FIRE_THRESHOLD
        clean_now = precise_now and self._stage2_shot_ray_hits_enemy(learner, enemy, world)
        strong_now = current_line_of_sight and current_angle_error < self.STAGE2_STRONG_FIRE_THRESHOLD
        strong_clean_now = strong_now and self._stage2_shot_ray_hits_enemy(learner, enemy, world)

        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "shot_fired":
                shot_fired_this_step = True
                if clean_now:
                    self.valid_shots += 1
                    reward += self.STAGE2_PRECISE_SHOT_REWARD
                else:
                    # A visible enemy is not enough. If the active firing ray is
                    # not clean, the shot is treated as bad trigger discipline.
                    self.bad_shots += 1
                    reward += self.BAD_SHOT_PENALTY
                    if current_line_of_sight and current_angle_error < self.STAGE2_ALIGN_THRESHOLD:
                        reward += self.STAGE2_WEAK_ALIGNED_SHOT_PENALTY
                if self.shots_fired > self.STAGE2_SOFT_SHOT_BUDGET:
                    excess = self.shots_fired - self.STAGE2_SOFT_SHOT_BUDGET
                    reward += self.STAGE2_EXTRA_SHOT_BUDGET_PENALTY * min(6, excess)
            elif event.kind == "shot_hit":
                reward += self.STAGE2_EXTRA_HIT_REWARD
            elif event.kind == "shot_missed":
                reward += self.STAGE2_EXTRA_MISS_PENALTY
            elif event.kind == "shot_blocked":
                # In open Stage 2 maps, most wall-blocked projectiles are simply
                # bad/wasted shots that travelled past the enemy and hit the arena.
                reward += self.STAGE2_EXTRA_BLOCKED_SHOT_PENALTY

        if learner.fire_ready >= 0.5 and not shot_fired_this_step:
            if strong_clean_now:
                # Tiny pressure to fire only at very high-confidence moments.
                reward += self.STAGE2_HESITATION_PENALTY
            elif not clean_now:
                # Tiny patience reward. This is deliberately small so the agent
                # does not learn passivity; it only makes waiting better than
                # firing an obviously unclean shot.
                reward += self.STAGE2_PATIENCE_REWARD
        return reward

    def _stage2_shot_ray_hits_enemy(self, learner: Agent, enemy: Agent, world: World) -> bool:
        if not enemy.is_alive:
            return False
        direction = Vec2(math.cos(learner.heading), math.sin(learner.heading))
        muzzle = learner.position + direction * 20.0
        distance_to_enemy = muzzle.distance_to(enemy.position)
        # The check mirrors the actual Glock line at the moment of firing. It is
        # used only for reward shaping; it does not control or mask the action.
        ray = detection.trace_ray(
            world=world,
            origin=muzzle,
            direction=direction,
            max_length=distance_to_enemy + enemy.radius + 8.0,
            step_size=4.0,
            target_agent_id=enemy.agent_id,
            observer_agent_id=learner.agent_id,
            include_pickups=False,
            include_projectiles=False,
            enemy_margin=3.0,
        )
        return ray.hit_type == detection.HIT_ENEMY and ray.hit_agent_id == enemy.agent_id

    def _shot_event_ray_hits_enemy(self, event: Event, enemy: Agent, world: World) -> bool:
        # Use the enemy position even if the shot killed the enemy during this tick.
        # Otherwise clean kill shots would be incorrectly treated as unclean after
        # damage resolution sets enemy.is_alive to False.
        try:
            position_xy = event.data.get("owner_position")
            heading = float(event.data.get("owner_heading"))
            muzzle_start = Vec2(float(position_xy[0]), float(position_xy[1]))
        except (TypeError, ValueError, IndexError):
            return False
        direction = Vec2(math.cos(heading), math.sin(heading))
        muzzle = muzzle_start + direction * 20.0
        distance_to_enemy = muzzle.distance_to(enemy.position)
        ray = detection.trace_ray(
            world=world,
            origin=muzzle,
            direction=direction,
            max_length=distance_to_enemy + enemy.radius + 8.0,
            step_size=4.0,
            target_agent_id=enemy.agent_id,
            observer_agent_id=event.agent_id,
            include_pickups=False,
            include_projectiles=False,
            enemy_margin=3.0,
        )
        return ray.hit_type == detection.HIT_ENEMY and ray.hit_agent_id == enemy.agent_id

    def _stage3_reward(
        self,
        learner: Agent,
        current_distance: float,
        current_line_of_sight: bool,
        current_angle_error: float,
        moving: bool,
        nav_reference: NavigationReference | None,
        boundary_clearance: float,
    ) -> float:
        # Stage 3 is not just path following. The intended behavior is closed-loop
        # combat tracking: detect the enemy, keep aim aligned, move to the ideal
        # combat distance, and then continuously chase/withdraw as the enemy moves.
        reward = self.STAGE3_AIM_SHAPING_SCALE * self._aim_tracking_reward(
            current_line_of_sight,
            current_angle_error,
            self.STAGE3_AIM_THRESHOLD,
        )

        distance_error_now = abs(current_distance - self.IDEAL_COMBAT_RANGE)
        min_useful_range = self.IDEAL_COMBAT_RANGE - self.STAGE3_RANGE_TOLERANCE
        max_useful_range = self.IDEAL_COMBAT_RANGE + self.STAGE3_RANGE_TOLERANCE
        in_useful_range = min_useful_range <= current_distance <= max_useful_range
        stage3_aligned = current_line_of_sight and current_angle_error < self.STAGE3_AIM_THRESHOLD
        current_speed = learner.velocity.length()
        in_tactical_pocket = (
            current_line_of_sight
            and stage3_aligned
            and abs(current_distance - self.IDEAL_COMBAT_RANGE) <= PLAYER_RADIUS_PX
        )
        slow_or_still_in_pocket = current_speed <= max(8.0, learner.move_speed * 0.10)

        range_quality = self._stage3_range_quality(current_distance)
        aim_quality = self._stage3_aim_quality(current_angle_error)
        visibility_quality = 1.0 if current_line_of_sight else 0.0
        combined_quality = range_quality * aim_quality * visibility_quality
        correcting_range_direction = self._stage3_is_correcting_range(current_distance)
        range_correction_needed = current_line_of_sight and not in_useful_range

        # Mostly additive shaping: the agent receives separate feedback for
        # range, aim, visibility, wall safety, A* progress, and smoothness. This
        # avoids the previous problem where range*aim*visibility could become too
        # small whenever one component was weak.
        reward += self.STAGE3_RANGE_QUALITY_REWARD * range_quality
        if current_line_of_sight:
            reward += self.STAGE3_VISIBILITY_REWARD
            reward += self.STAGE3_AIM_QUALITY_REWARD * aim_quality
        else:
            self.range_streak_steps = 0
            reward -= 0.050

        # Reintroduced moderate combined-quality bonus. It gives full credit
        # only when the agent is actually in useful range. Outside useful range,
        # it gives partial credit only if the agent is moving in the correct
        # chase/withdraw direction. This prevents the deterministic eval policy
        # from farming aim + visibility while standing still.
        if in_useful_range:
            combined_scale = 1.0
        elif moving and correcting_range_direction:
            combined_scale = 0.50
        elif moving:
            combined_scale = 0.20
        else:
            combined_scale = 0.0
        reward += self.STAGE3_COMBINED_REWARD_SCALE * combined_quality * combined_scale

        # Direct distance-control shaping: chase if too far, withdraw
        # if too close, stabilize if already near the ideal distance.
        reward += self._stage3_distance_transition_reward(current_distance, current_line_of_sight)

        if stage3_aligned:
            self.stage3_usable_aim_steps += 1
            self.aligned_consecutive_steps += 1
            self.max_aligned_consecutive_steps = max(
                self.max_aligned_consecutive_steps,
                self.aligned_consecutive_steps,
            )
        else:
            self.aligned_consecutive_steps = 0

        # Keep one small conjunctive bonus for the actual desired combined state.
        # This says: the best state is visible enemy + usable aim + useful range.
        if stage3_aligned and in_useful_range:
            reward += self.STAGE3_COMBINED_TRACKING_BONUS

        # Tactical pocket exemption: once the agent is close to the ideal combat
        # distance and has usable aim, slowing down is not passivity. This helps
        # Stage 3 learn to stabilize instead of constantly overshooting, while
        # keeping anti-freeze pressure active when distance correction is needed.
        if in_tactical_pocket:
            reward += self.STAGE3_TACTICAL_POCKET_REWARD * aim_quality
            if slow_or_still_in_pocket:
                reward += self.STAGE3_TACTICAL_POCKET_SLOW_REWARD * aim_quality

        if boundary_clearance < self.BOUNDARY_SAFE_DISTANCE:
            closeness = 1.0 - max(0.0, boundary_clearance) / max(1.0, self.BOUNDARY_SAFE_DISTANCE)
            reward += self.BOUNDARY_HUGGING_PENALTY * closeness

        # Explicitly discourage rushing directly into the enemy. The desired
        # combat point is near ideal range, not on top of the target.
        if current_distance < min_useful_range:
            too_close_ratio = min(1.0, (min_useful_range - current_distance) / max(1.0, min_useful_range))
            reward += self.TOO_CLOSE_RANGE_PENALTY * too_close_ratio

        if current_line_of_sight and in_useful_range:
            self.useful_range_steps += 1
            self.range_streak_steps += 1
            self.max_range_streak_steps = max(self.max_range_streak_steps, self.range_streak_steps)
            reward += self.STAGE3_IN_RANGE_REWARD * aim_quality
            reward += min(self.STAGE3_RANGE_STREAK_REWARD, self.range_streak_steps * 0.0008) * aim_quality
        elif current_line_of_sight:
            self.range_streak_steps = 0
            normalized_range_error = min(1.0, distance_error_now / max(1.0, self.IDEAL_COMBAT_RANGE))
            # Being out of range is bad, but if the agent is already correcting
            # in the right direction, punish it less harshly so it does not panic
            # and rush/jitter just to remove the range error.
            penalty_scale = 0.35 if correcting_range_direction else 1.0
            reward += self.STAGE3_OUT_OF_RANGE_PENALTY * normalized_range_error * penalty_scale

        # Stage 3 is a movement stage. If the distance is wrong, standing still
        # must be worse than making a correct chase/withdraw adjustment. This
        # directly targets the eval-freeze shortcut seen after aim gating.
        if range_correction_needed:
            if moving and correcting_range_direction:
                reward += self.STAGE3_NEEDED_MOVEMENT_REWARD * (0.65 + 0.35 * aim_quality)
            elif not moving and self.stationary_steps > self.STAGE3_STATIONARY_GRACE_STEPS:
                stationary_scale = min(2.0, self.stationary_steps / max(1.0, self.STAGE3_STATIONARY_GRACE_STEPS * 2.0))
                reward += self.STAGE3_NEEDED_STATIONARY_PENALTY * stationary_scale

        is_good_tracking = stage3_aligned and in_useful_range
        if is_good_tracking:
            self.good_tracking_steps += 1
            reward += self.STAGE3_GOOD_TRACKING_REWARD
        else:
            reward -= 0.006

        # A* is now secondary. It helps the policy reach/correct the desired
        # combat position, but it is down-weighted near ideal range so it does not
        # keep pushing the agent when the correct behavior is to stabilize.
        range_need = min(1.0, distance_error_now / max(1.0, self.STAGE3_RANGE_TOLERANCE))
        aim_gate = self.STAGE3_NAV_AIM_GATE_FLOOR + (1.0 - self.STAGE3_NAV_AIM_GATE_FLOOR) * aim_quality
        range_gate = 0.30 + 0.70 * range_need
        nav_scale = max(self.STAGE3_NAV_MIN_SCALE, self.STAGE3_NAV_MAX_SCALE * aim_gate * range_gate)
        reward += self._navigation_reference_reward(
            learner,
            nav_reference,
            current_line_of_sight,
            current_angle_error,
            scale=nav_scale,
        )

        if moving:
            reward += self.STAGE3_MOVEMENT_ACTIVITY_REWARD
            if current_line_of_sight and current_angle_error > self.STAGE3_VERY_BAD_AIM_THRESHOLD:
                bad_aim_ratio = min(
                    1.0,
                    (current_angle_error - self.STAGE3_VERY_BAD_AIM_THRESHOLD)
                    / max(1e-6, math.pi - self.STAGE3_VERY_BAD_AIM_THRESHOLD),
                )
                reward += self.STAGE3_MOVING_BAD_AIM_PENALTY * (0.5 + 0.5 * bad_aim_ratio)
            if is_good_tracking:
                reward += self.STAGE3_MOVING_TRACKING_BONUS
            if current_line_of_sight and current_angle_error > self.prev_angle_error + 0.050:
                reward += self.AIM_WORSE_PENALTY * 0.7

            previous_speed = self.prev_velocity.length()
            if current_speed > 20.0 and previous_speed > 20.0:
                dot = learner.velocity.x * self.prev_velocity.x + learner.velocity.y * self.prev_velocity.y
                similarity = dot / max(1e-6, current_speed * previous_speed)
                if similarity < -0.25:
                    self.movement_jitter_count += 1
                    self.movement_jitter_streak += 1
                    streak_multiplier = min(3.0, 1.0 + 0.35 * (self.movement_jitter_streak - 1))
                    reward += self.STAGE3_MOVEMENT_JITTER_PENALTY * streak_multiplier
                elif similarity < 0.15 and is_good_tracking:
                    reward += self.STAGE3_SHARP_TURN_PENALTY
                    self.movement_jitter_streak = 0
                else:
                    self.movement_jitter_streak = 0
                    if similarity > 0.55:
                        if correcting_range_direction:
                            reward += self.STAGE3_SMOOTH_CORRECTION_REWARD * (0.70 + 0.30 * aim_quality)
                        elif is_good_tracking:
                            reward += self.STAGE3_SMOOTH_DIRECTION_REWARD
        else:
            self.movement_jitter_streak = 0
            if range_correction_needed:
                # Small immediate cost for not moving when distance is wrong;
                # the stronger anti-freeze penalty above starts after the grace
                # period to avoid punishing brief stabilization pauses.
                reward -= 0.012

        self.prev_velocity = Vec2(learner.velocity.x, learner.velocity.y)
        return reward

    def _navigation_reference_reward(
        self,
        learner: Agent,
        nav_reference: NavigationReference | None,
        current_line_of_sight: bool,
        current_angle_error: float,
        *,
        scale: float,
    ) -> float:
        if nav_reference is None or not nav_reference.valid:
            return 0.0

        self.nav_valid_steps += 1
        self.nav_path_deviation_sum += nav_reference.path_deviation
        self.nav_desired_distance_sum += nav_reference.desired_distance
        self.nav_path_progress_sum += max(0.0, nav_reference.progress_delta)
        self.nav_desired_progress_sum += max(0.0, nav_reference.desired_progress_delta)
        if nav_reference.recomputed:
            self.nav_recompute_count += 1

        reward = 0.0
        progress = max(nav_reference.progress_delta, nav_reference.desired_progress_delta)
        if progress > 0.25:
            self.nav_progress_positive_steps += 1
            reward += min(0.070, progress * self.NAV_PROGRESS_REWARD)
        elif progress < -0.35:
            self.nav_regress_steps += 1
            reward += max(-0.060, (-progress) * self.NAV_REGRESS_PENALTY)

        if nav_reference.waypoint_reached:
            self.nav_waypoint_reached_count += 1
            reward += self.NAV_WAYPOINT_REWARD

        if nav_reference.path_deviation <= 32.0:
            reward += self.NAV_CORRIDOR_REWARD
        elif nav_reference.path_deviation > self.STAGE3_MAX_AVG_PATH_DEVIATION:
            self.nav_path_deviation_bad_steps += 1
            excess = min(1.0, (nav_reference.path_deviation - self.STAGE3_MAX_AVG_PATH_DEVIATION) / self.STAGE3_MAX_AVG_PATH_DEVIATION)
            reward += self.NAV_DEVIATION_PENALTY * excess

        speed = learner.velocity.length()
        if speed > 12.0 and nav_reference.waypoint_distance > 1.0:
            to_waypoint = Vec2(nav_reference.waypoint.x - learner.position.x, nav_reference.waypoint.y - learner.position.y)
            direction_alignment = (learner.velocity.x * to_waypoint.x + learner.velocity.y * to_waypoint.y) / max(1e-6, speed * to_waypoint.length())
            if direction_alignment > 0.35:
                self.nav_direction_good_steps += 1
                reward += self.NAV_DIRECTION_REWARD * direction_alignment
            elif direction_alignment < -0.20:
                reward += self.NAV_WRONG_DIRECTION_PENALTY * abs(direction_alignment)

        if self.stationary_steps > 18 and nav_reference.desired_distance > 55.0:
            reward += self.NAV_STUCK_PENALTY

        if nav_reference.desired_distance < 55.0 and current_line_of_sight and current_angle_error < self.STAGE3_AIM_THRESHOLD:
            reward += 0.040

        return reward * scale

    def _stage3_is_correcting_range(self, current_distance: float) -> bool:
        """Return True when the distance change is already moving toward epsilon.

        This is used only to soften the out-of-range penalty. It does not update
        Stage 3 metrics, because metric accounting is handled in
        _stage3_distance_transition_reward.
        """
        d0 = self.prev_distance
        d1 = current_distance
        epsilon = self.IDEAL_COMBAT_RANGE
        ideal_band = PLAYER_RADIUS_PX
        delta = d1 - d0
        if d0 < epsilon - ideal_band:
            return delta > self.STAGE3_RANGE_CONTROL_DEADZONE
        if d0 > epsilon + ideal_band:
            return delta < -self.STAGE3_RANGE_CONTROL_DEADZONE
        return False

    def _stage3_distance_transition_reward(self, current_distance: float, current_line_of_sight: bool) -> float:
        """Closed-loop distance-control shaping for Stage 3.

        d0 is the previous enemy distance, d1 is the current distance, and epsilon
        is the ideal combat distance. The reward teaches three tactical actions:
        chase when too far, withdraw when too close, and stabilize when already
        near epsilon. Stable ideal distance is deliberately rewarded more than
        correction so the agent does not farm in/out oscillations.
        """
        if not current_line_of_sight:
            return 0.0

        d0 = self.prev_distance
        d1 = current_distance
        epsilon = self.IDEAL_COMBAT_RANGE
        ideal_band = PLAYER_RADIUS_PX
        danger_distance = PLAYER_DIAMETER_PX * 2.0
        delta = d1 - d0  # positive means distance increased; negative means distance decreased
        abs_delta = abs(delta)
        self.stage3_range_drift_sum += abs_delta

        if d1 <= danger_distance:
            self.stage3_wrong_range_control_steps += 1
            return self.STAGE3_DANGER_CLOSE_PENALTY

        reward = 0.0
        in_ideal_band = abs(d1 - epsilon) <= ideal_band
        was_too_close = d0 < epsilon - ideal_band
        was_too_far = d0 > epsilon + ideal_band

        # The ideal state: stay close to epsilon without unnecessary oscillation.
        if in_ideal_band:
            self.stage3_range_control_opportunities += 1
            if abs_delta <= self.STAGE3_STABLE_RANGE_DELTA:
                self.stage3_correct_range_control_steps += 1
                self.stage3_stable_range_steps += 1
                reward += self.STAGE3_STABLE_RANGE_REWARD
            else:
                self.stage3_wrong_range_control_steps += 1
                reward += self.STAGE3_RANGE_STABILITY_DRIFT_PENALTY
            return reward

        # If the previous state was already too close, the correct response is
        # to increase distance. If it was too far, the correct response is to
        # decrease distance. A small deadzone avoids rewarding numerical noise.
        if was_too_close:
            self.stage3_range_control_opportunities += 1
            if delta > self.STAGE3_RANGE_CONTROL_DEADZONE:
                self.stage3_correct_range_control_steps += 1
                reward += self.STAGE3_RANGE_CONTROL_REWARD
            elif delta < -self.STAGE3_RANGE_CONTROL_DEADZONE:
                self.stage3_wrong_range_control_steps += 1
                reward += self.STAGE3_RANGE_CONTROL_PENALTY
            else:
                reward -= 0.004
        elif was_too_far:
            self.stage3_range_control_opportunities += 1
            if delta < -self.STAGE3_RANGE_CONTROL_DEADZONE:
                self.stage3_correct_range_control_steps += 1
                reward += self.STAGE3_RANGE_CONTROL_REWARD
            elif delta > self.STAGE3_RANGE_CONTROL_DEADZONE:
                self.stage3_wrong_range_control_steps += 1
                reward += self.STAGE3_RANGE_CONTROL_PENALTY
            else:
                reward -= 0.004

        return reward

    def _stage3_range_quality(self, current_distance: float) -> float:
        # Smooth 0..1 score around the ideal distance. The useful-range band is
        # not binary; near-misses still produce learning signal.
        distance_error = abs(current_distance - self.IDEAL_COMBAT_RANGE)
        soft_width = max(1.0, self.STAGE3_RANGE_TOLERANCE * 2.5)
        return max(0.0, min(1.0, 1.0 - distance_error / soft_width))

    def _stage3_aim_quality(self, current_angle_error: float) -> float:
        # Exponential aim quality keeps the reward differentiable-like for PPO:
        # perfect aim is near 1, poor aim smoothly approaches 0.
        return max(0.0, min(1.0, math.exp(-self.STAGE3_AIM_QUALITY_DECAY * max(0.0, current_angle_error))))

    def _stage4_reward(
        self,
        learner: Agent,
        current_distance: float,
        current_line_of_sight: bool,
        moving: bool,
        threat: Any,
        step_count: int,
        boundary_clearance: float,
    ) -> float:
        """Dense Stage 4 shaping for learning real projectile dodging."""
        reward = 0.0
        danger = bool(getattr(threat, "danger", False))
        threat_exists = bool(getattr(threat, "exists", False))

        if danger:
            if moving:
                reward += 0.030

                if learner.velocity.length() > 1e-6:
                    move_dir = learner.velocity.normalize()
                    projectile_angle = learner.heading + float(getattr(threat, "direction_angle", 0.0))
                    projectile_dir = Vec2(math.cos(projectile_angle), math.sin(projectile_angle))
                    dot = move_dir.x * projectile_dir.x + move_dir.y * projectile_dir.y
                    lateral_quality = 1.0 - abs(max(-1.0, min(1.0, dot)))
                    reward += 0.035 * lateral_quality
                    if lateral_quality < 0.20:
                        reward -= 0.010
            else:
                reward -= 0.055
        elif threat_exists and moving:
            reward += 0.003

        if step_count > 120 and self.projectile_danger_steps < 3:
            reward -= 0.015

        if current_distance > 950.0:
            reward -= 0.020
        elif current_distance > 850.0:
            reward -= 0.008

        if boundary_clearance < self.BOUNDARY_SAFE_DISTANCE:
            closeness = 1.0 - max(0.0, boundary_clearance) / max(1.0, self.BOUNDARY_SAFE_DISTANCE)
            reward += self.STAGE4_BOUNDARY_PENALTY * closeness
            if danger:
                reward += self.STAGE4_DANGER_BOUNDARY_PENALTY * closeness

        if moving and current_line_of_sight:
            reward += 0.004
        if self.stationary_steps > 45:
            reward -= 0.018

        if learner.is_alive and step_count >= 460 and self.damage_taken <= 0.0 and self.projectile_dodges == 0:
            reward += self.STAGE4_FAKE_SURVIVAL_PENALTY

        return reward

    def _stage5_reward(self, events: list[Event], learner: Agent, current_distance: float, current_line_of_sight: bool, current_angle_error: float, moving: bool, nav_reference: NavigationReference | None) -> float:
        """Combat-centered Stage 5 shaping.

        Stage 5 should not reward movement by itself. It should reward hits,
        damage, and shooting decisions that happen while the learner is actually
        moving. Aim/range shaping is kept small and gated by visible combat so
        the policy cannot farm passive positioning rewards.
        """
        reward = 0.0
        speed_ratio = 0.0
        if learner.move_speed > 1e-6:
            speed_ratio = min(1.0, learner.velocity.length() / learner.move_speed)
        moving_fast_enough = moving and speed_ratio >= self.STAGE5_MIN_MOVING_SPEED_RATIO
        useful_range = 160.0 <= current_distance <= 340.0

        if current_line_of_sight and moving_fast_enough:
            if current_angle_error + 0.025 < self.prev_angle_error:
                reward += self.STAGE5_COMBAT_AIM_SUPPORT
            if current_angle_error < self.STAGE2_ALIGN_THRESHOLD:
                reward += self.STAGE5_COMBAT_AIM_SUPPORT
            if useful_range:
                reward += self.STAGE5_COMBAT_RANGE_SUPPORT

        if current_line_of_sight and self.shots_fired == 0 and self.aim_error_count > 90 and current_angle_error < self.STAGE2_ALIGN_THRESHOLD:
            reward += self.STAGE5_HESITATION_PENALTY

        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "shot_fired":
                shot_was_moving = bool(event.data.get("moving_shot", False)) and moving_fast_enough
                if not shot_was_moving:
                    reward += self.STAGE5_STATIONARY_SHOT_PENALTY
                if not current_line_of_sight or current_angle_error > 0.35:
                    self.bad_shots += 1
                    reward += self.BAD_SHOT_PENALTY * 0.85
                elif current_angle_error < self.STAGE2_FIRE_THRESHOLD and shot_was_moving:
                    reward += self.STAGE5_ALIGNED_MOVING_SHOT_REWARD
            elif event.kind == "shot_hit":
                shot_was_moving = bool(event.data.get("moving_shot", False))
                if shot_was_moving:
                    # Capped speed scaling prevents a sprint-at-all-costs policy.
                    speed_bonus = min(1.0, max(0.0, speed_ratio) / 0.60)
                    reward += self.STAGE5_MOVING_HIT_BONUS * max(0.55, speed_bonus)
                else:
                    reward += self.STAGE5_STATIONARY_HIT_PENALTY

        return reward

    def _stage6_combat_refinement(self, events: list[Event], learner: Agent, current_line_of_sight: bool, current_angle_error: float, moving: bool) -> float:
        reward = 0.0
        phase_index = max(1, min(4, int(getattr(self, "stage6_curriculum_phase_index", 4) or 4)))

        # Small support for controlled pressure: moving, visible, and still aimed.
        # This is not a movement reward by itself; it only applies in a live duel
        # state that can produce usable shots.
        if moving and current_line_of_sight and current_angle_error < 0.22:
            reward += self.STAGE6_ALIGNED_MOVING_PRESSURE_REWARD

        # Do not let the Stage 5 policy solve Stage 6 by eating damage and racing
        # for the kill.  The penalty becomes stricter in the final phase.
        soft_damage_limit = 84.0 if phase_index <= 3 else 63.0
        if self.damage_taken > soft_damage_limit:
            reward += self.STAGE6_LOW_DAMAGE_PENALTY * min(8.0, (self.damage_taken - soft_damage_limit) / 21.0)

        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "damage_taken":
                reward += self.STAGE6_PROJECTILE_HIT_EXTRA_PENALTY
            elif event.kind == "shot_failed_no_ammo":
                reward += self.STAGE6_EMPTY_TRIGGER_EXTRA_PENALTY
            elif event.kind in {"reload_failed_already_reloading", "reload_failed_full_ammo"}:
                reward += self.STAGE6_RELOAD_FAILURE_EXTRA_PENALTY
            elif event.kind == "kill" and self.damage_taken <= soft_damage_limit:
                reward += self.STAGE6_CLEAN_DAMAGE_BONUS * max(0.0, soft_damage_limit - self.damage_taken)

        return reward

    def _combat_reward(self, learner: Agent, current_distance: float, current_line_of_sight: bool, current_angle_error: float, moving: bool, obstacles: bool, nav_reference: NavigationReference | None = None) -> float:
        reward = self._aim_tracking_reward(current_line_of_sight, current_angle_error, self.STAGE2_ALIGN_THRESHOLD)
        reward += self._movement_position_reward(current_distance, current_line_of_sight)
        if obstacles and self.stage_id == "stage7":
            phase_index = max(1, min(5, int(getattr(self, "stage7_curriculum_phase_index", 5) or 5)))
            # Obstacle stages need stronger path guidance than open combat,
            # especially while LOS is lost.  This keeps Stage 7 closer to the
            # Stage 3 movement objective: make path progress, do not just move.
            if not current_line_of_sight:
                nav_scale = 0.55 if phase_index <= 2 else 0.45
            else:
                nav_scale = 0.34 if phase_index <= 2 else 0.30
        else:
            nav_scale = 0.45 if obstacles else 0.30
        # Later movement/combat stages reuse the same reference.  It remains a
        # reward guide only; PPO still chooses move/strafe/rotate/shoot/reload.
        reward += self._navigation_reference_reward(learner, nav_reference, current_line_of_sight, current_angle_error, scale=nav_scale)
        if not current_line_of_sight:
            self.passive_steps += 1
            if self.passive_steps > (70 if obstacles else 45):
                reward += self.PASSIVE_HIDING_PENALTY
            if obstacles:
                if moving:
                    reward += self.STAGE7_SEARCH_MOVEMENT_REWARD
                elif self.los_lost_streak > self.STAGE7_NO_LOS_GRACE_STEPS:
                    reward += self.STAGE7_NO_LOS_STATIONARY_PENALTY
                if self.los_lost_streak > 120:
                    reward += self.STAGE7_LONG_NO_LOS_EXTRA_PENALTY
        else:
            self.passive_steps = 0
        if moving:
            reward += 0.004
        if self.shots_fired == 0 and self.aim_error_count > 100 and current_line_of_sight:
            reward -= 0.010
        return reward

    def _obstacle_visibility_reward(self, current_line_of_sight: bool, moving: bool = False) -> float:
        reward = 0.0
        regained_los = current_line_of_sight and not self.prev_line_of_sight
        if current_line_of_sight and self.los_lost_streak == 0 and self.line_of_sight_gained_count > 0 and self.los_recovery_cooldown == 0:
            # Small capped reward; cooldown prevents line-of-sight farming.
            reward += self.REGAIN_LOS_REWARD
            if regained_los or moving:
                reward += self.STAGE7_ACTIVE_LOS_REGAIN_BONUS
            self.los_recovery_cooldown = 45
        if not current_line_of_sight and self.los_lost_streak > 80:
            reward += self.LOS_LOSS_PENALTY
        return reward

    def _stage7_active_obstacle_reward(
        self,
        events: list[Event],
        learner: Agent,
        enemy: Agent,
        world: World,
        current_line_of_sight: bool,
        current_clear_shot: bool,
        current_angle_error: float,
        moving: bool,
        nav_reference: NavigationReference | None = None,
    ) -> float:
        """Extra Stage 7 shaping for active obstacle navigation and visible aiming.

        The base combat reward already covers kills, damage, LOS recovery, and
        generic navigation.  This refinement targets the failure seen in logs:
        the agent moves without making path progress, then sees the enemy but
        fails to aim/fire aggressively enough.
        """
        if self.stage_id != "stage7":
            return 0.0

        reward = 0.0
        shot_fired_this_step = any(event.agent_id == learner.agent_id and event.kind == "shot_fired" for event in events)
        hit_this_step = any(event.agent_id == learner.agent_id and event.kind == "shot_hit" for event in events)
        weapon_ready = bool(
            learner.is_alive
            and learner.ammo > 0
            and learner.fire_ready >= 0.5
            and not getattr(learner, "reloading", False)
        )

        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "shot_blocked":
                reward += self.STAGE7_BLOCKED_SHOT_EXTRA_PENALTY
            elif event.kind == "shot_fired" and not hit_this_step and not self._shot_event_ray_hits_enemy(event, enemy, world):
                # This should be rare because the environment masks/blocks unclean
                # Stage 7 shots. Keep a strong backup penalty for non-mask callers.
                reward += self.STAGE7_UNCLEAN_SHOT_ATTEMPT_PENALTY
            elif event.kind == "shot_hit":
                reward += self.STAGE7_CLEAN_HIT_BONUS

        if current_line_of_sight:
            aim_quality = self._stage3_aim_quality(current_angle_error)
            if current_angle_error + 0.025 < self.prev_angle_error:
                reward += self.STAGE7_VISIBLE_AIM_IMPROVE_REWARD
            elif current_angle_error > self.prev_angle_error + 0.035:
                reward += self.STAGE7_VISIBLE_AIM_WORSE_PENALTY

            reward += 0.018 * aim_quality
            if current_clear_shot:
                reward += self.STAGE7_CLEAR_SHOT_REWARD * max(0.35, aim_quality)
                if not self.prev_clear_shot:
                    reward += self.STAGE7_CLEAR_SHOT_GAIN_BONUS
            else:
                reward += self.STAGE7_LOS_NO_CLEAR_SHOT_PENALTY
                if weapon_ready and not moving:
                    reward += self.STAGE7_LOS_NO_CLEAR_SHOT_PENALTY

            if current_angle_error < self.STAGE1_ALIGN_THRESHOLD:
                reward += self.STAGE7_VISIBLE_ALIGNED_REWARD
            elif current_angle_error > self.STAGE3_VERY_BAD_AIM_THRESHOLD:
                bad_ratio = min(
                    1.0,
                    (current_angle_error - self.STAGE3_VERY_BAD_AIM_THRESHOLD)
                    / max(1e-6, math.pi - self.STAGE3_VERY_BAD_AIM_THRESHOLD),
                )
                reward += self.STAGE7_VISIBLE_BAD_AIM_PENALTY * (0.5 + 0.5 * bad_ratio)

            if weapon_ready and current_clear_shot and not shot_fired_this_step:
                if current_angle_error < self.STAGE2_ALIGN_THRESHOLD:
                    reward += self.STAGE7_READY_CLEAR_NO_SHOT_PENALTY
                elif current_angle_error < self.STAGE3_AIM_THRESHOLD:
                    reward += self.STAGE7_READY_VISIBLE_NO_SHOT_PENALTY
            return reward

        if nav_reference is not None and nav_reference.valid:
            progress = max(nav_reference.progress_delta, nav_reference.desired_progress_delta)
            if progress > 0.25:
                reward += min(0.090, progress * self.STAGE7_LOS_NAV_PROGRESS_REWARD)
            elif progress < -0.35:
                reward += max(-0.090, (-progress) * self.STAGE7_LOS_NAV_REGRESS_PENALTY)

            if nav_reference.path_deviation <= 32.0:
                reward += self.STAGE7_NAV_CORRIDOR_BONUS
            if moving and progress < 0.05 and self.los_lost_streak > 40:
                reward += self.STAGE7_MOVING_WITHOUT_NAV_PROGRESS_PENALTY
        elif moving and self.los_lost_streak > 60:
            # Movement without a usable path reference should not be treated as
            # good obstacle search forever.
            reward += self.STAGE7_MOVING_WITHOUT_NAV_PROGRESS_PENALTY * 0.5

        return reward

    def _stage8_tactical_reload_reward(
        self,
        events: list[Event],
        learner: Agent,
        enemy: Agent,
        world: World,
        current_line_of_sight: bool,
        current_clear_shot: bool,
        current_angle_error: float,
        moving: bool,
        heading_delta: float,
        projectile_danger: bool,
        nav_reference: NavigationReference | None = None,
    ) -> float:
        """Stage 8 shaping: tactical reload plus stricter retained aim/movement.

        Stage 8 no longer uses map pickups or health regain. The new skill is to
        reload before the Glock reaches empty, preferably around one third of the
        magazine, without abandoning clean combat behavior learned in Stage 7.
        """
        reward = 0.0
        ammo_ratio = learner.ammo_ratio
        target = self.STAGE8_TARGET_RELOAD_RATIO
        phase_index = max(1, min(5, int(getattr(self, "stage8_curriculum_phase_index", 5) or 5)))
        is_phase_8a = phase_index <= 1
        target_ammo_now = max(1, int(round(float(max(1, learner.max_ammo)) / 3.0)))
        shot_fired_this_step = any(event.agent_id == learner.agent_id and event.kind == "shot_fired" for event in events)
        hit_this_step = any(event.agent_id == learner.agent_id and event.kind == "shot_hit" for event in events)
        reload_started_this_step = any(event.agent_id == learner.agent_id and event.kind == "reload_started" for event in events)
        weapon_ready = bool(learner.is_alive and learner.ammo > 0 and learner.fire_ready >= 0.5 and not getattr(learner, "reloading", False))
        killable_now = bool(current_clear_shot and current_angle_error <= self.STAGE8_ALIGNMENT_THRESHOLD and enemy.health <= learner.projectile_damage + 1e-6)

        if self.stage8_reload_started > 0 and self.stage8_shots_after_reload == 0 and not getattr(learner, "reloading", False):
            # Reloading is only useful if the bot returns to combat afterwards.
            reward += self.STAGE8_NO_POST_RELOAD_COMBAT_PENALTY

        if ammo_ratio <= 0.0 and not getattr(learner, "reloading", False):
            self.stage8_empty_ammo_steps += 1
        if 0.0 < ammo_ratio <= target:
            self.stage8_low_ammo_steps += 1
            if not reload_started_this_step and not projectile_danger and not killable_now:
                reward += self.STAGE8_LOW_AMMO_NO_RELOAD_PENALTY
        if 0.0 < ammo_ratio <= 0.16:
            self.stage8_critical_ammo_steps += 1
            if not reload_started_this_step and not projectile_danger:
                reward += self.STAGE8_CRITICAL_AMMO_NO_RELOAD_PENALTY

        if is_phase_8a and self.shots_fired <= 0 and not shot_fired_this_step and not getattr(learner, "reloading", False):
            # 8A is a reload-sequence tutor, not an aim-hold stage.  After a short
            # grace period the policy is pushed to fire the first round so it can
            # reach the 3/9 reload target.
            if self.survival_steps > self.STAGE8_8A_NO_FIRST_SHOT_GRACE_STEPS:
                self.stage8_no_first_shot_pressure_steps += 1
                overdue_blocks = min(5.0, (self.survival_steps - self.STAGE8_8A_NO_FIRST_SHOT_GRACE_STEPS) / 40.0)
                reward += self.STAGE8_8A_NO_FIRST_SHOT_BASE_PENALTY * (1.0 + overdue_blocks)
            if weapon_ready and current_line_of_sight and current_clear_shot and current_angle_error <= self.STAGE8_ALIGNMENT_THRESHOLD:
                reward += self.STAGE8_8A_ALIGNED_NO_FIRST_SHOT_PENALTY

        if is_phase_8a and not getattr(learner, "reloading", False):
            # In 8A the main sub-skill is the precise unmasked reload sequence.
            # Reaching 3/9 after at least one shot is explicitly rewarded once,
            # while sitting there without reloading is penalized.
            if learner.ammo == target_ammo_now and self.shots_fired >= 1 and self.stage8_reload_started <= 0:
                self.stage8_target_ammo_steps += 1
                if not self.stage8_target_ammo_reached:
                    self.stage8_target_ammo_reached = True
                    reward += self.STAGE8_8A_REACH_TARGET_AMMO_REWARD
                else:
                    reward += self.STAGE8_8A_TARGET_AMMO_HOLD_REWARD
                if not reload_started_this_step and not projectile_danger:
                    reward += self.STAGE8_8A_TARGET_NO_RELOAD_PENALTY
                    if self.stage8_target_ammo_steps > self.STAGE8_8A_TARGET_NO_RELOAD_TIMEOUT_STEPS:
                        self.stage8_target_no_reload_timeout_steps += 1
                        reward += self.STAGE8_8A_TARGET_NO_RELOAD_TIMEOUT_PENALTY
            elif learner.ammo <= 0 and self.stage8_reload_started <= 0:
                reward += self.STAGE8_8A_EMPTY_AMMO_STEP_PENALTY

        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "reload_started":
                self.stage8_reload_started += 1
                before_ratio = float(event.data.get("ammo_ratio_before", ammo_ratio))
                ammo_before = int(event.data.get("ammo_before", round(before_ratio * max(1, learner.max_ammo))))
                max_ammo_before = int(event.data.get("max_ammo", learner.max_ammo))
                target_ammo = max(1, int(round(float(max_ammo_before) / 3.0)))

                # Stage 8 is intentionally strict now. With 4/9 starting ammo,
                # reloading at 4 is still too early; the desired first reload is
                # after spending at least one round, at exactly 3/9. This avoids
                # the learned shortcut "spawn -> reload".
                premature_reload = self.shots_fired < self.STAGE8_MIN_SHOTS_BEFORE_RELOAD
                if premature_reload:
                    self.stage8_early_reloads += 1
                    self.stage8_reload_quality_sum += 0.0
                    reward += self.STAGE8_PREMATURE_RELOAD_PENALTY
                    if is_phase_8a:
                        reward += self.STAGE8_8A_EARLY_RELOAD_EXTRA_PENALTY
                elif ammo_before <= 0:
                    self.stage8_empty_reloads += 1
                    self.stage8_reload_quality_sum += 0.0
                    reward += self.STAGE8_EMPTY_RELOAD_PENALTY
                    if is_phase_8a:
                        reward += self.STAGE8_8A_EMPTY_SHOT_EXTRA_PENALTY
                elif ammo_before == target_ammo:
                    self.stage8_good_reloads += 1
                    self.stage8_reload_quality_sum += 1.0
                    # Safer reloads are preferred: no clear shot or not aligned.
                    safety_scale = 1.0 if not current_clear_shot else 0.70
                    if killable_now:
                        safety_scale = 0.25
                    reward += self.STAGE8_GOOD_RELOAD_REWARD * safety_scale
                    if is_phase_8a:
                        reward += self.STAGE8_8A_GOOD_RELOAD_EXTRA_REWARD
                elif ammo_before == target_ammo - 1:
                    self.stage8_acceptable_reloads += 1
                    self.stage8_reload_quality_sum += 0.55
                    reward += self.STAGE8_ACCEPTABLE_RELOAD_REWARD
                elif ammo_before > target_ammo:
                    self.stage8_early_reloads += 1
                    self.stage8_reload_quality_sum += 0.0
                    reward += self.STAGE8_EARLY_RELOAD_PENALTY
                    if is_phase_8a:
                        reward += self.STAGE8_8A_EARLY_RELOAD_EXTRA_PENALTY
                else:
                    self.stage8_late_reloads += 1
                    self.stage8_reload_quality_sum += 0.20
                    reward += self.STAGE8_LATE_RELOAD_PENALTY

                if killable_now:
                    self.stage8_reload_while_killable += 1
                    reward += self.STAGE8_RELOAD_WHILE_KILLABLE_PENALTY
            elif event.kind == "shot_fired":
                # Stage 8 final integration: bad tactical shots are no longer
                # masked.  The policy may fire without LOS/clear shot or while
                # badly aimed, but these choices are explicitly penalized and
                # logged so the skill is learned rather than hard-blocked.
                ammo_before_shot = int(event.data.get("ammo_before", learner.ammo + 1))
                if is_phase_8a and self.stage8_reload_started <= 0 and not self.stage8_first_shot_rewarded and ammo_before_shot == target_ammo_now + 1:
                    self.stage8_first_shot_rewarded = True
                    reward += self.STAGE8_8A_FIRST_SHOT_REWARD
                if is_phase_8a and self.stage8_reload_started <= 0 and ammo_before_shot <= target_ammo_now:
                    self.stage8_shot_past_target_before_reload += 1
                    reward += self.STAGE8_8A_SHOOT_PAST_TARGET_PENALTY
                if not current_line_of_sight or not current_clear_shot:
                    self.stage8_no_los_shot_attempts += 1
                    reward += self.STAGE8_NO_LOS_SHOT_ATTEMPT_PENALTY
                if current_angle_error > self.STAGE8_ALIGNMENT_THRESHOLD:
                    self.stage8_unaligned_shot_attempts += 1
                    reward += self.STAGE8_UNALIGNED_SHOT_ATTEMPT_PENALTY
                if self.stage8_reload_started > 0:
                    self.stage8_shots_after_reload += 1
                    reward += self.STAGE8_SHOT_AFTER_RELOAD_REWARD
                    if is_phase_8a:
                        reward += self.STAGE8_8A_POST_RELOAD_SHOT_EXTRA_REWARD
            elif event.kind == "shot_failed_no_ammo":
                reward += self.STAGE8_EMPTY_TRIGGER_EXTRA_PENALTY
                if is_phase_8a:
                    reward += self.STAGE8_8A_EMPTY_SHOT_EXTRA_PENALTY
            elif event.kind == "shot_blocked":
                reward += self.STAGE7_BLOCKED_SHOT_EXTRA_PENALTY
            elif event.kind == "shot_hit":
                if current_line_of_sight and current_clear_shot and current_angle_error <= self.STAGE8_ALIGNMENT_THRESHOLD:
                    self.stage8_clean_aligned_shots += 1
                    reward += self.STAGE8_CLEAN_ALIGNED_SHOT_REWARD

        if current_line_of_sight:
            aim_reward_scale = self.STAGE8_8A_AIM_REWARD_SCALE_BEFORE_FIRST_SHOT if (is_phase_8a and self.shots_fired <= 0 and not shot_fired_this_step) else 1.0
            if current_angle_error + 0.020 < self.prev_angle_error:
                self.stage8_aim_improve_steps += 1
                reward += self.STAGE8_AIM_IMPROVE_REWARD * aim_reward_scale
            elif current_angle_error > self.prev_angle_error + 0.030:
                self.stage8_aim_worse_steps += 1
                reward += self.STAGE8_AIM_WORSE_PENALTY

            if current_clear_shot and current_angle_error <= self.STAGE8_STRICT_ALIGNMENT_THRESHOLD:
                self.stage8_aim_hold_steps += 1
                reward += self.STAGE8_AIM_HOLD_REWARD * aim_reward_scale
            elif current_angle_error > self.STAGE3_VERY_BAD_AIM_THRESHOLD:
                reward += self.STAGE8_VISIBLE_BAD_AIM_PENALTY

            if current_clear_shot and not self.prev_clear_shot:
                reward += self.STAGE8_CLEAR_SHOT_GAIN_BONUS
            if weapon_ready and current_clear_shot and current_angle_error <= self.STAGE8_ALIGNMENT_THRESHOLD and not shot_fired_this_step and not reload_started_this_step:
                reward += self.STAGE8_CLEAR_ALIGNED_NO_SHOT_PENALTY

        # Smoothness pressure. Do not punish rapid steering while the bot is in projectile danger.
        abs_heading_delta = abs(heading_delta)
        if abs_heading_delta > 0.075:
            self.stage8_large_turn_count += 1
            if not projectile_danger:
                reward += self.STAGE8_LARGE_TURN_PENALTY
        rotation_sign = 1 if heading_delta > 0.010 else (-1 if heading_delta < -0.010 else 0)
        if rotation_sign != 0 and self.prev_rotation_sign != 0 and rotation_sign != self.prev_rotation_sign:
            self.stage8_turn_flip_count += 1
            if not projectile_danger:
                reward += self.STAGE8_ROTATION_FLIP_PENALTY
        if rotation_sign != 0:
            self.prev_rotation_sign = rotation_sign

        if learner.blocked_last_step:
            reward += self.STAGE8_BLOCKED_MOVE_EXTRA_PENALTY
            if rotation_sign != 0 and self.prev_rotation_sign != 0 and rotation_sign != self.prev_rotation_sign and not projectile_danger:
                reward += self.STAGE8_BLOCKED_TURN_FLIP_EXTRA_PENALTY
            if abs_heading_delta > 0.050 and not projectile_danger:
                self.stage8_blocked_spin_steps += 1
                reward += self.STAGE8_BLOCKED_SPIN_PENALTY
            if self.consecutive_blocked_moves >= 2:
                reward += self.STAGE8_REPEATED_BLOCK_EXTRA_PENALTY * min(3.0, self.consecutive_blocked_moves)

        current_speed = learner.velocity.length()
        previous_speed = self.prev_velocity.length()
        if moving and current_speed > 20.0 and previous_speed > 20.0:
            dot = learner.velocity.x * self.prev_velocity.x + learner.velocity.y * self.prev_velocity.y
            similarity = dot / max(1e-6, current_speed * previous_speed)
            if similarity < -0.20:
                self.stage8_movement_flip_count += 1
                if not projectile_danger:
                    reward += self.STAGE8_MOVEMENT_FLIP_PENALTY
            elif similarity > 0.65 and not learner.blocked_last_step:
                if nav_reference is not None and nav_reference.valid and max(nav_reference.progress_delta, nav_reference.desired_progress_delta) > 0.10:
                    self.stage8_smooth_move_steps += 1
                    reward += self.STAGE8_SMOOTH_MOVE_REWARD

        return reward

    def _pickup_reward(self, events: list[Event], learner: Agent) -> float:
        reward = 0.0
        for event in events:
            if event.agent_id != learner.agent_id:
                continue
            if event.kind == "useful_pickup_collected":
                reward += self.USEFUL_PICKUP_REWARD
            elif event.kind == "useless_pickup_collected":
                reward += self.USELESS_PICKUP_PENALTY
        return reward

    def _aim_tracking_reward(self, current_line_of_sight: bool, current_angle_error: float, alignment_threshold: float) -> float:
        reward = 0.0
        if current_line_of_sight and current_angle_error + 0.025 < self.prev_angle_error:
            reward += self.AIM_IMPROVEMENT_REWARD
        elif current_angle_error > self.prev_angle_error + 0.025:
            reward += self.AIM_WORSE_PENALTY
        if current_line_of_sight and current_angle_error < alignment_threshold:
            reward += self.ALIGNED_HOLD_REWARD
        return reward

    def _movement_position_reward(self, current_distance: float, current_line_of_sight: bool) -> float:
        previous_error = abs(self.prev_distance - self.IDEAL_COMBAT_RANGE)
        current_error = abs(current_distance - self.IDEAL_COMBAT_RANGE)
        if current_line_of_sight and current_error + 5.0 < previous_error:
            return self.POSITION_IMPROVE_REWARD
        if current_line_of_sight and current_error > previous_error + 10.0:
            return self.POSITION_WORSE_PENALTY
        return 0.0

    @staticmethod
    def _boundary_clearance(learner: Agent, world: World) -> float:
        return min(
            learner.position.x - learner.radius,
            learner.position.y - learner.radius,
            float(world.width) - learner.position.x - learner.radius,
            float(world.height) - learner.position.y - learner.radius,
        )

    def episode_success(self) -> bool:
        return self.stage_success_metric() >= 1.0

    def stage8_8a_sequence_success(self) -> bool:
        """True once the 8A reload-tutor sequence has been completed.

        8A should teach only the basic reload timing chain:
        fire once from 4/9, reach 3/9, reload at 3/9, then fire after reload.
        Once this happens cleanly, the episode should end so later accidental
        early/empty reloads cannot erase the learned sequence.
        """
        if self.stage_id != "stage8":
            return False
        phase_index = max(1, min(5, int(getattr(self, "stage8_curriculum_phase_index", 5) or 5)))
        if phase_index > 1:
            return False
        return (
            bool(self.stage8_first_shot_rewarded)
            and bool(self.stage8_target_ammo_reached)
            and self.stage8_good_reloads >= 1
            and self.stage8_shots_after_reload >= 1
            and self.stage8_early_reloads == 0
            and self.stage8_empty_reloads == 0
            and self.stage8_shot_past_target_before_reload == 0
        )

    def stage_success_metric(self) -> float:
        metric, criteria = self._stage_metric_and_criteria()
        self.last_criteria = criteria
        return metric

    def _stage_metric_and_criteria(self) -> tuple[float, dict[str, tuple[bool, str]]]:
        steps = max(1, self.aim_error_count)
        hit_rate = self.hit_rate()
        bad_rate = self.bad_shot_rate()
        visible_ratio = self.enemy_visible_ratio(steps)
        time_on_target_ratio = self.time_on_target_ratio(steps)
        mean_aim_error = self.aim_error_mean()
        blocked_rate = self.blocked_moves / steps
        survival_ratio = self.survival_steps / steps
        damage_ratio = self.damage_dealt_taken_ratio()
        movement_ratio = min(1.0, self.movement_distance_total / 250.0)
        moving_shot_rate = self.moving_shot_rate()
        avoidance_rate = self.projectile_avoidance_rate()

        criteria: dict[str, tuple[bool, str]] = {}

        def add(name: str, ok: bool, detail: str) -> None:
            criteria[name] = (bool(ok), detail)

        if self.stage_id == "stage1":
            full_episode = steps == self.STAGE1_REQUIRED_STEPS
            no_invalid_actions = self.invalid_actions == 0

            metric = min(
                self.max_aligned_consecutive_steps / self.STAGE1_REQUIRED_CONSEC,
                time_on_target_ratio / self.STAGE1_TARGET_RATIO,
                self.STAGE1_MAX_MEAN_ERROR / max(mean_aim_error, 1e-6),
                steps / self.STAGE1_REQUIRED_STEPS,
                1.0 if no_invalid_actions else 0.0,
            )

            add(
                "mean_aim_error",
                mean_aim_error <= self.STAGE1_MAX_MEAN_ERROR,
                f"{mean_aim_error:.3f} <= {self.STAGE1_MAX_MEAN_ERROR:.3f}",
            )
            add(
                "time_on_target_ratio",
                time_on_target_ratio >= self.STAGE1_TARGET_RATIO,
                f"{time_on_target_ratio:.2%} >= {self.STAGE1_TARGET_RATIO:.2%}",
            )
            add(
                "max_aligned_steps",
                self.max_aligned_consecutive_steps >= self.STAGE1_REQUIRED_CONSEC,
                f"{self.max_aligned_consecutive_steps} >= {self.STAGE1_REQUIRED_CONSEC}",
            )
            add(
                "invalid_actions",
                no_invalid_actions,
                f"{self.invalid_actions} == 0",
            )
            add(
                "full_episode",
                full_episode,
                f"{steps} == {self.STAGE1_REQUIRED_STEPS}",
            )

            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(metric, 0.999)), criteria

        if self.stage_id == "stage2":
            enough_hits = self.shots_hit >= self.STAGE2_REQUIRED_HITS or self.damage_dealt >= 80.0 or self.kills > 0
            # No relaxed dirty-kill exception here: Stage 2 is specifically about
            # clean shooting timing, so a kill with poor hit rate must not pass.
            hit_rate_ok = hit_rate >= self.STAGE2_REQUIRED_HIT_RATE
            blocked_shot_rate = (self.shots_blocked / self.shots_fired) if self.shots_fired > 0 else 0.0
            blocked_rate_ok = blocked_shot_rate <= self.STAGE2_MAX_BLOCKED_SHOT_RATE
            bad_rate_ok = bad_rate <= self.STAGE2_MAX_BAD_SHOT_RATE
            aim_error_ok = mean_aim_error <= self.STAGE2_MAX_MEAN_AIM_ERROR
            target_time_ok = time_on_target_ratio >= self.STAGE2_MIN_TIME_ON_TARGET

            metric = min(
                self.shots_hit / self.STAGE2_REQUIRED_HITS,
                hit_rate / self.STAGE2_REQUIRED_HIT_RATE if self.shots_fired else 0.0,
                self.STAGE2_MAX_BAD_SHOT_RATE / max(bad_rate, 1e-6) if self.shots_fired else 0.0,
                self.STAGE2_MAX_BLOCKED_SHOT_RATE / max(blocked_shot_rate, 1e-6) if self.shots_fired else 0.0,
                self.STAGE2_MAX_MEAN_AIM_ERROR / max(mean_aim_error, 1e-6),
                time_on_target_ratio / self.STAGE2_MIN_TIME_ON_TARGET,
            )
            add("enough hits", enough_hits, f"hits={self.shots_hit}, damage={self.damage_dealt:.1f}")
            add("hit rate", hit_rate_ok, f"hit_rate={hit_rate:.2%} >= {self.STAGE2_REQUIRED_HIT_RATE:.2%}")
            add("bad-shot rate", bad_rate_ok, f"bad_rate={bad_rate:.2%} <= {self.STAGE2_MAX_BAD_SHOT_RATE:.2%}")
            add("blocked-shot rate", blocked_rate_ok, f"blocked_rate={blocked_shot_rate:.2%} <= {self.STAGE2_MAX_BLOCKED_SHOT_RATE:.2%}")
            add("aim retained", aim_error_ok, f"mean_aim_error={mean_aim_error:.3f} <= {self.STAGE2_MAX_MEAN_AIM_ERROR:.3f}")
            add("time on target", target_time_ok, f"time_on_target={time_on_target_ratio:.2%} >= {self.STAGE2_MIN_TIME_ON_TARGET:.2%}")
            all_criteria_ok = enough_hits and hit_rate_ok and bad_rate_ok and blocked_rate_ok and aim_error_ok and target_time_ok
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria

        if self.stage_id == "stage3":
            useful_range_ratio = self.useful_range_steps / steps
            usable_aim_ratio = self.stage3_usable_aim_steps / steps
            boundary_safe_ratio = self.boundary_safe_steps / steps
            jitter_rate = self.movement_jitter_count / steps
            nav_progress_ratio = self.nav_progress_positive_steps / max(1, self.nav_valid_steps)
            nav_direction_ratio = self.nav_direction_good_steps / max(1, self.nav_valid_steps)
            avg_path_deviation = self.nav_path_deviation_sum / max(1, self.nav_valid_steps)
            avg_desired_distance = self.nav_desired_distance_sum / max(1, self.nav_valid_steps)
            range_control_ratio = self.stage3_correct_range_control_steps / max(1, self.stage3_range_control_opportunities)
            stable_range_ratio = self.stage3_stable_range_steps / steps
            blocked_score = 1.0 if blocked_rate <= self.STAGE3_BLOCKED_RATE_LIMIT else max(
                0.0,
                1.0 - (blocked_rate - self.STAGE3_BLOCKED_RATE_LIMIT) / self.STAGE3_BLOCKED_RATE_LIMIT,
            )
            nav_quality = min(
                nav_progress_ratio / self.STAGE3_MIN_NAV_PROGRESS_RATIO,
                max(0.0, 1.0 - avg_path_deviation / max(1.0, self.STAGE3_MAX_AVG_PATH_DEVIATION * 1.4)),
                # A low average desired-distance can compensate for few explicit
                # progress steps when the target moves and the policy is already
                # maintaining the moving combat position.
                max(nav_direction_ratio / 0.35, max(0.0, 1.0 - avg_desired_distance / 150.0)),
            )
            smoothness_score = 1.0 if jitter_rate <= self.STAGE3_MAX_JITTER_RATE else max(
                0.0,
                1.0 - jitter_rate / max(1e-6, self.STAGE3_MAX_JITTER_RATE * 2.0),
            )
            metric = min(
                nav_quality,
                self.movement_distance_total / self.STAGE3_REQUIRED_MOVEMENT,
                useful_range_ratio / self.STAGE3_MIN_USEFUL_RANGE_RATIO,
                visible_ratio / self.STAGE3_MIN_VISIBLE_RATIO,
                usable_aim_ratio / self.STAGE3_MIN_USABLE_AIM_RATIO,
                self.STAGE3_MAX_MEAN_AIM_ERROR / max(mean_aim_error, 1e-6),
                self.max_aligned_consecutive_steps / self.STAGE3_MIN_STRICT_AIM_STREAK,
                range_control_ratio / self.STAGE3_MIN_RANGE_CONTROL_RATIO,
                blocked_score,
                boundary_safe_ratio / self.STAGE3_MIN_BOUNDARY_SAFE_RATIO,
                smoothness_score,
            )
            add(
                "navigation/path following",
                self.nav_valid_steps > 0 and nav_progress_ratio >= self.STAGE3_MIN_NAV_PROGRESS_RATIO and avg_path_deviation <= self.STAGE3_MAX_AVG_PATH_DEVIATION,
                f"nav_progress={nav_progress_ratio:.2%} >= {self.STAGE3_MIN_NAV_PROGRESS_RATIO:.2%}; avg_path_deviation={avg_path_deviation:.1f} <= {self.STAGE3_MAX_AVG_PATH_DEVIATION:.1f}; waypoints={self.nav_waypoint_reached_count}; recomputes={self.nav_recompute_count}",
            )
            add("movement activity", self.movement_distance_total >= self.STAGE3_REQUIRED_MOVEMENT, f"movement={self.movement_distance_total:.1f} >= {self.STAGE3_REQUIRED_MOVEMENT:.1f}")
            add("useful combat range", useful_range_ratio >= self.STAGE3_MIN_USEFUL_RANGE_RATIO, f"useful_range={useful_range_ratio:.2%} >= {self.STAGE3_MIN_USEFUL_RANGE_RATIO:.2%}; range_steps={self.useful_range_steps}; max_range_streak={self.max_range_streak_steps}; stable_range={stable_range_ratio:.2%}")
            add("enemy visibility", visible_ratio >= self.STAGE3_MIN_VISIBLE_RATIO, f"visible_ratio={visible_ratio:.2%} >= {self.STAGE3_MIN_VISIBLE_RATIO:.2%}")
            add("usable aim while moving", usable_aim_ratio >= self.STAGE3_MIN_USABLE_AIM_RATIO and mean_aim_error <= self.STAGE3_MAX_MEAN_AIM_ERROR and self.max_aligned_consecutive_steps >= self.STAGE3_MIN_STRICT_AIM_STREAK, f"usable_aim={usable_aim_ratio:.2%} >= {self.STAGE3_MIN_USABLE_AIM_RATIO:.2%}; mean_aim_error={mean_aim_error:.3f} <= {self.STAGE3_MAX_MEAN_AIM_ERROR:.3f}; max_aligned={self.max_aligned_consecutive_steps} >= {self.STAGE3_MIN_STRICT_AIM_STREAK}")
            add("closed-loop range control", range_control_ratio >= self.STAGE3_MIN_RANGE_CONTROL_RATIO and self.stage3_range_control_opportunities > 0, f"correct_range_control={range_control_ratio:.2%} >= {self.STAGE3_MIN_RANGE_CONTROL_RATIO:.2%}; opportunities={self.stage3_range_control_opportunities}; correct={self.stage3_correct_range_control_steps}; wrong={self.stage3_wrong_range_control_steps}; stable={self.stage3_stable_range_steps}")
            add("blocked movement rate", blocked_rate <= self.STAGE3_BLOCKED_RATE_LIMIT, f"blocked_rate={blocked_rate:.2%} <= {self.STAGE3_BLOCKED_RATE_LIMIT:.2%}")
            add("boundary safety", boundary_safe_ratio >= self.STAGE3_MIN_BOUNDARY_SAFE_RATIO, f"boundary_safe={boundary_safe_ratio:.2%} >= {self.STAGE3_MIN_BOUNDARY_SAFE_RATIO:.2%}; boundary_hugging_steps={self.boundary_hugging_steps}")
            add("movement smoothness", jitter_rate <= self.STAGE3_MAX_JITTER_RATE, f"jitter_rate={jitter_rate:.2%} <= {self.STAGE3_MAX_JITTER_RATE:.2%}; jitter_count={self.movement_jitter_count}")
            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria

        if self.stage_id == "stage4":
            stage4_max_steps = 460
            max_allowed_damage = 24.0
            min_survival_ratio = 0.85
            min_avoidance_rate = 0.55
            min_danger_steps = 8
            required_movement = 160.0
            min_boundary_safe_ratio = 0.85

            survival_full_ratio = min(1.0, steps / stage4_max_steps)
            boundary_safe_ratio = self.boundary_safe_steps / steps

            damage_ok = self.damage_taken <= max_allowed_damage
            if damage_ok:
                damage_score = 1.0
            else:
                damage_score = max(
                    0.0,
                    1.0 - (self.damage_taken - max_allowed_damage) / max(1.0, 100.0 - max_allowed_damage),
                )

            avoidance_ok = (
                self.projectile_danger_steps > min_danger_steps
                and avoidance_rate >= min_avoidance_rate
            )

            metric = min(
                survival_full_ratio / min_survival_ratio,
                damage_score,
                avoidance_rate / min_avoidance_rate if self.projectile_danger_steps > min_danger_steps else 0.0,
                self.movement_distance_total / required_movement,
                boundary_safe_ratio / min_boundary_safe_ratio,
            )

            add(
                "damage taken",
                damage_ok,
                f"damage_taken={self.damage_taken:.1f} <= {max_allowed_damage:.1f}",
            )
            add(
                "survival time",
                survival_full_ratio >= min_survival_ratio,
                f"survival_ratio={survival_full_ratio:.2%} >= {min_survival_ratio:.2%}; steps={steps}/{stage4_max_steps}",
            )
            add(
                "projectile avoidance",
                avoidance_ok,
                f"avoidance={avoidance_rate:.2%} >= {min_avoidance_rate:.2%}; danger_steps={self.projectile_danger_steps} > {min_danger_steps}",
            )
            add(
                "movement activity",
                self.movement_distance_total >= required_movement,
                f"movement={self.movement_distance_total:.1f} >= {required_movement:.1f}",
            )
            add(
                "boundary safety",
                boundary_safe_ratio >= min_boundary_safe_ratio,
                f"boundary_safe={boundary_safe_ratio:.2%} >= {min_boundary_safe_ratio:.2%}; boundary_hugging_steps={self.boundary_hugging_steps}",
            )

            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria
        
        if self.stage_id == "stage5":
            phase_index = max(1, min(3, int(getattr(self, "stage5_curriculum_phase_index", 3) or 3)))
            if phase_index <= 1:
                required_damage = 42.0
                required_moving_hits = 2
                required_hit_rate = 0.18
                required_moving_hit_rate = 0.18
                required_movement = 220.0
                max_allowed_damage = 105.0
                min_survival_ratio = 0.0
            elif phase_index == 2:
                required_damage = 63.0
                required_moving_hits = 2
                required_hit_rate = 0.22
                required_moving_hit_rate = 0.20
                required_movement = 230.0
                max_allowed_damage = 84.0
                min_survival_ratio = 0.60
            else:
                required_damage = 63.0
                required_moving_hits = 3
                required_hit_rate = 0.26
                required_moving_hit_rate = 0.24
                required_movement = 240.0
                max_allowed_damage = 84.0
                min_survival_ratio = 0.75

            moving_hit_rate = self.moving_hit_rate()
            moving_activity_rate = self.moving_shot_rate()
            offense_quality = max(
                min(self.damage_dealt / required_damage, 1.0),
                1.0 if self.kills > 0 else 0.0,
            )
            moving_hit_quality = max(
                min(self.moving_shots_hit / max(1.0, float(required_moving_hits)), 1.0),
                min(self.moving_damage_dealt / max(1.0, required_damage), 1.0),
            )
            hit_quality = hit_rate / required_hit_rate if self.shots_fired else 0.0
            moving_hit_rate_quality = moving_hit_rate / required_moving_hit_rate if self.moving_shots_fired else 0.0
            movement_quality = self.movement_distance_total / required_movement
            survival_quality = 1.0
            if min_survival_ratio > 0.0:
                survival_quality = min(
                    1.0 if self.deaths == 0 else 0.0,
                    max_allowed_damage / max(self.damage_taken, 1.0),
                    survival_ratio / min_survival_ratio,
                )
            reload_sanity = max(
                0.0,
                1.0
                - min(1.0, self.reload_failures / max(1.0, self.shots_fired + 5.0))
                - min(0.5, self.shot_unavailable_count / max(1.0, self.shots_fired + 12.0)),
            )

            metric = min(
                offense_quality,
                moving_hit_quality,
                hit_quality,
                moving_hit_rate_quality,
                movement_quality,
                survival_quality,
                reload_sanity,
            )
            phase_label = {1: "5A", 2: "5B", 3: "5C"}.get(phase_index, "5C")
            add("damage dealt", self.damage_dealt >= required_damage or self.kills > 0, f"phase={phase_label}; damage={self.damage_dealt:.1f} >= {required_damage:.1f}")
            add("moving hits", self.moving_shots_hit >= required_moving_hits, f"moving_hits={self.moving_shots_hit} >= {required_moving_hits}; moving_damage={self.moving_damage_dealt:.1f}")
            add("hit rate", hit_rate >= required_hit_rate, f"hit_rate={hit_rate:.2%} >= {required_hit_rate:.2%}")
            add("moving hit rate", moving_hit_rate >= required_moving_hit_rate, f"moving_hit_rate={moving_hit_rate:.2%} >= {required_moving_hit_rate:.2%}; activity={moving_activity_rate:.2%}")
            add("movement activity", self.movement_distance_total >= required_movement, f"movement={self.movement_distance_total:.1f} >= {required_movement:.1f}")
            if min_survival_ratio > 0.0:
                add("survival", self.deaths == 0 and self.damage_taken <= max_allowed_damage and survival_ratio >= min_survival_ratio, f"deaths={self.deaths}; damage_taken={self.damage_taken:.1f} <= {max_allowed_damage:.1f}; survival={survival_ratio:.2%} >= {min_survival_ratio:.2%}")
            add("reload sanity", reload_sanity >= 0.70, f"reload_sanity={reload_sanity:.2f}; unavailable={self.shot_unavailable_count}; reload_failures={self.reload_failures}")
            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria

        if self.stage_id == "stage6":
            phase_index = max(1, min(4, int(getattr(self, "stage6_curriculum_phase_index", 4) or 4)))
            if phase_index <= 1:
                required_damage = 84.0
                required_damage_ratio = 1.05
                required_hit_rate = 0.24
                max_allowed_damage = 84.0
                max_projectile_hits = 4
                max_passive_steps = 140
                required_movement = 220.0
                min_reload_sanity = 0.45
            elif phase_index == 2:
                required_damage = 100.0
                required_damage_ratio = 1.15
                required_hit_rate = 0.26
                max_allowed_damage = 84.0
                max_projectile_hits = 4
                max_passive_steps = 120
                required_movement = 230.0
                min_reload_sanity = 0.55
            elif phase_index == 3:
                required_damage = 100.0
                required_damage_ratio = 1.20
                required_hit_rate = 0.27
                max_allowed_damage = 84.0
                max_projectile_hits = 4
                max_passive_steps = 115
                required_movement = 230.0
                min_reload_sanity = 0.60
            else:
                required_damage = 100.0
                required_damage_ratio = 1.25
                required_hit_rate = 0.28
                max_allowed_damage = 84.0
                max_projectile_hits = 4
                max_passive_steps = 105
                # Final 6D should not fail a clean fast kill only because the
                # agent moved slightly below the old 230 px edge threshold.
                # Core gates still remain strict: kill, no death, damage ratio,
                # damage taken, projectile hits, hit rate, and reload sanity.
                required_movement = 210.0
                min_reload_sanity = 0.65

            phase_label = {1: "6A", 2: "6B", 3: "6C", 4: "6D"}.get(phase_index, "6D")
            win_ok = self.kills > 0 and self.deaths == 0
            damage_ok = self.damage_dealt >= required_damage or self.kills > 0
            ratio_ok = damage_ratio >= required_damage_ratio
            hit_ok = hit_rate >= required_hit_rate
            survival_ok = self.deaths == 0 and self.damage_taken <= max_allowed_damage
            projectile_ok = self.projectile_hits_taken <= max_projectile_hits
            movement_ok = self.movement_distance_total >= required_movement
            passive_ok = self.passive_steps < max_passive_steps
            reload_sanity = max(
                0.0,
                1.0
                - min(1.0, self.reload_failures / max(1.0, self.shots_fired + 5.0))
                - min(0.5, self.shot_unavailable_count / max(1.0, self.shots_fired + 10.0)),
            )
            reload_ok = reload_sanity >= min_reload_sanity

            offense_quality = max(
                1.0 if win_ok else 0.0,
                min(self.damage_dealt / max(1.0, required_damage), 1.0),
            )
            ratio_quality = min(damage_ratio / max(1e-6, required_damage_ratio), 1.0)
            hit_quality = hit_rate / required_hit_rate if self.shots_fired else 0.0
            survival_quality = min(
                1.0 if self.deaths == 0 else 0.0,
                max_allowed_damage / max(self.damage_taken, 1.0),
            )
            projectile_quality = max(0.0, 1.0 - max(0.0, self.projectile_hits_taken - max_projectile_hits) / max(1.0, max_projectile_hits))
            movement_quality = min(self.movement_distance_total / max(1.0, required_movement), 1.0)
            passive_quality = 1.0 if passive_ok else max(0.0, 1.0 - (self.passive_steps - max_passive_steps) / max(1.0, max_passive_steps))

            metric = min(
                offense_quality,
                ratio_quality,
                hit_quality,
                survival_quality,
                projectile_quality,
                movement_quality,
                passive_quality,
                reload_sanity / max(1e-6, min_reload_sanity),
            )
            add("win", win_ok, f"phase={phase_label}; kills={self.kills}, deaths={self.deaths}")
            add("damage dealt", damage_ok, f"damage={self.damage_dealt:.1f} >= {required_damage:.1f} or kill")
            add("damage ratio", ratio_ok, f"ratio={damage_ratio:.2f} >= {required_damage_ratio:.2f}")
            add("hit rate", hit_ok, f"hit_rate={hit_rate:.2%} >= {required_hit_rate:.2%}")
            add("survival", survival_ok, f"deaths={self.deaths}; damage_taken={self.damage_taken:.1f} <= {max_allowed_damage:.1f}")
            add("projectile hits", projectile_ok, f"projectile_hits_taken={self.projectile_hits_taken} <= {max_projectile_hits}")
            add("movement activity", movement_ok, f"movement={self.movement_distance_total:.1f} >= {required_movement:.1f}")
            add("not passive", passive_ok, f"passive_steps={self.passive_steps} < {max_passive_steps}")
            add("reload sanity", reload_ok, f"reload_sanity={reload_sanity:.2f} >= {min_reload_sanity:.2f}; unavailable={self.shot_unavailable_count}; reload_failures={self.reload_failures}")
            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria

        if self.stage_id == "stage7":
            phase_index = max(1, min(5, int(getattr(self, "stage7_curriculum_phase_index", 5) or 5)))
            if phase_index <= 1:
                max_blocked_rate = 0.105
                max_blocked_shots = 3
                required_damage_ratio = 0.85
                max_passive_steps = 130
                max_allowed_damage = 92.0
                required_movement = 240.0
                min_visible_ratio = 0.18
                min_clear_shot_ratio = 0.045
                min_time_on_target_ratio = 0.080
                min_nav_progress_ratio = 0.120
                max_avg_path_deviation = 86.0
            elif phase_index == 2:
                max_blocked_rate = 0.095
                max_blocked_shots = 3
                required_damage_ratio = 0.90
                max_passive_steps = 120
                max_allowed_damage = 94.0
                required_movement = 270.0
                min_visible_ratio = 0.16
                min_clear_shot_ratio = 0.040
                min_time_on_target_ratio = 0.070
                min_nav_progress_ratio = 0.110
                max_avg_path_deviation = 84.0
            elif phase_index == 3:
                max_blocked_rate = 0.085
                max_blocked_shots = 4
                required_damage_ratio = 0.95
                max_passive_steps = 115
                max_allowed_damage = 96.0
                required_movement = 300.0
                min_visible_ratio = 0.14
                min_clear_shot_ratio = 0.035
                min_time_on_target_ratio = 0.060
                min_nav_progress_ratio = 0.100
                max_avg_path_deviation = 82.0
            elif phase_index == 4:
                max_blocked_rate = 0.080
                max_blocked_shots = 5
                required_damage_ratio = 1.00
                max_passive_steps = 110
                max_allowed_damage = 96.0
                required_movement = 320.0
                min_visible_ratio = 0.12
                min_clear_shot_ratio = 0.030
                min_time_on_target_ratio = 0.055
                min_nav_progress_ratio = 0.085
                max_avg_path_deviation = 80.0
            else:
                max_blocked_rate = 0.070
                max_blocked_shots = 5
                required_damage_ratio = 1.00
                max_passive_steps = 105
                max_allowed_damage = 90.0
                required_movement = 340.0
                min_visible_ratio = 0.12
                min_clear_shot_ratio = 0.030
                min_time_on_target_ratio = 0.050
                min_nav_progress_ratio = 0.080
                max_avg_path_deviation = 78.0

            nav_progress_ratio = self.nav_progress_positive_steps / max(1, self.nav_valid_steps)
            avg_path_deviation = self.nav_path_deviation_sum / max(1, self.nav_valid_steps)
            clear_shot_ratio = self.enemy_clear_shot_ratio()
            phase_label = {1: "7A", 2: "7B", 3: "7C", 4: "7D", 5: "7E"}.get(phase_index, "7E")
            win_ok = self.kills > 0 and self.deaths == 0
            blocked_move_ok = blocked_rate <= max_blocked_rate
            blocked_shot_ok = self.shots_blocked <= max_blocked_shots
            ratio_ok = damage_ratio >= required_damage_ratio
            survival_ok = self.deaths == 0 and self.damage_taken <= max_allowed_damage
            passive_ok = self.passive_steps < max_passive_steps
            movement_ok = self.movement_distance_total >= required_movement
            visibility_ok = visible_ratio >= min_visible_ratio or self.damage_dealt > 0.0
            clear_shot_ok = clear_shot_ratio >= min_clear_shot_ratio or self.damage_dealt > 0.0
            aim_ok = time_on_target_ratio >= min_time_on_target_ratio
            nav_ok = self.nav_valid_steps > 0 and nav_progress_ratio >= min_nav_progress_ratio and avg_path_deviation <= max_avg_path_deviation

            offense_quality = max(1.0 if win_ok else 0.0, min(self.damage_dealt / 105.0, 1.0))
            blocked_move_quality = max(0.0, 1.0 - max(0.0, blocked_rate - max_blocked_rate) / max(max_blocked_rate, 1e-6))
            blocked_shot_quality = max(0.0, 1.0 - max(0.0, self.shots_blocked - max_blocked_shots) / max(1.0, max_blocked_shots))
            ratio_quality = min(damage_ratio / max(1e-6, required_damage_ratio), 1.0)
            survival_quality = min(1.0 if self.deaths == 0 else 0.0, max_allowed_damage / max(self.damage_taken, 1.0))
            passive_quality = 1.0 if passive_ok else max(0.0, 1.0 - (self.passive_steps - max_passive_steps) / max(1.0, max_passive_steps))
            movement_quality = min(self.movement_distance_total / max(1.0, required_movement), 1.0)
            visibility_quality = 1.0 if visibility_ok else min(visible_ratio / max(1e-6, min_visible_ratio), 1.0)
            clear_shot_quality = 1.0 if clear_shot_ok else min(clear_shot_ratio / max(1e-6, min_clear_shot_ratio), 1.0)
            aim_quality_metric = min(time_on_target_ratio / max(1e-6, min_time_on_target_ratio), 1.0)
            nav_progress_quality = min(nav_progress_ratio / max(1e-6, min_nav_progress_ratio), 1.0)
            path_deviation_quality = max(0.0, 1.0 - max(0.0, avg_path_deviation - max_avg_path_deviation) / max(1.0, max_avg_path_deviation))
            metric = min(
                offense_quality,
                blocked_move_quality,
                blocked_shot_quality,
                ratio_quality,
                survival_quality,
                passive_quality,
                movement_quality,
                visibility_quality,
                clear_shot_quality,
                aim_quality_metric,
                nav_progress_quality,
                path_deviation_quality,
            )

            add("win", win_ok, f"phase={phase_label}; kills={self.kills}, deaths={self.deaths}")
            add("blocked movement", blocked_move_ok, f"blocked_rate={blocked_rate:.2%} <= {max_blocked_rate:.2%}")
            add("blocked shots", blocked_shot_ok, f"blocked_shots={self.shots_blocked} <= {max_blocked_shots}")
            add("damage ratio", ratio_ok, f"ratio={damage_ratio:.2f} >= {required_damage_ratio:.2f}")
            add("survival", survival_ok, f"deaths={self.deaths}; damage_taken={self.damage_taken:.1f} <= {max_allowed_damage:.1f}")
            add("movement activity", movement_ok, f"movement={self.movement_distance_total:.1f} >= {required_movement:.1f}")
            add("path progress", nav_ok, f"nav_progress={nav_progress_ratio:.2%} >= {min_nav_progress_ratio:.2%}; avg_path_deviation={avg_path_deviation:.1f} <= {max_avg_path_deviation:.1f}; valid_steps={self.nav_valid_steps}")
            add("aim retained", aim_ok, f"time_on_target={time_on_target_ratio:.2%} >= {min_time_on_target_ratio:.2%}")
            add("line-of-sight activity", visibility_ok, f"visible_ratio={visible_ratio:.2%} >= {min_visible_ratio:.2%} or damage={self.damage_dealt:.1f}")
            add("clear-shot activity", clear_shot_ok, f"clear_shot_ratio={clear_shot_ratio:.2%} >= {min_clear_shot_ratio:.2%} or damage={self.damage_dealt:.1f}; clear_gained={self.clear_shot_gained_count}")
            add("not passive", passive_ok, f"passive_steps={self.passive_steps} < {max_passive_steps}")
            all_criteria_ok = all(v[0] for v in criteria.values())
            return (1.0 if all_criteria_ok else min(0.999, metric)), criteria

        if self.stage_id == "stage8":
            phase_index = max(1, min(5, int(getattr(self, "stage8_curriculum_phase_index", 5) or 5)))
            phase_label = {1: "8A", 2: "8B", 3: "8C", 4: "8D", 5: "8E"}.get(phase_index, "8E")
            clear_shot_ratio = self.enemy_clear_shot_ratio()
            reload_quality = self.stage8_reload_quality_sum / max(1, self.stage8_reload_started)

            if phase_index <= 1:
                first_shot_ok = bool(self.stage8_first_shot_rewarded)
                target_ok = bool(self.stage8_target_ammo_reached)
                good_reload_ok = self.stage8_good_reloads >= 1
                post_reload_ok = self.stage8_shots_after_reload >= 1
                no_early_ok = self.stage8_early_reloads == 0
                no_empty_ok = self.stage8_empty_reloads == 0
                no_past_target_ok = self.stage8_shot_past_target_before_reload == 0
                sequence_ok = (
                    first_shot_ok
                    and target_ok
                    and good_reload_ok
                    and post_reload_ok
                    and no_early_ok
                    and no_empty_ok
                    and no_past_target_ok
                )
                qualities = [
                    1.0 if first_shot_ok else 0.0,
                    1.0 if target_ok else 0.0,
                    1.0 if good_reload_ok else 0.0,
                    min(self.stage8_shots_after_reload / 1.0, 1.0),
                    1.0 if no_early_ok else 0.0,
                    1.0 if no_empty_ok else 0.0,
                    1.0 if no_past_target_ok else 0.0,
                ]
                metric = min(qualities)
                add("8A first shot", first_shot_ok, f"first_shot_rewarded={int(first_shot_ok)}")
                add("8A target ammo", target_ok, f"target_3_of_9_reached={int(target_ok)}; target_steps={self.stage8_target_ammo_steps}")
                add("8A good reload", good_reload_ok, f"good_reloads={self.stage8_good_reloads}; reloads={self.stage8_reload_started}; quality={reload_quality:.2f}")
                add("8A shot after reload", post_reload_ok, f"shots_after_reload={self.stage8_shots_after_reload} >= 1")
                add("8A no early reload", no_early_ok, f"early_reloads={self.stage8_early_reloads} == 0")
                add("8A no empty reload", no_empty_ok, f"empty_reloads={self.stage8_empty_reloads} == 0; empty_ammo_steps={self.stage8_empty_ammo_steps}")
                add("8A no shot past target", no_past_target_ok, f"shot_past_target_before_reload={self.stage8_shot_past_target_before_reload} == 0")
                return (1.0 if sequence_ok else min(0.999, metric)), criteria
            good_reload_ratio = self.stage8_good_reloads / max(1, self.stage8_reload_started)
            aim_mean = self.aim_error_mean()
            hit_rate = self.hit_rate()

            # 8A/8B are reload-learning phases.  They do not require a clean kill;
            # they require the agent to discover the unmasked reload timing and
            # return to shooting after reload.  Later phases progressively restore
            # full Stage 7 combat expectations.
            if phase_index <= 1:
                required_damage = 35.0
                required_ratio = 0.35
                min_hit_rate = 0.40
                max_blocked_allowed = 0.20
                win_ok = (self.damage_dealt >= required_damage or self.kills > 0) and self.deaths == 0
            elif phase_index == 2:
                required_damage = 55.0
                required_ratio = 0.60
                min_hit_rate = 0.50
                max_blocked_allowed = 0.16
                win_ok = (self.damage_dealt >= required_damage or self.kills > 0) and self.deaths == 0
            elif phase_index == 3:
                required_damage = 75.0
                required_ratio = 0.80
                min_hit_rate = 0.58
                max_blocked_allowed = 0.12
                win_ok = (self.damage_dealt >= required_damage or self.kills > 0) and self.deaths == 0
            elif phase_index == 4:
                required_damage = 90.0
                required_ratio = 0.95
                min_hit_rate = 0.65
                max_blocked_allowed = 0.09
                win_ok = (self.kills > 0 or self.damage_dealt >= required_damage) and self.deaths == 0
            else:
                required_damage = 100.0
                required_ratio = 1.10
                min_hit_rate = self.STAGE8_MIN_HIT_RATE
                max_blocked_allowed = self.STAGE8_MAX_BLOCKED_RATE
                win_ok = self.kills > 0 and self.deaths == 0

            blocked_ok = blocked_rate <= max_blocked_allowed
            ratio_ok = damage_ratio >= required_ratio or phase_index <= 2
            aim_error_ok = aim_mean <= self.STAGE8_MAX_MEAN_AIM_ERROR
            practical_aim_ok = (hit_rate >= self.STAGE8_MIN_HIT_RATE_FOR_AIM_OVERRIDE and self.stage8_unaligned_shot_attempts <= (3 if phase_index <= 2 else 0))
            aim_ok = ((aim_error_ok or practical_aim_ok) and time_on_target_ratio >= (0.08 if phase_index <= 2 else self.STAGE8_MIN_TIME_ON_TARGET)) or phase_index <= 1
            hit_ok = hit_rate >= min_hit_rate and self.shots_blocked <= (5 if phase_index <= 3 else 2)
            visibility_ok = visible_ratio >= (0.12 if phase_index <= 2 else self.STAGE8_MIN_VISIBLE_RATIO) or self.damage_dealt >= required_damage
            clear_ok = clear_shot_ratio >= (0.03 if phase_index <= 3 else self.STAGE8_MIN_CLEAR_SHOT_RATIO) or self.damage_dealt >= required_damage
            if phase_index <= 1:
                reload_ok = (
                    self.stage8_reload_started >= 1
                    and self.stage8_good_reloads >= 1
                    and self.stage8_target_ammo_reached
                    and self.stage8_shots_after_reload >= 1
                    and reload_quality >= 0.65
                    and self.stage8_early_reloads == 0
                    and self.stage8_empty_reloads == 0
                    and self.stage8_shot_past_target_before_reload == 0
                    and self.shot_unavailable_count <= 4
                    and self.reload_failures <= 1
                )
            else:
                reload_ok = (
                    self.stage8_reload_started >= 1
                    and (self.stage8_good_reloads >= 1 or (phase_index <= 2 and self.stage8_acceptable_reloads >= 1))
                    and self.stage8_shots_after_reload >= 1
                    and reload_quality >= (0.45 if phase_index <= 2 else 0.55)
                    and self.stage8_early_reloads <= (2 if phase_index <= 2 else 1)
                    and self.stage8_empty_reloads <= (1 if phase_index <= 2 else 0)
                    and self.stage8_reload_while_killable <= (1 if phase_index <= 3 else 0)
                    and self.shot_unavailable_count <= (6 if phase_index <= 3 else self.STAGE8_MAX_UNAVAILABLE_SHOTS)
                    and self.reload_failures <= (2 if phase_index <= 2 else self.STAGE8_MAX_RELOAD_FAILURES)
                )
            episode_steps = max(1, self.survival_steps)
            stage8_turn_flip_rate = self.stage8_turn_flip_count / episode_steps
            stage8_movement_flip_rate = self.stage8_movement_flip_count / episode_steps
            smooth_ok = (
                self.stage8_blocked_spin_steps <= self.STAGE8_MAX_BLOCKED_SPIN_STEPS
                and stage8_movement_flip_rate <= self.STAGE8_MAX_MOVEMENT_FLIP_RATE
                and stage8_turn_flip_rate <= self.STAGE8_MAX_TURN_FLIP_RATE
            )

            qualities = [
                1.0 if win_ok else min(self.damage_dealt / 105.0, 0.95),
                max(0.0, 1.0 - max(0.0, blocked_rate - max_blocked_allowed) / max(max_blocked_allowed, 1e-6)),
                min(damage_ratio / max(required_ratio, 1e-6), 1.0),
                min(self.STAGE8_MAX_MEAN_AIM_ERROR / max(aim_mean, 1e-6), 1.0),
                min(time_on_target_ratio / max(self.STAGE8_MIN_TIME_ON_TARGET, 1e-6), 1.0),
                min(hit_rate / max(min_hit_rate, 1e-6), 1.0),
                min(clear_shot_ratio / max(self.STAGE8_MIN_CLEAR_SHOT_RATIO, 1e-6), 1.0),
                reload_quality,
                max(0.0, 1.0 - self.stage8_blocked_spin_steps / max(1.0, float(self.STAGE8_MAX_BLOCKED_SPIN_STEPS))),
                max(0.0, 1.0 - stage8_turn_flip_rate / max(self.STAGE8_MAX_TURN_FLIP_RATE, 1e-6)),
                max(0.0, 1.0 - stage8_movement_flip_rate / max(self.STAGE8_MAX_MOVEMENT_FLIP_RATE, 1e-6)),
            ]
            metric = min(qualities)
            all_ok = win_ok and blocked_ok and ratio_ok and aim_ok and hit_ok and visibility_ok and clear_ok and reload_ok and smooth_ok
            add("win", win_ok, f"phase={phase_label}; kills={self.kills}, deaths={self.deaths}, damage={self.damage_dealt:.1f} >= {required_damage:.1f}")
            add("blocked movement", blocked_ok, f"phase={phase_label}; blocked_rate={blocked_rate:.2%} <= {max_blocked_allowed:.2%}")
            add("damage ratio", ratio_ok, f"phase={phase_label}; ratio={damage_ratio:.2f} >= {required_ratio:.2f}")
            add("strict aim", aim_ok, f"aim_mean={aim_mean:.3f} <= {self.STAGE8_MAX_MEAN_AIM_ERROR:.3f} OR hit_rate={hit_rate:.2%} >= {self.STAGE8_MIN_HIT_RATE_FOR_AIM_OVERRIDE:.2%} with unaligned={self.stage8_unaligned_shot_attempts}; time_on_target={time_on_target_ratio:.2%} >= {self.STAGE8_MIN_TIME_ON_TARGET:.2%}")
            add("clean shooting", hit_ok, f"phase={phase_label}; hit_rate={hit_rate:.2%} >= {min_hit_rate:.2%}; blocked_shots={self.shots_blocked}")
            add("line of sight", visibility_ok, f"visible_ratio={visible_ratio:.2%} >= {self.STAGE8_MIN_VISIBLE_RATIO:.2%} or damage={self.damage_dealt:.1f}")
            add("clear shot", clear_ok, f"clear_shot_ratio={clear_shot_ratio:.2%} >= {self.STAGE8_MIN_CLEAR_SHOT_RATIO:.2%} or damage={self.damage_dealt:.1f}")
            add("tactical reload", reload_ok, f"reloads={self.stage8_reload_started}; good={self.stage8_good_reloads}; acceptable={self.stage8_acceptable_reloads}; shots_after_reload={self.stage8_shots_after_reload}; early={self.stage8_early_reloads}; late={self.stage8_late_reloads}; empty={self.stage8_empty_reloads}; quality={reload_quality:.2f}; good_ratio={good_reload_ratio:.2f}; unavailable={self.shot_unavailable_count}; failures={self.reload_failures}")
            add("smooth control", smooth_ok, f"blocked_spin={self.stage8_blocked_spin_steps} <= {self.STAGE8_MAX_BLOCKED_SPIN_STEPS}; movement_flip_rate={stage8_movement_flip_rate:.2%} <= {self.STAGE8_MAX_MOVEMENT_FLIP_RATE:.2%} ({self.stage8_movement_flip_count}/{episode_steps}); turn_flip_rate={stage8_turn_flip_rate:.2%} <= {self.STAGE8_MAX_TURN_FLIP_RATE:.2%} ({self.stage8_turn_flip_count}/{episode_steps})")
            return (1.0 if all_ok else min(0.999, metric)), criteria

        return 0.0, criteria

    def stage_promotion_reason(self) -> str:
        metric = self.stage_success_metric()
        stage_label = "Stage 1" if self.stage_id == "stage1" else self.stage_id
        if metric >= 1.0:
            return f"{stage_label} passed: " + "; ".join(
                f"{name} passed ({detail})" for name, (ok, detail) in self.last_criteria.items() if ok
            )
        parts = []
        for name, (ok, detail) in self.last_criteria.items():
            status = "passed" if ok else "failed"
            parts.append(f"{name} {status}: {detail}")
        return f"{stage_label} not passed: " + "; ".join(parts) if parts else f"{stage_label} not passed: no criteria available"

    def get_episode_stats(self, steps: int) -> dict[str, Any]:
        steps_safe = max(1, steps)
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "shots_missed": self.shots_missed,
            "shots_blocked": self.shots_blocked,
            "hit_rate": self.hit_rate(),
            "miss_rate": self.miss_rate(),
            "valid_shots": self.valid_shots,
            "bad_shots": self.bad_shots,
            "bad_shot_rate": self.bad_shot_rate(),
            "moving_shots_fired": self.moving_shots_fired,
            "moving_shots_hit": self.moving_shots_hit,
            "moving_shot_rate": self.moving_shot_rate(),
            "moving_hit_rate": self.moving_hit_rate(),
            "moving_damage_dealt": self.moving_damage_dealt,
            "stage5_curriculum_level": self.stage5_curriculum_level,
            "stage5_curriculum_total_levels": self.stage5_curriculum_total_levels,
            "stage5_curriculum_phase_index": self.stage5_curriculum_phase_index,
            "stage6_curriculum_level": self.stage6_curriculum_level,
            "stage6_curriculum_total_levels": self.stage6_curriculum_total_levels,
            "stage6_curriculum_phase_index": self.stage6_curriculum_phase_index,
            "stage7_curriculum_level": self.stage7_curriculum_level,
            "stage7_curriculum_total_levels": self.stage7_curriculum_total_levels,
            "stage7_curriculum_phase_index": self.stage7_curriculum_phase_index,
            "stage8_curriculum_level": self.stage8_curriculum_level,
            "stage8_curriculum_total_levels": self.stage8_curriculum_total_levels,
            "stage8_curriculum_phase_index": self.stage8_curriculum_phase_index,
            "invalid_actions": self.invalid_actions,
            "shot_unavailable_count": self.shot_unavailable_count,
            "reload_failures": self.reload_failures,
            "blocked_moves": self.blocked_moves,
            "blocked_move_rate": self.blocked_moves / steps_safe,
            "max_consecutive_blocked_moves": self.max_consecutive_blocked_moves,
            "line_of_sight_gained_count": self.line_of_sight_gained_count,
            "line_of_sight_lost_count": self.line_of_sight_lost_count,
            "clear_shot_gained_count": self.clear_shot_gained_count,
            "clear_shot_lost_count": self.clear_shot_lost_count,
            "steps_with_enemy_clear_shot": self.steps_with_enemy_clear_shot,
            "enemy_clear_shot_ratio": self.enemy_clear_shot_ratio(steps_safe),
            "stage7_unclean_shot_attempts": self.stage7_unclean_shot_attempts,
            "enemy_visible_events": self.enemy_visible_events,
            "enemy_lost_events": self.enemy_lost_events,
            "steps_with_enemy_visible": self.steps_with_enemy_visible,
            "enemy_visible_ratio": self.enemy_visible_ratio(steps_safe),
            "pickups_collected": self.pickups_collected,
            "useful_pickups": self.useful_pickups,
            "useless_pickups": self.useless_pickups,
            "stage8_reload_started": self.stage8_reload_started,
            "stage8_good_reloads": self.stage8_good_reloads,
            "stage8_acceptable_reloads": self.stage8_acceptable_reloads,
            "stage8_early_reloads": self.stage8_early_reloads,
            "stage8_late_reloads": self.stage8_late_reloads,
            "stage8_empty_reloads": self.stage8_empty_reloads,
            "stage8_reload_while_killable": self.stage8_reload_while_killable,
            "stage8_reload_quality": self.stage8_reload_quality_sum / max(1, self.stage8_reload_started),
            "stage8_target_ammo_reached": 1 if self.stage8_target_ammo_reached else 0,
            "stage8_target_ammo_steps": self.stage8_target_ammo_steps,
            "stage8_shot_past_target_before_reload": self.stage8_shot_past_target_before_reload,
            "stage8_first_shot_rewarded": 1 if self.stage8_first_shot_rewarded else 0,
            "stage8_no_first_shot_pressure_steps": self.stage8_no_first_shot_pressure_steps,
            "stage8_target_no_reload_timeout_steps": self.stage8_target_no_reload_timeout_steps,
            "stage8_8a_sequence_success": 1 if self.stage8_8a_sequence_success() else 0,
            "stage8_low_ammo_steps": self.stage8_low_ammo_steps,
            "stage8_critical_ammo_steps": self.stage8_critical_ammo_steps,
            "stage8_empty_ammo_steps": self.stage8_empty_ammo_steps,
            "stage8_clean_aligned_shots": self.stage8_clean_aligned_shots,
            "stage8_shots_after_reload": self.stage8_shots_after_reload,
            "stage8_unaligned_shot_attempts": self.stage8_unaligned_shot_attempts,
            "stage8_no_los_shot_attempts": self.stage8_no_los_shot_attempts,
            "stage8_aim_hold_steps": self.stage8_aim_hold_steps,
            "stage8_aim_improve_steps": self.stage8_aim_improve_steps,
            "stage8_aim_worse_steps": self.stage8_aim_worse_steps,
            "stage8_large_turn_count": self.stage8_large_turn_count,
            "stage8_turn_flip_count": self.stage8_turn_flip_count,
            "stage8_turn_flip_rate": self.stage8_turn_flip_count / steps_safe,
            "stage8_blocked_spin_steps": self.stage8_blocked_spin_steps,
            "stage8_movement_flip_count": self.stage8_movement_flip_count,
            "stage8_movement_flip_rate": self.stage8_movement_flip_count / steps_safe,
            "stage8_smooth_move_steps": self.stage8_smooth_move_steps,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "damage_dealt_taken_ratio": self.damage_dealt_taken_ratio(),
            "hits_taken": self.hits_taken,
            "projectile_danger_steps": self.projectile_danger_steps,
            "projectile_dodges": self.projectile_dodges,
            "projectile_hits_taken": self.projectile_hits_taken,
            "projectile_avoidance_rate": self.projectile_avoidance_rate(),
            "aim_error_mean": self.aim_error_mean(),
            "aim_error_min": self.aim_error_min if self.aim_error_count else None,
            "aim_error_final": self.aim_error_final if self.aim_error_count else None,
            "time_on_target_ratio": self.time_on_target_ratio(steps_safe),
            "max_aligned_steps": self.max_aligned_consecutive_steps,
            "heading_jitter_count": self.heading_jitter_count,
            "heading_delta_total": self.heading_delta_total,
            "good_tracking_steps": self.good_tracking_steps,
            "useful_range_steps": self.useful_range_steps,
            "range_streak_steps": self.range_streak_steps,
            "max_range_streak_steps": self.max_range_streak_steps,
            "stage3_usable_aim_steps": self.stage3_usable_aim_steps,
            "stage3_usable_aim_ratio": self.stage3_usable_aim_steps / steps_safe,
            "stage3_range_control_opportunities": self.stage3_range_control_opportunities,
            "stage3_correct_range_control_steps": self.stage3_correct_range_control_steps,
            "stage3_wrong_range_control_steps": self.stage3_wrong_range_control_steps,
            "stage3_range_control_ratio": self.stage3_correct_range_control_steps / max(1, self.stage3_range_control_opportunities),
            "stage3_stable_range_steps": self.stage3_stable_range_steps,
            "stage3_stable_range_ratio": self.stage3_stable_range_steps / steps_safe,
            "stage3_avg_range_drift": self.stage3_range_drift_sum / steps_safe,
            "boundary_safe_steps": self.boundary_safe_steps,
            "boundary_safe_ratio": self.boundary_safe_steps / steps_safe,
            "boundary_hugging_steps": self.boundary_hugging_steps,
            "nav_valid_steps": self.nav_valid_steps,
            "nav_progress_positive_steps": self.nav_progress_positive_steps,
            "nav_progress_ratio": self.nav_progress_positive_steps / max(1, self.nav_valid_steps),
            "nav_regress_steps": self.nav_regress_steps,
            "nav_waypoint_reached_count": self.nav_waypoint_reached_count,
            "nav_avg_path_deviation": self.nav_path_deviation_sum / max(1, self.nav_valid_steps),
            "nav_path_deviation_bad_steps": self.nav_path_deviation_bad_steps,
            "nav_avg_desired_distance": self.nav_desired_distance_sum / max(1, self.nav_valid_steps),
            "nav_path_progress_sum": self.nav_path_progress_sum,
            "nav_desired_progress_sum": self.nav_desired_progress_sum,
            "nav_direction_good_steps": self.nav_direction_good_steps,
            "nav_direction_good_ratio": self.nav_direction_good_steps / max(1, self.nav_valid_steps),
            "nav_recompute_count": self.nav_recompute_count,
            "movement_jitter_count": self.movement_jitter_count,
            "movement_jitter_streak": self.movement_jitter_streak,
            "movement_distance_total": self.movement_distance_total,
            "stationary_steps": self.stationary_steps,
            "passive_steps": self.passive_steps,
            "survival_steps": self.survival_steps,
            "survival_ratio": self.survival_steps / steps_safe,
            "total_reward": self.total_reward,
            "steps": steps,
            "success": self.episode_success(),
            "stage_success_metric": self.stage_success_metric(),
            "stage_promotion_reason": self.stage_promotion_reason(),
            "criteria_report": self.last_criteria,
        }

    def hit_rate(self) -> float:
        return (self.shots_hit / self.shots_fired) if self.shots_fired > 0 else 0.0

    def miss_rate(self) -> float:
        return (self.shots_missed / self.shots_fired) if self.shots_fired > 0 else 0.0

    def bad_shot_rate(self) -> float:
        return (self.bad_shots / self.shots_fired) if self.shots_fired > 0 else 0.0

    def moving_shot_rate(self) -> float:
        return (self.moving_shots_fired / self.shots_fired) if self.shots_fired > 0 else 0.0

    def moving_hit_rate(self) -> float:
        return (self.moving_shots_hit / self.moving_shots_fired) if self.moving_shots_fired > 0 else 0.0

    def set_stage5_curriculum_phase(self, level: int, total_levels: int, phase_index: int) -> None:
        self.stage5_curriculum_level = int(max(0, level))
        self.stage5_curriculum_total_levels = int(max(0, total_levels))
        self.stage5_curriculum_phase_index = int(max(1, min(3, phase_index)))

    def set_stage6_curriculum_phase(self, level: int, total_levels: int, phase_index: int) -> None:
        self.stage6_curriculum_level = int(max(0, level))
        self.stage6_curriculum_total_levels = int(max(0, total_levels))
        self.stage6_curriculum_phase_index = int(max(1, min(4, phase_index)))

    def set_stage7_curriculum_phase(self, level: int, total_levels: int, phase_index: int) -> None:
        self.stage7_curriculum_level = int(max(0, level))
        self.stage7_curriculum_total_levels = int(max(0, total_levels))
        self.stage7_curriculum_phase_index = int(max(1, min(5, phase_index)))

    def set_stage8_curriculum_phase(self, level: int, total_levels: int, phase_index: int) -> None:
        self.stage8_curriculum_level = int(max(0, level))
        self.stage8_curriculum_total_levels = int(max(0, total_levels))
        self.stage8_curriculum_phase_index = int(max(1, min(5, phase_index)))

    def enemy_visible_ratio(self, steps: int | None = None) -> float:
        denominator = max(1, steps or self.aim_error_count)
        return self.steps_with_enemy_visible / denominator

    def enemy_clear_shot_ratio(self, steps: int | None = None) -> float:
        denominator = max(1, steps or self.aim_error_count)
        return self.steps_with_enemy_clear_shot / denominator

    def time_on_target_ratio(self, steps: int | None = None) -> float:
        denominator = max(1, steps or self.aim_error_count)
        return self.time_on_target_steps / denominator

    def aim_error_mean(self) -> float:
        return self.aim_error_sum / self.aim_error_count if self.aim_error_count > 0 else math.pi

    def damage_dealt_taken_ratio(self) -> float:
        return self.damage_dealt / max(1.0, self.damage_taken)

    def projectile_avoidance_rate(self) -> float:
        attempts = self.projectile_dodges + self.projectile_hits_taken
        if attempts <= 0:
            return 0.0
        return self.projectile_dodges / attempts

    def _angle_error_to_enemy(self, learner: Agent, enemy: Agent) -> float:
        dx = enemy.position.x - learner.position.x
        dy = enemy.position.y - learner.position.y
        desired_heading = math.atan2(dy, dx)
        difference = desired_heading - learner.heading
        while difference > math.pi:
            difference -= 2 * math.pi
        while difference < -math.pi:
            difference += 2 * math.pi
        return abs(difference)

# Explicit public export used by tools/sanity_check.py and rl/env.py.
__all__ = ["RewardManager"]
