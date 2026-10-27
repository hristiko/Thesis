from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from config.stages import get_stage, normalize_stage_id
from game import detection
from game.agent import Agent, AgentCommand, wrap_angle
from game.world import Vec2, World
from rl.action_space import action_to_command, stage_action_mask
from rl.obs_builder import ObsBuilder
from training.skill_table import build_skill_table


class StockBot:
    

    _TEST_OFFSETS_DEG = (0, 18, -18, 35, -35, 55, -55, 75, -75, 95, -95, 120, -120, 150, -150, 180)
    _BASIC_TEST_OFFSETS_DEG = (0, 18, -18, 35, -35, 55, -55, 75, -75, 95, -95, 140, -140)
    _ZERO_VECTOR = Vec2(0.0, 0.0)

    _JITTER_CHANCE_BY_PROFILE = {
        "orbit_no_shoot": 0.0,
        "slow_shooter": 0.0,
        "passive_mover": 0.01,
        "easy_duel": 0.015,
        "moderate_duel": 0.02,
        "standard": 0.02,
    }

    def __init__(self, agent: Agent, world: World, target_id: str = "learner", profile: str = "standard", rng: Any | None = None):
        self.agent = agent
        self.world = world
        self.target_id = target_id
        self.profile = profile
        self.rng = rng or random

        self.preferred_range = {
            "orbit_no_shoot": 280.0,
            "passive_mover": 245.0,
            "slow_shooter": 300.0,
            "easy_duel": 270.0,
            "moderate_duel": 250.0,
            "standard": 240.0,
        }.get(profile, 240.0)

        self._combat_strafe_sign = self._choice_sign()
        self._combat_strafe_timer = self._uniform(0.3, 1.0)
        self._nav_sign = self._choice_sign()
        self._nav_commit_timer = self._uniform(0.3, 1.0)
        self._reaction_skip = 0

        self._patrol_points = [
            Vec2(220, 180),
            Vec2(world.width - 220, 180),
            Vec2(world.width - 220, world.height - 180),
            Vec2(220, world.height - 180),
        ]
        self._patrol_index = 0

        self._last_position = self.agent.position.copy()
        self._blocked_streak = 0
        self._stuck_frames = 0
        self._recovery_timer = 0.0
        self._recovery_angle = 0.0

    def _uniform(self, low: float, high: float) -> float:
        if hasattr(self.rng, "uniform"):
            return float(self.rng.uniform(low, high))
        return float(random.uniform(low, high))

    def _random(self) -> float:
        if hasattr(self.rng, "random"):
            return float(self.rng.random())
        return float(random.random())

    def _choice_sign(self) -> int:
        return -1 if self._random() < 0.5 else 1

    def reset_memory(self) -> None:
        self._last_position = self.agent.position.copy()
        self._blocked_streak = 0
        self._stuck_frames = 0
        self._recovery_timer = 0.0

    def tick(self) -> AgentCommand:
        target = self.world.agents.get(self.target_id)
        if target is None or not target.is_alive or not self.agent.is_alive:
            return AgentCommand()

        self._update_timers()
        self._update_stuck_state(target)
        recovery = self._recovery_command(target)
        if recovery is not None:
            return recovery

        if self.profile == "orbit_no_shoot":
            return self._tick_orbit_no_shoot(target)
        if self.profile == "passive_mover":
            return self._tick_passive_mover(target)
        if self.profile == "slow_shooter":
            return self._tick_slow_shooter(target)
        if self.profile == "easy_duel":
            return self._tick_duel(target, accuracy=0.09, reaction_mod=4, conservative_reload=False)
        if self.profile == "moderate_duel":
            return self._tick_duel(target, accuracy=0.10, reaction_mod=3, conservative_reload=False)
        return self._tick_duel(target, accuracy=0.12, reaction_mod=2, conservative_reload=True)



    def _tick_orbit_no_shoot(self, target: Agent) -> AgentCommand:
        to_target, distance, angle_to_target, angle_diff, line_of_sight = self._target_geometry(target)
        rotate = self._choose_rotation(angle_diff, dead_zone=0.10)

        intent = self._orbit_movement_intent(distance, angle_to_target)
        direction = self._drive_movement_intent(intent, target=target, current_line_of_sight=line_of_sight)
        move_idx, strafe_idx = self._world_direction_to_relative_command(direction)
        return AgentCommand(move_idx=move_idx, strafe_idx=strafe_idx, rot_idx=rotate)

    def _tick_passive_mover(self, target: Agent) -> AgentCommand:
        _, _, angle_to_target, angle_diff, _ = self._target_geometry(target)
        rotate = self._choose_rotation(angle_diff, dead_zone=0.12)

        intent = self._passive_movement_intent()
        direction = self._drive_movement_intent(intent)
        move_idx, strafe_idx = self._world_direction_to_relative_command(direction)
        return AgentCommand(move_idx=move_idx, strafe_idx=strafe_idx, rot_idx=rotate)

    def _tick_slow_shooter(self, target: Agent) -> AgentCommand:
        _, distance, _, angle_diff, line_of_sight = self._target_geometry(target)

        
        stage4_mode = getattr(self.world, "stage_id", "") == "stage4"
        max_shoot_distance = 950.0 if stage4_mode else 520.0
        aim_dead_zone = 0.07 if stage4_mode else 0.06
        shoot_tolerance = 0.12 if stage4_mode else 0.08
        first_shot_delay_steps = int(getattr(self.world, "stage4_first_shot_delay_steps", 0)) if stage4_mode else 0
        max_active_projectiles = int(getattr(self.world, "stage4_max_active_enemy_projectiles", 99)) if stage4_mode else 99
        active_projectiles = sum(
            1
            for projectile in getattr(self.world, "projectiles", [])
            if getattr(projectile, "owner_id", None) == self.agent.agent_id
        )

        rotate = self._choose_rotation(angle_diff, dead_zone=aim_dead_zone)
        shoot = (
            line_of_sight
            and abs(angle_diff) < shoot_tolerance
            and distance <= max_shoot_distance
            and getattr(self.world, "step_count", 0) >= first_shot_delay_steps
            and active_projectiles < max_active_projectiles
            and self.agent.ammo > 0
            and not self.agent.reloading
        )
        reload_allowed = (not stage4_mode) or bool(getattr(self.world, "stage4_enemy_reload_enabled", True))
        reload_now = reload_allowed and self.agent.ammo == 0 and not self.agent.reloading
        return AgentCommand(rot_idx=rotate, shoot=shoot, reload=reload_now)

    def _tick_duel(self, target: Agent, accuracy: float, reaction_mod: int, conservative_reload: bool) -> AgentCommand:
        _, distance, angle_to_target, angle_diff, line_of_sight = self._target_geometry(target)
        rotate = self._choose_rotation(angle_diff, dead_zone=max(0.07, accuracy * 0.75))
        reload_now = self._should_reload(line_of_sight, conservative=conservative_reload)

        intent = self._duel_movement_intent(distance, angle_to_target, line_of_sight)
        direction = self._drive_movement_intent(intent, target=target, current_line_of_sight=line_of_sight)

        
        if direction.length() <= 1e-9 and not line_of_sight:
            direction = self._drive_movement_intent(self._angle_to_unit(angle_to_target), target=target, current_line_of_sight=line_of_sight)
        if direction.length() <= 1e-9 and not line_of_sight:
            side_angle = angle_to_target + (self._nav_sign * math.pi / 2.0)
            direction = self._drive_movement_intent(self._angle_to_unit(side_angle), target=target, current_line_of_sight=line_of_sight)

        move_idx, strafe_idx = self._world_direction_to_relative_command(direction)

        shoot = False
        self._reaction_skip = (self._reaction_skip + 1) % max(1, reaction_mod)
        if (
            self._reaction_skip == 0
            and line_of_sight
            and abs(angle_diff) < accuracy
            and distance <= 380.0
            and self.agent.ammo > 0
            and not self.agent.reloading
        ):
            shoot = True

        return AgentCommand(move_idx=move_idx, strafe_idx=strafe_idx, rot_idx=rotate, shoot=shoot, reload=reload_now)

    

    def _orbit_movement_intent(self, distance: float, angle_to_target: float) -> Vec2:
        if distance > self.preferred_range + 80.0:
            return self._angle_to_unit(angle_to_target)
        if distance < self.preferred_range - 80.0:
            return self._angle_to_unit(angle_to_target + math.pi)
        return self._angle_to_unit(angle_to_target + (self._combat_strafe_sign * math.pi / 2.0))

    def _passive_movement_intent(self) -> Vec2:
        current_waypoint = self._patrol_points[self._patrol_index]
        if self.agent.position.distance_to(current_waypoint) < 28.0:
            self._patrol_index = (self._patrol_index + 1) % len(self._patrol_points)
            current_waypoint = self._patrol_points[self._patrol_index]

        desired_angle = math.atan2(current_waypoint.y - self.agent.position.y, current_waypoint.x - self.agent.position.x)
        return self._angle_to_unit(desired_angle)

    def _duel_movement_intent(self, distance: float, angle_to_target: float, line_of_sight: bool) -> Vec2:
        if self.agent.reloading:
            if distance < self.preferred_range + 20.0:
                return self._angle_to_unit(angle_to_target + math.pi)
            return self._angle_to_unit(angle_to_target + (self._combat_strafe_sign * math.pi / 2.0))

        if line_of_sight:
            if distance > self.preferred_range + 55.0:
                return self._angle_to_unit(angle_to_target)
            if distance < self.preferred_range - 60.0:
                return self._angle_to_unit(angle_to_target + math.pi)
            return self._angle_to_unit(angle_to_target + (self._combat_strafe_sign * math.pi / 2.0))

        return self._angle_to_unit(angle_to_target + (self._nav_sign * math.radians(65.0)))

    

    def _drive_movement_intent(
        self,
        intent: Vec2,
        target: Agent | None = None,
        current_line_of_sight: bool | None = None,
        max_probe_distance: float = 64.0,
    ) -> Vec2:
        intent = intent.normalize()
        if intent.length() <= 1e-9 or not self.agent.can_move:
            return Vec2(0.0, 0.0)

        candidates = self._rank_driver_directions(intent, target, current_line_of_sight, max_probe_distance)
        if not candidates:
            return Vec2(0.0, 0.0)

        selected = self._select_driver_candidate(candidates, max_probe_distance)
        return selected["direction"]

    def _rank_driver_directions(
        self,
        intent: Vec2,
        target: Agent | None,
        current_line_of_sight: bool | None,
        max_probe_distance: float,
    ) -> list[dict[str, Any]]:
        intent_angle = math.atan2(intent.y, intent.x)
        candidates: list[dict[str, Any]] = []

        for offset_deg in self._TEST_OFFSETS_DEG:
            angle = intent_angle + math.radians(offset_deg)
            direction = self._angle_to_unit(angle)
            clearance = self._direction_clearance(direction, max_probe_distance=max_probe_distance)
            if clearance <= 0.0:
                continue

            score = self._dot(direction, intent)
            score += clearance / 70.0

            test_position = self.agent.position + (direction * clearance)
            line_of_sight_after = False
            if target is not None:
                line_of_sight_after = self.world.has_line_of_sight(test_position, target.position)
                if line_of_sight_after:
                    score += 0.8
                elif current_line_of_sight:
                    score -= 0.8
                if current_line_of_sight is False and line_of_sight_after:
                    score += 1.8

            if offset_deg * self._nav_sign > 0:
                score += 0.08

            candidates.append(
                {
                    "score": score,
                    "direction": direction,
                    "clearance": clearance,
                    "fully_safe": clearance >= max_probe_distance - 1e-6,
                    "line_of_sight_after": line_of_sight_after,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def _select_driver_candidate(self, candidates: list[dict[str, Any]], max_probe_distance: float) -> dict[str, Any]:
        best = candidates[0]
        jitter_chance = max(0.0, min(0.05, self._JITTER_CHANCE_BY_PROFILE.get(self.profile, 0.0)))
        if jitter_chance <= 0.0 or self._random() >= jitter_chance:
            return best

        
        alternatives = [
            candidate
            for candidate in candidates[1:3]
            if candidate["fully_safe"] and candidate["score"] >= best["score"] - 0.65
        ]
        if not alternatives:
            return best
        return alternatives[0] if len(alternatives) == 1 or self._random() < 0.5 else alternatives[1]

    

    def _target_geometry(self, target: Agent) -> tuple[Vec2, float, float, float, bool]:
        to_target = target.position - self.agent.position
        distance = to_target.length()
        angle_to_target = math.atan2(to_target.y, to_target.x)
        angle_diff = wrap_angle(angle_to_target - self.agent.heading)
        line_of_sight = self.world.has_line_of_sight(self.agent.position, target.position)
        return to_target, distance, angle_to_target, angle_diff, line_of_sight

    def _angle_to_unit(self, angle: float) -> Vec2:
        return Vec2(math.cos(angle), math.sin(angle)).normalize()

    def _update_timers(self) -> None:
        self._combat_strafe_timer -= self.world.dt
        self._nav_commit_timer -= self.world.dt
        self._recovery_timer -= self.world.dt

        if self.profile == "orbit_no_shoot":
            strafe_range = (2.0, 3.0)
            nav_probability = 0.10
            nav_range = (1.4, 2.2)
        elif self.profile == "easy_duel":
            strafe_range = (1.6, 2.4)
            nav_probability = 0.15
            nav_range = (1.2, 2.0)
        elif self.profile == "moderate_duel":
            strafe_range = (1.2, 1.8)
            nav_probability = 0.22
            nav_range = (1.0, 1.6)
        else:
            strafe_range = (0.8, 1.6)
            nav_probability = 0.35
            nav_range = (0.7, 1.2)

        if self._combat_strafe_timer <= 0.0:
            self._combat_strafe_sign *= -1
            self._combat_strafe_timer = self._uniform(*strafe_range)
        if self._nav_commit_timer <= 0.0:
            if self._random() < nav_probability:
                self._nav_sign *= -1
            self._nav_commit_timer = self._uniform(*nav_range)

    def _update_stuck_state(self, target: Agent) -> None:
        moved = self.agent.position.distance_to(self._last_position)
        self._last_position = self.agent.position.copy()
        if self.agent.blocked_last_step:
            self._blocked_streak += 1
        else:
            self._blocked_streak = max(0, self._blocked_streak - 1)

        if moved < 0.25 and self.agent.can_move:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0

        if self._blocked_streak >= 2 or self._stuck_frames >= 35:
            self._recovery_timer = self._uniform(0.45, 0.85)
            self._recovery_angle = self._best_recovery_angle(target)
            self._blocked_streak = 0
            self._stuck_frames = 0
            self._combat_strafe_sign *= -1
            self._nav_sign *= -1

    def _recovery_command(self, target: Agent) -> AgentCommand | None:
        if self._recovery_timer <= 0.0 or not self.agent.can_move:
            return None
        _, _, angle_to_target, angle_diff, _ = self._target_geometry(target)
        rotate = self._choose_rotation(angle_diff, dead_zone=0.12)
        direction = self._angle_to_unit(self._recovery_angle)
        move_idx, strafe_idx = self._world_direction_to_relative_command(direction)
        return AgentCommand(move_idx=move_idx, strafe_idx=strafe_idx, rot_idx=rotate, shoot=False, reload=False)

    def _best_recovery_angle(self, target: Agent) -> float:
        away_from_target = math.atan2(self.agent.position.y - target.position.y, self.agent.position.x - target.position.x)
        candidates = [away_from_target + math.radians(offset) for offset in (0, 35, -35, 70, -70, 110, -110, 150, -150, 180)]
        best_angle = away_from_target
        best_clearance = -1.0
        for angle in candidates:
            direction = self._angle_to_unit(angle)
            clearance = self._direction_clearance(direction, max_probe_distance=90.0)
            if clearance > best_clearance:
                best_clearance = clearance
                best_angle = angle
        return best_angle

    def _choose_rotation(self, angle_diff: float, dead_zone: float) -> int:
        if angle_diff < -dead_zone:
            return 1
        if angle_diff > dead_zone:
            return 2
        return 0

    def _should_reload(self, line_of_sight: bool, conservative: bool) -> bool:
        if self.agent.reloading:
            return False
        charges = getattr(self.agent, "reload_charges", None)
        if charges is not None and charges <= 0:
            return False

        
        if getattr(self.world, "stage_id", "") == "stage8":
            target_ammo = max(1, int(round(float(self.agent.max_ammo) / 3.0)))
            if self.agent.ammo == 0:
                return True
            
            if self.agent.ammo <= target_ammo and not line_of_sight:
                return True
            if self.agent.ammo < target_ammo:
                return True
            return False

        if self.agent.ammo == 0:
            return True
        if conservative and self.agent.ammo <= 2 and not line_of_sight:
            return True
        return False

    def _choose_move_direction(self, desired_angle: float, target: Agent, current_line_of_sight: bool, prefer_line_of_sight: bool, keep_range: bool) -> Vec2:
        return self._drive_movement_intent(self._angle_to_unit(desired_angle), target=target, current_line_of_sight=current_line_of_sight)

    def _choose_basic_direction(self, desired_angle: float) -> Vec2:
        return self._drive_movement_intent(self._angle_to_unit(desired_angle))

    def _direction_clearance(self, direction: Vec2, max_probe_distance: float) -> float:
        direction = direction.normalize()
        last_clear = 0.0
        for probe_distance in (18.0, 30.0, 44.0, 60.0, max_probe_distance):
            probe_position = self.agent.position + (direction * probe_distance)
            if self.world.is_circle_blocked(probe_position, self.agent.radius):
                return last_clear
            last_clear = probe_distance
        return last_clear

    def _world_direction_to_relative_command(self, direction: Vec2) -> tuple[int, int]:
        if direction.length() <= 1e-9:
            return 0, 0
        forward = Vec2(math.cos(self.agent.heading), math.sin(self.agent.heading))
        right = Vec2(-forward.y, forward.x)
        forward_amount = self._dot(direction, forward)
        right_amount = self._dot(direction, right)
        move_idx = 1 if forward_amount > 0.15 else 2 if forward_amount < -0.15 else 0
        strafe_idx = 2 if right_amount > 0.15 else 1 if right_amount < -0.15 else 0
        return move_idx, strafe_idx

    @staticmethod
    def _dot(a: Vec2, b: Vec2) -> float:
        return a.x * b.x + a.y * b.y


class ModelBot:
    
    def __init__(self, model_path: str | Path, stage_id: str, agent: Agent, world: World, target_id: str):
        from training.model_utils import load_any_sb3_model, predict_with_action_masks

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.stage_id = normalize_stage_id(stage_id)
        self.stage = get_stage(self.stage_id)
        self.model = load_any_sb3_model(str(self.model_path), device="cpu")
        self._predict_with_action_masks = predict_with_action_masks
        self.agent = agent
        self.world = world
        self.target_id = target_id
        self.obs_builder = ObsBuilder()
        self.skill_table = build_skill_table(self.stage_id, evaluation_mode=True)
        self._frames: deque[np.ndarray] = deque(maxlen=self.stage.frame_stack)
        self._stage8_starting_ammo = self.agent.max_ammo
        self._stage8_shots_fired = 0
        self._last_ammo: int | None = None

    def reset_memory(self) -> None:
        self._frames.clear()
        self._stage8_shots_fired = 0
        self._last_ammo = int(self.agent.ammo)

    def _target(self) -> Agent | None:
        target = self.world.agents.get(self.target_id)
        if target is None or not target.is_alive or not self.agent.is_alive:
            return None
        return target

    def _has_line_of_sight_to_target(self, target: Agent | None = None) -> bool:
        target = target or self._target()
        if target is None:
            return False
        return bool(self.world.has_line_of_sight(self.agent.position, target.position))

    def _aim_error_to_target(self, target: Agent | None = None, heading: float | None = None) -> float:
        target = target or self._target()
        if target is None:
            return math.pi
        shot_heading = self.agent.heading if heading is None else float(heading)
        target_angle = math.atan2(target.position.y - self.agent.position.y, target.position.x - self.agent.position.x)
        return abs(wrap_angle(target_angle - shot_heading))

    def _clear_shot_to_target(self, target: Agent | None = None, heading: float | None = None) -> bool:
        target = target or self._target()
        if target is None:
            return False
        shot_heading = self.agent.heading if heading is None else float(heading)
        direction = Vec2(math.cos(shot_heading), math.sin(shot_heading))
        muzzle = self.agent.position + direction * 20.0
        distance_to_target = muzzle.distance_to(target.position)
        ray = detection.trace_ray(
            world=self.world,
            origin=muzzle,
            direction=direction,
            max_length=distance_to_target + target.radius + 8.0,
            step_size=4.0,
            target_agent_id=target.agent_id,
            observer_agent_id=self.agent.agent_id,
            include_pickups=False,
            include_projectiles=False,
            enemy_margin=3.0,
        )
        return ray.hit_type == detection.HIT_ENEMY and ray.hit_agent_id == target.agent_id

    def _stage8_shoot_allowed_now(self, target: Agent | None = None) -> bool:
        
        target = target or self._target()
        if target is None:
            return False
        return bool(
            self.agent.is_alive
            and self.agent.fire_ready >= 0.5
            and self.agent.ammo > 0
            and not getattr(self.agent, "reloading", False)
        )

    def _stage8_reload_allowed_now(self) -> bool:
        
        if self.stage_id != "stage8" or not self.agent.is_alive:
            return True
        if getattr(self.agent, "reloading", False):
            return False
        if self.agent.ammo >= self.agent.max_ammo:
            return False
        charges = getattr(self.agent, "reload_charges", None)
        if charges is not None and charges <= 0:
            return False
        return True

    def _update_stage8_shot_counter(self) -> None:
        if self.stage_id != "stage8":
            self._last_ammo = int(self.agent.ammo)
            return
        current_ammo = int(self.agent.ammo)
        if self._last_ammo is None:
            self._last_ammo = current_ammo
            return
        if current_ammo < self._last_ammo:
            self._stage8_shots_fired += int(self._last_ammo - current_ammo)
        self._last_ammo = current_ammo

    def action_masks(self) -> np.ndarray:
        target = self._target()
        if target is None:
            return stage_action_mask(self.stage_id, shoot_allowed=False, reload_allowed=False)
        if self.stage_id == "stage8":
            return stage_action_mask(
                self.stage_id,
                shoot_allowed=self._stage8_shoot_allowed_now(target),
                reload_allowed=self._stage8_reload_allowed_now(),
            )
        if self.stage_id == "stage7":
            shoot_allowed = bool(self.agent.fire_ready >= 0.5 and self._clear_shot_to_target(target))
            return stage_action_mask(self.stage_id, shoot_allowed=shoot_allowed)
        if self.stage_id in {"stage2", "stage5", "stage6"}:
            shoot_allowed = bool(self.agent.fire_ready >= 0.5 and self.agent.ammo > 0 and not getattr(self.agent, "reloading", False))
            return stage_action_mask(self.stage_id, shoot_allowed=shoot_allowed)
        return stage_action_mask(self.stage_id)

    def tick(self) -> AgentCommand:
        target = self._target()
        if target is None:
            return AgentCommand()

        self._update_stage8_shot_counter()

        obs = self.obs_builder.build(self.agent, target, self.world, self.skill_table, stage_id=self.stage_id)
        if not self._frames:
            for _ in range(self.stage.frame_stack):
                self._frames.append(obs)
        else:
            self._frames.append(obs)
        stacked_obs = np.concatenate(list(self._frames), axis=0).astype(np.float32)
        self.world.debug_rays = list(self.obs_builder.last_debug_rays)

       
        action, _ = self._predict_with_action_masks(self.model, stacked_obs, self, deterministic=True)
        command = action_to_command(self.stage_id, action)

        
        if self.stage_id == "stage8":
            if command.shoot and not self._stage8_shoot_allowed_now(target):
                command.shoot = False
            if command.reload and not self._stage8_reload_allowed_now():
                command.reload = False
        elif self.stage_id == "stage7" and command.shoot and not self._clear_shot_to_target(target):
            command.shoot = False

        return command
