from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

from config.game_constants import (
    GLOCK_DAMAGE_HP,
    GLOCK_FIRE_RATE,
    GLOCK_MAX_AMMO,
    GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
    GLOCK_RELOAD_TIME,
    PLAYER_MAX_HEALTH,
    PLAYER_MOVE_SPEED_PX_PER_SEC,
)
from config.stages import get_stage, normalize_stage_id
from game import detection
from game.agent import Agent, AgentCommand, wrap_angle
from game.bot import StockBot
from game.maps import build_layout
from game.world import DebugPolyline, Event, Vec2, World
from rendering.renderer import Renderer
from rl.action_space import ACTION_NVECS, decode_action, sample_valid_action, stage_action_mask
from rl.obs_builder import ObsBuilder
from rl.reward_manager import RewardManager
from navigation import NavigationReferenceTracker
from training.skill_table import build_skill_table


STAGE8_RELOAD_CHARGES = 3
# Stage 8 now keeps the same Glock model as the rest of the project.
# Reload pressure is produced by sequential enemy waves, not by shrinking the magazine.
STAGE8_MAX_AMMO = GLOCK_MAX_AMMO
STAGE8_STARTING_AMMO = GLOCK_MAX_AMMO
STAGE8_DEFAULT_ENEMY_WAVES = 5
# In final Stage 8E, terminate episodes where the learner sits at 0 ammo
# despite having reload reserves.  This prevents thousand-step empty-ammo tails.
STAGE8_EMPTY_AMMO_TIMEOUT_STEPS = 30
STAGE8_LOW_AMMO_DECISION_THRESHOLD = 8


class ShooterEnv(gym.Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        stage_id: str = "stage1",
        render_mode: str | None = None,
        show_raycast_debug: bool = False,
        skill_table: dict | None = None,
        validation_mode: bool = False,
        frame_stack: int | None = None,
        observation_stage_id: str | None = None,
    ):
        super().__init__()

        self.stage_id = normalize_stage_id(stage_id)
        self.stage = get_stage(self.stage_id)
        self.observation_stage_id = normalize_stage_id(observation_stage_id or self.stage_id)

        self.render_mode = render_mode
        self.show_raycast_debug = show_raycast_debug
        self.validation_mode = bool(validation_mode)

        # Stage 4 curriculum: training can use a temporary projectile-speed
        # scale, but validation/evaluation always uses the official speed.
        self.stage4_projectile_speed_scale = 1.0
        self.stage4_enemy_fire_rate_scale = 1.0
        self.stage4_first_shot_delay_steps = 0
        self.stage4_max_active_enemy_projectiles = 99
        self.stage4_curriculum_level = 0
        self.stage4_curriculum_total_levels = 0

        self.stage5_curriculum_level = 1
        self.stage5_curriculum_total_levels = 3
        self.stage5_curriculum_phase_index = 1
        self.stage5_enemy_fire_rate_scale = 0.0
        self.stage5_first_shot_delay_steps = 9999
        self.stage5_max_active_enemy_projectiles = 0
        self.stage5_enemy_can_move = False
        self.stage5_enemy_can_shoot = False

        self.stage6_curriculum_level = 1
        self.stage6_curriculum_total_levels = 4
        self.stage6_curriculum_phase_index = 1
        self.stage6_enemy_fire_rate_scale = 0.55
        self.stage6_first_shot_delay_steps = 90
        self.stage6_max_active_enemy_projectiles = 1
        self.stage6_enemy_can_move = True
        self.stage6_enemy_can_shoot = True

        self.stage7_curriculum_level = 5 if self.validation_mode and self.stage_id == "stage7" else 1
        self.stage7_curriculum_total_levels = 5
        self.stage7_curriculum_phase_index = 5 if self.validation_mode and self.stage_id == "stage7" else 1
        self.stage7_max_steps = 2000 if self.validation_mode and self.stage_id == "stage7" else 1200
        self.stage7_obstacle_count = 4 if self.validation_mode and self.stage_id == "stage7" else 1
        self.stage7_enemy_fire_rate_scale = 1.00 if self.validation_mode and self.stage_id == "stage7" else 0.55
        self.stage7_first_shot_delay_steps = 0 if self.validation_mode and self.stage_id == "stage7" else 120
        self.stage7_max_active_enemy_projectiles = 99 if self.validation_mode and self.stage_id == "stage7" else 1
        self.stage7_enemy_can_move = True if self.validation_mode and self.stage_id == "stage7" else False
        self.stage7_enemy_can_shoot = True

        self.stage8_curriculum_level = 6 if self.validation_mode and self.stage_id == "stage8" else 1
        self.stage8_curriculum_total_levels = 6
        self.stage8_curriculum_phase_index = 6 if self.validation_mode and self.stage_id == "stage8" else 1
        self.stage8_max_steps = 2600 if self.validation_mode and self.stage_id == "stage8" else 1600
        self.stage8_obstacle_count = 1 if self.validation_mode and self.stage_id == "stage8" else 0
        self.stage8_enemy_fire_rate_scale = 0.25 if self.validation_mode and self.stage_id == "stage8" else 0.0
        self.stage8_first_shot_delay_steps = 150 if self.validation_mode and self.stage_id == "stage8" else 9999
        self.stage8_max_active_enemy_projectiles = 1 if self.validation_mode and self.stage_id == "stage8" else 0
        self.stage8_enemy_can_move = False
        self.stage8_enemy_can_shoot = True if self.validation_mode and self.stage_id == "stage8" else False
        self.stage8_enemy_waves = STAGE8_DEFAULT_ENEMY_WAVES if self.stage_id == "stage8" else 0
        self.stage8_current_wave = 1
        self.stage8_waves_completed = 0
        self.stage8_consecutive_empty_ammo_steps = 0
        self.stage8_empty_ammo_timeout_triggered = False
        self.stage8_low_ammo_decision_steps = 0
        self.stage8_low_ammo_decision_active = False
        self.stage8_last_enemy_respawn_index = -1
        self.stage8_controlled_respawn_used = False

        self.skill_table = skill_table or build_skill_table(
            self.stage_id,
            evaluation_mode=validation_mode,
        )

        self.frame_stack = int(frame_stack or self.stage.frame_stack)

        # Simulation timestep.
        # World uses 30 FPS logic, so dt = 1 / 30 seconds.
        self.dt = 1.0 / 30.0

        base_dim = ObsBuilder.obs_dim_for_stage(self.observation_stage_id)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(base_dim * self.frame_stack,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(ACTION_NVECS)

        self.layout = build_layout(
            self.stage_id,
            validation=self.validation_mode,
            stage7_phase_index=self.stage7_curriculum_phase_index if self.stage_id == "stage7" else None,
            stage8_phase_index=self.stage8_curriculum_phase_index if self.stage_id == "stage8" else None,
        )

        self.world: World | None = None
        self.learner: Agent | None = None
        self.enemy: Agent | None = None
        self.enemy_bot: StockBot | None = None

        self.obs_builder = ObsBuilder()
        self.reward_manager = RewardManager(self.stage_id)

        self.nav_tracker = (
            NavigationReferenceTracker(ideal_range=RewardManager.IDEAL_COMBAT_RANGE)
            if self._uses_navigation_reference()
            else None
        )

        self.renderer: Renderer | None = None
        self._step_count = 0
        self._save_requested = False
        self.episode_spawn_index = -1
        self.episode_spawn_pair_id = -1
        self.episode_obstacle_variant_id = -1
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def _stage8_empty_ammo_timeout_exceeded(self) -> bool:
        """Return True when 8E is stuck at empty ammo with reloads available."""
        if self.stage_id != "stage8" or self.learner is None:
            return False
        if int(getattr(self, "stage8_curriculum_phase_index", 1)) < 5:
            self.stage8_consecutive_empty_ammo_steps = 0
            return False

        # In 8E/8F, sitting at 0 ammo is a terminal tactical failure whether
        # reload reserves remain or were already wasted through early reloads.
        # This removes long empty-ammo tails and gives PPO a cleaner signal.
        stuck_empty = (
            self.learner.ammo <= 0
            and not getattr(self.learner, "reloading", False)
            and self.learner.is_alive
        )
        if stuck_empty:
            self.stage8_consecutive_empty_ammo_steps += 1
        else:
            self.stage8_consecutive_empty_ammo_steps = 0

        return self.stage8_consecutive_empty_ammo_steps >= STAGE8_EMPTY_AMMO_TIMEOUT_STEPS

    def _stage8_low_ammo_decision_active_now(self) -> bool:
        """Return True when 8E should prefer reloading before reaching empty ammo."""
        if self.stage_id != "stage8" or self.learner is None:
            return False
        if int(getattr(self, "stage8_curriculum_phase_index", 1)) < 5:
            return False

        reload_charges = getattr(self.learner, "reload_charges", None)
        has_reload_reserve = reload_charges is None or int(reload_charges) > 0
        return bool(
            self.learner.is_alive
            and has_reload_reserve
            and not getattr(self.learner, "reloading", False)
            and 0 < int(getattr(self.learner, "ammo", 0)) <= STAGE8_LOW_AMMO_DECISION_THRESHOLD
        )

    def _update_stage8_low_ammo_decision_window(self) -> None:
        """Track low-ammo 8E decision opportunities for logging/debugging."""
        active = self._stage8_low_ammo_decision_active_now()
        self.stage8_low_ammo_decision_active = active
        if active:
            self.stage8_low_ammo_decision_steps += 1

    def _effective_max_steps(self) -> int:
        if self.stage_id == "stage7":
            return int(getattr(self, "stage7_max_steps", self.stage.max_steps))
        if self.stage_id == "stage8":
            return int(getattr(self, "stage8_max_steps", self.stage.max_steps))
        return int(self.stage.max_steps)

    def _uses_navigation_reference(self) -> bool:
        # Stage 3 introduces the reward-guided navigation reference.
        # Later combat movement stages reuse it, especially when obstacles are enabled.
        # Stage 1 and Stage 2 defaults are intentionally left unchanged.
        return self.stage_id in {"stage3", "stage5", "stage6", "stage7", "stage8"}

    def _update_navigation_debug_points(self, nav_reference) -> None:
        if self.world is None or nav_reference is None or not nav_reference.valid:
            return

        self.world.debug_points.append(
            (
                nav_reference.desired_position.x,
                nav_reference.desired_position.y,
                "desired_combat_position",
            )
        )
        self.world.debug_points.append(
            (
                nav_reference.waypoint.x,
                nav_reference.waypoint.y,
                "navigation_waypoint",
            )
        )

        path_points = [(point.x, point.y) for point in getattr(nav_reference, "path", ())]

        # The reward target remains the desired combat point, but the visual A*
        # guide continues to the enemy so the viewer understands what the path is
        # relative to.
        if self.enemy is not None and path_points:
            enemy_point = (self.enemy.position.x, self.enemy.position.y)
            if path_points[-1] != enemy_point:
                path_points = [*path_points, enemy_point]

        if len(path_points) >= 2:
            self.world.debug_polylines.append(
                DebugPolyline(points=path_points, kind="astar_path", width=3)
            )

    def _learner_has_line_of_sight_to_enemy(self) -> bool:
        if self.world is None or self.learner is None or self.enemy is None:
            return True
        if not self.learner.is_alive or not self.enemy.is_alive:
            return False
        return bool(self.world.has_line_of_sight(self.learner.position, self.enemy.position))

    def _predicted_heading_after_rotation(self, rot_idx: int) -> float:
        if self.learner is None:
            return 0.0
        heading = self.learner.heading
        if int(rot_idx) == 1:
            heading -= self.learner.TURN_SPEED * self.dt
        elif int(rot_idx) == 2:
            heading += self.learner.TURN_SPEED * self.dt
        return wrap_angle(heading)

    def _learner_clear_shot_to_enemy(self, heading: float | None = None) -> bool:
        """Return whether the current Glock muzzle ray would directly hit the enemy.

        This is deliberately stricter than center-to-center line of sight. Stage 7
        obstacles can leave the target center visible while the actual projectile
        muzzle/path is still blocked by cover.
        """
        if self.world is None or self.learner is None or self.enemy is None:
            return True
        if not self.learner.is_alive or not self.enemy.is_alive:
            return False
        shot_heading = self.learner.heading if heading is None else float(heading)
        direction = Vec2(math.cos(shot_heading), math.sin(shot_heading))
        muzzle = self.learner.position + direction * 20.0
        distance_to_enemy = muzzle.distance_to(self.enemy.position)
        ray = detection.trace_ray(
            world=self.world,
            origin=muzzle,
            direction=direction,
            max_length=distance_to_enemy + self.enemy.radius + 8.0,
            step_size=4.0,
            target_agent_id=self.enemy.agent_id,
            observer_agent_id=self.learner.agent_id,
            include_pickups=False,
            include_projectiles=False,
            enemy_margin=3.0,
        )
        return ray.hit_type == detection.HIT_ENEMY and ray.hit_agent_id == self.enemy.agent_id

    def _learner_clear_shot_after_action_rotation(self, rot_idx: int) -> bool:
        return self._learner_clear_shot_to_enemy(self._predicted_heading_after_rotation(rot_idx))

    def _learner_aim_error_after_action_rotation(self, rot_idx: int) -> float:
        if self.learner is None or self.enemy is None or not self.enemy.is_alive:
            return math.pi
        predicted = self._predicted_heading_after_rotation(rot_idx)
        dx = self.enemy.position.x - self.learner.position.x
        dy = self.enemy.position.y - self.learner.position.y
        target_angle = math.atan2(dy, dx)
        return abs(wrap_angle(target_angle - predicted))

    def _stage8_shoot_allowed_after_rotation(self, rot_idx: int = 0) -> bool:
        """Physical Stage 8 shoot availability.

        Stage 8 is the final integration stage.  Tactical choices such as
        shooting without LOS, shooting while badly aimed, or shooting into cover
        must remain available so PPO can learn to avoid them through reward
        penalties.  The mask therefore only blocks physically unavailable shots.
        """
        if self.learner is None:
            return False
        learner = self.learner
        return bool(
            learner.is_alive
            and learner.ammo > 0
            and learner.fire_ready >= 0.5
            and not getattr(learner, "reloading", False)
        )

    def _stage8_reload_allowed_now(self) -> bool:
        """Physical Stage 8 reload availability.

        Reload timing is no longer tactically masked.  Reloading too early, too
        late, or at empty ammo is handled by reward penalties.  The mask only
        blocks physically impossible reloads: full magazine, already reloading,
        dead agent, or no remaining reload charges.
        """
        if self.stage_id != "stage8" or self.learner is None:
            return True
        learner = self.learner
        if not learner.is_alive or getattr(learner, "reloading", False):
            return False
        if learner.ammo >= learner.max_ammo:
            return False
        charges = getattr(learner, "reload_charges", None)
        if charges is not None and charges <= 0:
            return False
        return True

    def action_masks(self) -> np.ndarray:
        if self.stage_id in {"stage2", "stage5", "stage6", "stage7", "stage8"} and self.learner is not None:
            # These stages should learn when to fire, not spam unavailable shots
            # while the weapon is cooling down, reloading, or empty. In Stage 7,
            # shooting is also blocked when obstacles hide the enemy, because the
            # stage objective is to recover line of sight before firing.
            shoot_allowed = self.learner.fire_ready >= 0.5
            reload_allowed = None
            if self.stage_id == "stage7":
                shoot_allowed = shoot_allowed and self._learner_clear_shot_to_enemy()
            elif self.stage_id == "stage8":
                # Final integration stage: only physically unavailable actions are
                # masked.  Tactical mistakes remain selectable and are learned via
                # reward penalties.
                shoot_allowed = self._stage8_shoot_allowed_after_rotation(0)
                reload_allowed = self._stage8_reload_allowed_now()

            return stage_action_mask(
                self.stage_id,
                shoot_allowed=shoot_allowed,
                reload_allowed=reload_allowed,
            )

        return stage_action_mask(self.stage_id)

    def sample_valid_action(self) -> np.ndarray:
        return sample_valid_action(self.stage_id, self.np_random)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.layout = build_layout(
            self.stage_id,
            rng=self.np_random,
            validation=self.validation_mode,
            stage7_phase_index=self.stage7_curriculum_phase_index if self.stage_id == "stage7" else None,
            stage8_phase_index=self.stage8_curriculum_phase_index if self.stage_id == "stage8" else None,
        )

        self.world = World(
            self.layout,
            dt=self.dt,
            stage_id=self.stage_id,
        )
        self._apply_stage4_pressure_settings_to_world()
        self._apply_stage5_settings_to_world()
        self._apply_stage6_settings_to_world()
        self._apply_stage7_settings_to_world()
        self._apply_stage8_settings_to_world()

        self.learner = Agent(
            agent_id="learner",
            can_move=self.stage.learner_can_move,
            move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
            max_health=PLAYER_MAX_HEALTH,
            max_ammo=GLOCK_MAX_AMMO,
            fire_rate=GLOCK_FIRE_RATE,
            reload_time=GLOCK_RELOAD_TIME,
            projectile_damage=GLOCK_DAMAGE_HP,
            projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
            auto_reload_when_empty=self.stage.learner_auto_reload,
            reload_charges=STAGE8_RELOAD_CHARGES if self.stage_id == "stage8" else None,
        )

        self.enemy = Agent(
            agent_id="enemy",
            can_move=self.stage.enemy_can_move,
            move_speed=self._episode_enemy_move_speed(),
            max_health=PLAYER_MAX_HEALTH,
            max_ammo=GLOCK_MAX_AMMO,
            fire_rate=self._episode_enemy_fire_rate(),
            reload_time=self.stage.enemy_reload_time or GLOCK_RELOAD_TIME,
            projectile_damage=GLOCK_DAMAGE_HP,
            projectile_speed=self._episode_enemy_projectile_speed(),
            auto_reload_when_empty=False if self.stage_id == "stage8" else True,
            reload_charges=STAGE8_RELOAD_CHARGES if self.stage_id == "stage8" else None,
        )

        self.world.add_agent(self.learner)
        self.world.add_agent(self.enemy)
        self.world.reset_round()
        self._apply_stage8_reload_pressure()
        if self.stage_id == "stage8":
            self.stage8_current_wave = 1
            self.stage8_waves_completed = 0
            self.stage8_consecutive_empty_ammo_steps = 0
            self.stage8_empty_ammo_timeout_triggered = False
            self.stage8_low_ammo_decision_steps = 0
            self.stage8_low_ammo_decision_active = False
            self.stage8_last_enemy_respawn_index = -1
            self.stage8_controlled_respawn_used = False

        self._place_agents()

        self.enemy_bot = StockBot(
            agent=self.enemy,
            world=self.world,
            target_id="learner",
            profile=self.stage.enemy_behavior_profile,
            rng=self.np_random,
        )

        self.reward_manager.reset(self.learner, self.enemy, self.world)
        self._apply_stage5_settings_to_reward_manager()
        self._apply_stage6_settings_to_reward_manager()
        self._apply_stage7_settings_to_reward_manager()
        self._apply_stage8_settings_to_reward_manager()

        if self.nav_tracker is not None:
            self.nav_tracker.reset()
            self.nav_tracker.update(
                learner=self.learner,
                enemy=self.enemy,
                world=self.world,
                force=True,
            )

        self._step_count = 0
        self._save_requested = False

        obs = self.obs_builder.build(
            self.learner,
            self.enemy,
            self.world,
            self.skill_table,
            stage_id=self.observation_stage_id,
        )

        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(obs)

        self.world.debug_rays = list(self.obs_builder.last_debug_rays)

        if self.render_mode == "human":
            self._ensure_renderer()
            self.render()

        return self._stacked_obs(), {}

    def step(self, action: Any):
        if (
            self.world is None
            or self.learner is None
            or self.enemy is None
            or self.enemy_bot is None
        ):
            raise RuntimeError("Environment must be reset before calling step().")

        self._step_count += 1

        if self.render_mode == "human":
            self._ensure_renderer()
            if not self.renderer.sync_runtime(lambda: self._draw_frame()):
                raise KeyboardInterrupt("Training/test window closed by user.")
            if self.renderer.consume_save_request():
                self._save_requested = True

        decoded = decode_action(self.stage_id, action)
        learner_command = decoded.command

        events: list[Event] = []

        if self.stage_id == "stage8" and learner_command.reload and not self._stage8_reload_allowed_now():
            learner_command.reload = False
            events.append(
                Event(
                    "invalid_action",
                    self.learner.agent_id,
                    data={
                        "ability": "stage8_reload_physically_unavailable",
                        "ammo": self.learner.ammo,
                        "max_ammo": self.learner.max_ammo,
                        "reload_charges": getattr(self.learner, "reload_charges", None),
                        "raw_action": decoded.raw_action,
                    },
                )
            )

        if self.stage_id == "stage7" and learner_command.shoot:
            predicted_heading = self._predicted_heading_after_rotation(learner_command.rot_idx)
            has_center_los = self._learner_has_line_of_sight_to_enemy()
            has_clear_shot = self._learner_clear_shot_to_enemy(predicted_heading)
            aim_error_after_rotation = self._learner_aim_error_after_action_rotation(learner_command.rot_idx)
            if not has_clear_shot:
                learner_command.shoot = False
                events.append(
                    Event(
                        "invalid_action",
                        self.learner.agent_id,
                        data={
                            "ability": "shoot_without_clear_shot",
                            "has_center_line_of_sight": has_center_los,
                            "aim_error_after_rotation": aim_error_after_rotation,
                            "raw_action": decoded.raw_action,
                        },
                    )
                )

        for ability in decoded.invalid_actions:
            events.append(
                Event(
                    "invalid_action",
                    self.learner.agent_id,
                    data={
                        "ability": ability,
                        "raw_action": decoded.raw_action,
                    },
                )
            )

        events.extend(self.learner.apply_command(learner_command, self.world))

        enemy_command = self.enemy_bot.tick()
        enemy_command = self._mask_enemy_command(enemy_command)

        events.extend(self.enemy.apply_command(enemy_command, self.world))
        events.extend(self.world.tick())

        nav_reference = None
        if self.nav_tracker is not None:
            nav_reference = self.nav_tracker.update(
                learner=self.learner,
                enemy=self.enemy,
                world=self.world,
            )
            self._update_navigation_debug_points(nav_reference)

        reward = self.reward_manager.compute(
            events,
            self.learner,
            self.enemy,
            self.world,
            self._step_count,
            nav_reference=nav_reference,
        )

        self._update_stage8_low_ammo_decision_window()
        stage8_empty_timeout = self._stage8_empty_ammo_timeout_exceeded()
        if stage8_empty_timeout:
            self.stage8_empty_ammo_timeout_triggered = True

        stage_success_now = self.reward_manager.episode_success()

        stage8_wave_advanced = self._advance_stage8_wave_if_needed()
        terminated = (
            (not self.learner.is_alive)
            or ((not self.enemy.is_alive) and not stage8_wave_advanced)
            or stage8_empty_timeout
        )

        if self.stage.end_episode_on_success and stage_success_now:
            terminated = True

        truncated = self._step_count >= self._effective_max_steps()

        obs = self.obs_builder.build(
            self.learner,
            self.enemy,
            self.world,
            self.skill_table,
            stage_id=self.observation_stage_id,
        )

        self._frames.append(obs)
        self.world.debug_rays = list(self.obs_builder.last_debug_rays)

        info: dict[str, Any] = {}

        if terminated or truncated:
            info = self.reward_manager.get_episode_stats(steps=self._step_count)
            info["stage"] = self.stage_id
            info["stage_name"] = self.stage.name
            info["skill_table"] = self.skill_table
            info["validation_mode"] = self.validation_mode
            info["max_steps"] = self._effective_max_steps()
            if self.stage_id == "stage7":
                info["stage7_max_steps"] = self._effective_max_steps()
            info["stage4_curriculum_level"] = self._effective_stage4_curriculum_level()
            info["stage4_curriculum_total_levels"] = self.stage4_curriculum_total_levels
            info["stage4_projectile_speed_scale"] = self._effective_stage4_projectile_speed_scale()
            info["stage4_enemy_fire_rate_scale"] = self._effective_stage4_enemy_fire_rate_scale()
            info["stage4_max_active_enemy_projectiles"] = self._effective_stage4_max_active_enemy_projectiles()
            info["stage4_first_shot_delay_steps"] = self._effective_stage4_first_shot_delay_steps()
            info["stage5_curriculum_level"] = self.stage5_curriculum_level if self.stage_id == "stage5" else 0
            info["stage5_curriculum_total_levels"] = self.stage5_curriculum_total_levels if self.stage_id == "stage5" else 0
            info["stage5_curriculum_phase_index"] = self.stage5_curriculum_phase_index if self.stage_id == "stage5" else 0
            info["stage5_enemy_fire_rate_scale"] = self.stage5_enemy_fire_rate_scale if self.stage_id == "stage5" else 0.0
            info["stage5_first_shot_delay_steps"] = self.stage5_first_shot_delay_steps if self.stage_id == "stage5" else 0
            info["stage5_max_active_enemy_projectiles"] = self.stage5_max_active_enemy_projectiles if self.stage_id == "stage5" else 0
            info["stage5_enemy_can_move"] = 1 if (self.stage_id == "stage5" and self.stage5_enemy_can_move) else 0
            info["stage5_enemy_can_shoot"] = 1 if (self.stage_id == "stage5" and self.stage5_enemy_can_shoot) else 0
            info["stage6_curriculum_level"] = self.stage6_curriculum_level if self.stage_id == "stage6" else 0
            info["stage6_curriculum_total_levels"] = self.stage6_curriculum_total_levels if self.stage_id == "stage6" else 0
            info["stage6_curriculum_phase_index"] = self.stage6_curriculum_phase_index if self.stage_id == "stage6" else 0
            info["stage6_enemy_fire_rate_scale"] = self.stage6_enemy_fire_rate_scale if self.stage_id == "stage6" else 0.0
            info["stage6_first_shot_delay_steps"] = self.stage6_first_shot_delay_steps if self.stage_id == "stage6" else 0
            info["stage6_max_active_enemy_projectiles"] = self.stage6_max_active_enemy_projectiles if self.stage_id == "stage6" else 0
            info["stage6_enemy_can_move"] = 1 if (self.stage_id == "stage6" and self.stage6_enemy_can_move) else 0
            info["stage6_enemy_can_shoot"] = 1 if (self.stage_id == "stage6" and self.stage6_enemy_can_shoot) else 0
            info["stage7_curriculum_level"] = self.stage7_curriculum_level if self.stage_id == "stage7" else 0
            info["stage7_curriculum_total_levels"] = self.stage7_curriculum_total_levels if self.stage_id == "stage7" else 0
            info["stage7_curriculum_phase_index"] = self.stage7_curriculum_phase_index if self.stage_id == "stage7" else 0
            info["stage7_obstacle_count"] = self.stage7_obstacle_count if self.stage_id == "stage7" else 0
            info["stage7_enemy_fire_rate_scale"] = self.stage7_enemy_fire_rate_scale if self.stage_id == "stage7" else 0.0
            info["stage7_first_shot_delay_steps"] = self.stage7_first_shot_delay_steps if self.stage_id == "stage7" else 0
            info["stage7_max_active_enemy_projectiles"] = self.stage7_max_active_enemy_projectiles if self.stage_id == "stage7" else 0
            info["stage7_enemy_can_move"] = 1 if (self.stage_id == "stage7" and self.stage7_enemy_can_move) else 0
            info["stage7_enemy_can_shoot"] = 1 if (self.stage_id == "stage7" and self.stage7_enemy_can_shoot) else 0
            info["stage7_obstacle_variant_id"] = self.episode_obstacle_variant_id if self.stage_id == "stage7" else -1
            info["stage7_spawn_index"] = self.episode_spawn_index if self.stage_id == "stage7" else -1
            info["stage7_spawn_pair_id"] = self.episode_spawn_pair_id if self.stage_id == "stage7" else -1
            info["stage8_curriculum_level"] = self.stage8_curriculum_level if self.stage_id == "stage8" else 0
            info["stage8_curriculum_total_levels"] = self.stage8_curriculum_total_levels if self.stage_id == "stage8" else 0
            info["stage8_curriculum_phase_index"] = self.stage8_curriculum_phase_index if self.stage_id == "stage8" else 0
            info["stage8_phase_name"] = (f"8{chr(ord('A') + self.stage8_curriculum_phase_index - 1)}" if self.stage_id == "stage8" else "")
            info["stage8_max_steps"] = self._effective_max_steps() if self.stage_id == "stage8" else 0
            info["stage8_obstacle_count"] = self.stage8_obstacle_count if self.stage_id == "stage8" else 0
            info["stage8_enemy_fire_rate_scale"] = self.stage8_enemy_fire_rate_scale if self.stage_id == "stage8" else 0.0
            info["stage8_first_shot_delay_steps"] = self.stage8_first_shot_delay_steps if self.stage_id == "stage8" else 0
            info["stage8_max_active_enemy_projectiles"] = self.stage8_max_active_enemy_projectiles if self.stage_id == "stage8" else 0
            info["stage8_enemy_can_move"] = 1 if (self.stage_id == "stage8" and self.stage8_enemy_can_move) else 0
            info["stage8_enemy_can_shoot"] = 1 if (self.stage_id == "stage8" and self.stage8_enemy_can_shoot) else 0
            info["stage8_enemy_waves"] = self.stage8_enemy_waves if self.stage_id == "stage8" else 0
            info["stage8_current_wave"] = self.stage8_current_wave if self.stage_id == "stage8" else 0
            info["stage8_waves_cleared"] = min(int(info.get("kills", 0)), self.stage8_enemy_waves) if self.stage_id == "stage8" else 0
            info["stage8_wave_clear_ratio"] = (min(int(info.get("kills", 0)), self.stage8_enemy_waves) / max(1, self.stage8_enemy_waves)) if self.stage_id == "stage8" else 0.0
            info["stage8_consecutive_empty_ammo_steps"] = self.stage8_consecutive_empty_ammo_steps if self.stage_id == "stage8" else 0
            info["stage8_empty_ammo_timeout"] = 1 if (self.stage_id == "stage8" and self.stage8_empty_ammo_timeout_triggered) else 0
            info["stage8_low_ammo_decision_steps"] = self.stage8_low_ammo_decision_steps if self.stage_id == "stage8" else 0
            info["stage8_low_ammo_decision_active"] = 1 if (self.stage_id == "stage8" and self.stage8_low_ammo_decision_active) else 0
            info["stage8_obstacle_variant_id"] = self.episode_obstacle_variant_id if self.stage_id == "stage8" else -1
            info["stage8_spawn_index"] = self.episode_spawn_index if self.stage_id == "stage8" else -1
            info["stage8_spawn_pair_id"] = self.episode_spawn_pair_id if self.stage_id == "stage8" else -1
            info["stage8_last_enemy_respawn_index"] = self.stage8_last_enemy_respawn_index if self.stage_id == "stage8" else -1
            info["stage8_controlled_respawn_used"] = 1 if (self.stage_id == "stage8" and self.stage8_controlled_respawn_used) else 0
            info["enemy_projectile_speed"] = self.enemy.projectile_speed

            if (
                self.stage.end_episode_on_success
                and stage_success_now
                and self.learner.is_alive
                and self.enemy.is_alive
            ):
                info["outcome"] = "stage_success"
            elif not self.learner.is_alive:
                info["outcome"] = "learner_died"
            elif self.stage_id == "stage8" and self.stage8_empty_ammo_timeout_triggered:
                info["outcome"] = "stage8_empty_ammo_timeout"
            elif not self.enemy.is_alive:
                info["outcome"] = "enemy_killed"
            else:
                info["outcome"] = "timeout"

        if self.render_mode == "human":
            self.render()

        return self._stacked_obs(), reward, terminated, truncated, info

    def render(self):
        self._draw_frame()

    def consume_save_request(self) -> bool:
        requested = self._save_requested
        self._save_requested = False
        return requested

    def _stacked_obs(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return np.concatenate(list(self._frames), axis=0).astype(np.float32)

    def set_stage4_projectile_speed_scale(self, scale: float) -> None:
        """Set the Stage 4 training-only projectile-speed scale.

        This is intentionally ignored in validation mode so evaluation keeps the
        official projectile speed.
        """
        self.stage4_projectile_speed_scale = float(max(0.1, min(1.0, scale)))
        if self.enemy is not None:
            self.enemy.projectile_speed = self._episode_enemy_projectile_speed()

    def set_stage4_projectile_pressure(
        self,
        fire_rate_scale: float = 1.0,
        first_shot_delay_steps: int = 0,
        max_active_enemy_projectiles: int = 99,
    ) -> None:
        """Set Stage 4 training-only projectile pressure.

        Validation ignores these values and keeps the official enemy pressure.
        """
        self.stage4_enemy_fire_rate_scale = float(max(0.20, min(1.0, fire_rate_scale)))
        self.stage4_first_shot_delay_steps = int(max(0, first_shot_delay_steps))
        self.stage4_max_active_enemy_projectiles = int(max(1, max_active_enemy_projectiles))
        if self.enemy is not None:
            self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self._apply_stage4_pressure_settings_to_world()

    def set_stage4_curriculum_level(self, level: int, total_levels: int) -> None:
        self.stage4_curriculum_level = int(max(0, level))
        self.stage4_curriculum_total_levels = int(max(0, total_levels))

    def _effective_stage4_curriculum_level(self) -> int:
        if self.stage_id == "stage4" and not self.validation_mode:
            return self.stage4_curriculum_level
        # Zero means official fixed validation pressure rather than a training
        # curriculum level.
        return 0

    def _effective_stage4_projectile_speed_scale(self) -> float:
        if self.stage_id == "stage4" and not self.validation_mode:
            return self.stage4_projectile_speed_scale
        return 1.0

    def _effective_stage4_enemy_fire_rate_scale(self) -> float:
        if self.stage_id == "stage4" and not self.validation_mode:
            return self.stage4_enemy_fire_rate_scale
        return 1.0

    def _effective_stage4_first_shot_delay_steps(self) -> int:
        if self.stage_id == "stage4" and not self.validation_mode:
            return self.stage4_first_shot_delay_steps
        return 0

    def _effective_stage4_max_active_enemy_projectiles(self) -> int:
        if self.stage_id == "stage4" and not self.validation_mode:
            return self.stage4_max_active_enemy_projectiles
        return 99

    def set_stage5_curriculum_phase(
        self,
        level: int = 1,
        total_levels: int = 4,
        phase_index: int = 1,
        enemy_can_move: bool = False,
        enemy_can_shoot: bool = False,
        fire_rate_scale: float = 0.0,
        first_shot_delay_steps: int = 9999,
        max_active_enemy_projectiles: int = 0,
    ) -> None:
        self.stage5_curriculum_level = int(max(1, level))
        self.stage5_curriculum_total_levels = int(max(1, total_levels))
        self.stage5_curriculum_phase_index = int(max(1, min(3, phase_index)))
        self.stage5_enemy_can_move = bool(enemy_can_move)
        self.stage5_enemy_can_shoot = bool(enemy_can_shoot)
        self.stage5_enemy_fire_rate_scale = float(max(0.0, min(1.0, fire_rate_scale)))
        self.stage5_first_shot_delay_steps = int(max(0, first_shot_delay_steps))
        self.stage5_max_active_enemy_projectiles = int(max(0, max_active_enemy_projectiles))
        if self.enemy is not None:
            self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self._apply_stage5_settings_to_world()
        self._apply_stage5_settings_to_reward_manager()

    def _apply_stage5_settings_to_world(self) -> None:
        if self.world is None or self.stage_id != "stage5":
            return
        self.world.stage5_enemy_can_move = bool(self.stage5_enemy_can_move)
        self.world.stage5_enemy_can_shoot = bool(self.stage5_enemy_can_shoot)
        self.world.stage5_first_shot_delay_steps = int(self.stage5_first_shot_delay_steps)
        self.world.stage5_max_active_enemy_projectiles = int(self.stage5_max_active_enemy_projectiles)

    def _apply_stage5_settings_to_reward_manager(self) -> None:
        if self.stage_id != "stage5" or self.reward_manager is None:
            return
        self.reward_manager.set_stage5_curriculum_phase(
            self.stage5_curriculum_level,
            self.stage5_curriculum_total_levels,
            self.stage5_curriculum_phase_index,
        )

    def set_stage6_curriculum_phase(
        self,
        level: int = 1,
        total_levels: int = 4,
        phase_index: int = 1,
        enemy_can_move: bool = True,
        enemy_can_shoot: bool = True,
        fire_rate_scale: float = 0.55,
        first_shot_delay_steps: int = 90,
        max_active_enemy_projectiles: int = 1,
    ) -> None:
        self.stage6_curriculum_level = int(max(1, level))
        self.stage6_curriculum_total_levels = int(max(1, total_levels))
        self.stage6_curriculum_phase_index = int(max(1, min(4, phase_index)))
        self.stage6_enemy_can_move = bool(enemy_can_move)
        self.stage6_enemy_can_shoot = bool(enemy_can_shoot)
        self.stage6_enemy_fire_rate_scale = float(max(0.0, min(1.0, fire_rate_scale)))
        self.stage6_first_shot_delay_steps = int(max(0, first_shot_delay_steps))
        self.stage6_max_active_enemy_projectiles = int(max(0, max_active_enemy_projectiles))
        if self.enemy is not None:
            self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self._apply_stage6_settings_to_world()
        self._apply_stage6_settings_to_reward_manager()

    def _apply_stage6_settings_to_world(self) -> None:
        if self.world is None or self.stage_id != "stage6":
            return
        self.world.stage6_enemy_can_move = bool(self.stage6_enemy_can_move)
        self.world.stage6_enemy_can_shoot = bool(self.stage6_enemy_can_shoot)
        self.world.stage6_first_shot_delay_steps = int(self.stage6_first_shot_delay_steps)
        self.world.stage6_max_active_enemy_projectiles = int(self.stage6_max_active_enemy_projectiles)

    def _apply_stage6_settings_to_reward_manager(self) -> None:
        if self.stage_id != "stage6" or self.reward_manager is None:
            return
        self.reward_manager.set_stage6_curriculum_phase(
            self.stage6_curriculum_level,
            self.stage6_curriculum_total_levels,
            self.stage6_curriculum_phase_index,
        )

    def set_stage7_curriculum_phase(
        self,
        level: int = 1,
        total_levels: int = 5,
        phase_index: int = 1,
        max_steps: int | None = None,
        obstacle_count: int = 1,
        enemy_can_move: bool = False,
        enemy_can_shoot: bool = True,
        fire_rate_scale: float = 0.55,
        first_shot_delay_steps: int = 120,
        max_active_enemy_projectiles: int = 1,
    ) -> None:
        self.stage7_curriculum_level = int(max(1, level))
        self.stage7_curriculum_total_levels = int(max(1, total_levels))
        self.stage7_curriculum_phase_index = int(max(1, min(5, phase_index)))
        if max_steps is not None:
            self.stage7_max_steps = int(max(1, max_steps))
        self.stage7_obstacle_count = int(max(1, min(4, obstacle_count)))
        self.stage7_enemy_can_move = bool(enemy_can_move)
        self.stage7_enemy_can_shoot = bool(enemy_can_shoot)
        self.stage7_enemy_fire_rate_scale = float(max(0.0, min(1.0, fire_rate_scale)))
        self.stage7_first_shot_delay_steps = int(max(0, first_shot_delay_steps))
        self.stage7_max_active_enemy_projectiles = int(max(0, max_active_enemy_projectiles))
        if self.enemy is not None:
            self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self._apply_stage7_settings_to_world()
        self._apply_stage7_settings_to_reward_manager()

    def _apply_stage7_settings_to_world(self) -> None:
        if self.world is None or self.stage_id != "stage7":
            return
        self.world.stage7_enemy_can_move = bool(self.stage7_enemy_can_move)
        self.world.stage7_enemy_can_shoot = bool(self.stage7_enemy_can_shoot)
        self.world.stage7_first_shot_delay_steps = int(self.stage7_first_shot_delay_steps)
        self.world.stage7_max_active_enemy_projectiles = int(self.stage7_max_active_enemy_projectiles)
        self.world.stage7_obstacle_count = int(self.stage7_obstacle_count)

    def _apply_stage7_settings_to_reward_manager(self) -> None:
        if self.stage_id != "stage7" or self.reward_manager is None:
            return
        self.reward_manager.set_stage7_curriculum_phase(
            self.stage7_curriculum_level,
            self.stage7_curriculum_total_levels,
            self.stage7_curriculum_phase_index,
        )

    def set_stage8_curriculum_phase(
        self,
        level: int = 1,
        total_levels: int = 5,
        phase_index: int = 1,
        max_steps: int | None = None,
        obstacle_count: int = 0,
        enemy_can_move: bool = False,
        enemy_can_shoot: bool = False,
        fire_rate_scale: float = 0.0,
        first_shot_delay_steps: int = 9999,
        max_active_enemy_projectiles: int = 0,
        enemy_waves: int = STAGE8_DEFAULT_ENEMY_WAVES,
    ) -> None:
        self.stage8_curriculum_level = int(max(1, level))
        self.stage8_curriculum_total_levels = int(max(1, total_levels))
        self.stage8_curriculum_phase_index = int(max(1, min(6, phase_index)))
        if max_steps is not None:
            self.stage8_max_steps = int(max(1, max_steps))
        self.stage8_obstacle_count = int(max(0, min(4, obstacle_count)))
        self.stage8_enemy_can_move = bool(enemy_can_move)
        self.stage8_enemy_can_shoot = bool(enemy_can_shoot)
        self.stage8_enemy_fire_rate_scale = float(max(0.0, min(1.0, fire_rate_scale)))
        self.stage8_first_shot_delay_steps = int(max(0, first_shot_delay_steps))
        self.stage8_max_active_enemy_projectiles = int(max(0, max_active_enemy_projectiles))
        self.stage8_enemy_waves = int(max(1, enemy_waves))
        if self.enemy is not None:
            self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self._apply_stage8_settings_to_world()
        self._apply_stage8_settings_to_reward_manager()

    def _apply_stage8_settings_to_world(self) -> None:
        if self.world is None or self.stage_id != "stage8":
            return
        self.world.stage8_enemy_can_move = bool(self.stage8_enemy_can_move)
        self.world.stage8_enemy_can_shoot = bool(self.stage8_enemy_can_shoot)
        self.world.stage8_first_shot_delay_steps = int(self.stage8_first_shot_delay_steps)
        self.world.stage8_max_active_enemy_projectiles = int(self.stage8_max_active_enemy_projectiles)
        self.world.stage8_obstacle_count = int(self.stage8_obstacle_count)
        self.world.stage8_curriculum_phase_index = int(self.stage8_curriculum_phase_index)
        self.world.stage8_enemy_waves = int(self.stage8_enemy_waves)

    def _apply_stage8_settings_to_reward_manager(self) -> None:
        if self.stage_id != "stage8" or self.reward_manager is None:
            return
        self.reward_manager.set_stage8_curriculum_phase(
            self.stage8_curriculum_level,
            self.stage8_curriculum_total_levels,
            self.stage8_curriculum_phase_index,
            enemy_waves=self.stage8_enemy_waves,
        )

    def _apply_stage8_reload_pressure(self) -> None:
        """Make Stage 8 test reload management without changing the weapon model.

        Both combatants keep the normal Glock magazine and damage model.  The
        learner starts full, but the episode contains several sequential enemies
        while reload charges are limited, so reloading becomes useful naturally
        during extended combat rather than because of an artificial 4/9 setup.
        """
        if self.stage_id != "stage8":
            return

        for agent in (self.learner, self.enemy):
            if agent is None:
                continue
            agent.max_ammo = int(STAGE8_MAX_AMMO)
            agent.max_reload_charges = STAGE8_RELOAD_CHARGES
            agent.reload_charges = STAGE8_RELOAD_CHARGES
            agent.ammo = max(1, min(int(STAGE8_STARTING_AMMO), int(agent.max_ammo)))
            agent.reloading = False
            agent.reload_timer = 0.0
            agent.fire_timer = 0.0

        if self.world is not None:
            self.world.stage8_max_ammo = STAGE8_MAX_AMMO
            self.world.stage8_starting_ammo = STAGE8_STARTING_AMMO
            self.world.stage8_reload_charges = STAGE8_RELOAD_CHARGES

    def _episode_enemy_projectile_speed(self) -> float:
        return self.stage.enemy_projectile_speed * self._effective_stage4_projectile_speed_scale()

    def _episode_enemy_fire_rate(self) -> float:
        base_fire_rate = self.stage.enemy_fire_rate or GLOCK_FIRE_RATE
        if self.stage_id == "stage5":
            if not self.stage5_enemy_can_shoot or self.stage5_enemy_fire_rate_scale <= 0.0:
                return base_fire_rate
            return max(0.5, base_fire_rate * self.stage5_enemy_fire_rate_scale)
        if self.stage_id == "stage6":
            if not self.stage6_enemy_can_shoot or self.stage6_enemy_fire_rate_scale <= 0.0:
                return base_fire_rate
            return max(0.5, base_fire_rate * self.stage6_enemy_fire_rate_scale)
        if self.stage_id == "stage7":
            if not self.stage7_enemy_can_shoot or self.stage7_enemy_fire_rate_scale <= 0.0:
                return base_fire_rate
            return max(0.5, base_fire_rate * self.stage7_enemy_fire_rate_scale)
        if self.stage_id == "stage8":
            if not self.stage8_enemy_can_shoot or self.stage8_enemy_fire_rate_scale <= 0.0:
                return base_fire_rate
            return max(0.5, base_fire_rate * self.stage8_enemy_fire_rate_scale)
        return max(0.5, base_fire_rate * self._effective_stage4_enemy_fire_rate_scale())

    def _apply_stage4_pressure_settings_to_world(self) -> None:
        if self.world is None:
            return
        self.world.stage4_first_shot_delay_steps = self._effective_stage4_first_shot_delay_steps()
        self.world.stage4_max_active_enemy_projectiles = self._effective_stage4_max_active_enemy_projectiles()

    def _episode_enemy_move_speed(self) -> float:
        # CS2D-style consistency: movement speed is a game constant, not a
        # curriculum/profile parameter. Stages can still disable movement, but
        # when a character moves, learner and stock bot use the same speed.
        return PLAYER_MOVE_SPEED_PX_PER_SEC

    def _mask_enemy_command(self, command: AgentCommand) -> AgentCommand:
        if not self.stage.enemy_can_move:
            command.move_idx = 0
            command.strafe_idx = 0
            command.use_world_movement = False
            command.world_move_x = 0.0
            command.world_move_y = 0.0

        if not self.stage.enemy_can_shoot:
            command.shoot = False
            command.reload = False

        if self.stage_id == "stage5":
            if not self.stage5_enemy_can_move:
                command.move_idx = 0
                command.strafe_idx = 0
                command.use_world_movement = False
                command.world_move_x = 0.0
                command.world_move_y = 0.0
            if not self.stage5_enemy_can_shoot:
                command.shoot = False
                command.reload = False
            else:
                active_projectiles = sum(
                    1
                    for projectile in getattr(self.world, "projectiles", [])
                    if getattr(projectile, "owner_id", None) == self.enemy.agent_id
                ) if self.world is not None and self.enemy is not None else 0
                if self._step_count < self.stage5_first_shot_delay_steps or active_projectiles >= self.stage5_max_active_enemy_projectiles:
                    command.shoot = False

        if self.stage_id == "stage6":
            if not self.stage6_enemy_can_move:
                command.move_idx = 0
                command.strafe_idx = 0
                command.use_world_movement = False
                command.world_move_x = 0.0
                command.world_move_y = 0.0
            if not self.stage6_enemy_can_shoot:
                command.shoot = False
                command.reload = False
            else:
                active_projectiles = sum(
                    1
                    for projectile in getattr(self.world, "projectiles", [])
                    if getattr(projectile, "owner_id", None) == self.enemy.agent_id
                ) if self.world is not None and self.enemy is not None else 0
                if self._step_count < self.stage6_first_shot_delay_steps or active_projectiles >= self.stage6_max_active_enemy_projectiles:
                    command.shoot = False

        if self.stage_id == "stage7":
            if not self.stage7_enemy_can_move:
                command.move_idx = 0
                command.strafe_idx = 0
                command.use_world_movement = False
                command.world_move_x = 0.0
                command.world_move_y = 0.0
            if not self.stage7_enemy_can_shoot:
                command.shoot = False
                command.reload = False
            else:
                active_projectiles = sum(
                    1
                    for projectile in getattr(self.world, "projectiles", [])
                    if getattr(projectile, "owner_id", None) == self.enemy.agent_id
                ) if self.world is not None and self.enemy is not None else 0
                if self._step_count < self.stage7_first_shot_delay_steps or active_projectiles >= self.stage7_max_active_enemy_projectiles:
                    command.shoot = False

        if self.stage_id == "stage8":
            if not self.stage8_enemy_can_move:
                command.move_idx = 0
                command.strafe_idx = 0
                command.use_world_movement = False
                command.world_move_x = 0.0
                command.world_move_y = 0.0
            if not self.stage8_enemy_can_shoot:
                command.shoot = False
                # Allow reload even when shooting is disabled: 8A should still let the
                # enemy manage resources if needed, but it will not fire.
            else:
                active_projectiles = sum(
                    1
                    for projectile in getattr(self.world, "projectiles", [])
                    if getattr(projectile, "owner_id", None) == self.enemy.agent_id
                ) if self.world is not None and self.enemy is not None else 0
                if self._step_count < self.stage8_first_shot_delay_steps or active_projectiles >= self.stage8_max_active_enemy_projectiles:
                    command.shoot = False

        return command


    def _stage8_respawn_enemy_for_next_wave(self) -> None:
        if self.world is None or self.learner is None or self.enemy is None:
            return
        if not self.layout.spawns:
            return

        spawn_count = len(self.layout.spawns)
        phase_index = int(getattr(self, "stage8_curriculum_phase_index", 1) or 1)
        controlled_respawn = phase_index >= 5
        chosen_index = int(self.np_random.integers(0, spawn_count))

        # 8E is reload-endurance validation, not a hard obstacle rediscovery task.
        # Prefer a small fixed set of valid, usually visible respawn points so the
        # deterministic policy has repeated opportunities to practice reload timing.
        # 8F keeps the harder rotating spawn-pair behavior for integrated testing.
        fixed_8e_points = []
        if phase_index == 5:
            fixed_8e_points = [
                Vec2(850.0, 220.0),
                Vec2(850.0, 580.0),
                Vec2(950.0, 400.0),
                Vec2(760.0, 400.0),
                Vec2(900.0, 300.0),
                Vec2(900.0, 500.0),
                Vec2(350.0, 220.0),
                Vec2(350.0, 580.0),
                Vec2(250.0, 400.0),
                Vec2(440.0, 400.0),
            ]

        chosen_x = None
        chosen_y = None
        chosen_heading = None
        if fixed_8e_points:
            wave_offset = max(1, int(getattr(self, "stage8_current_wave", 1)) - 1)
            start = int((self.episode_spawn_index + wave_offset) % len(fixed_8e_points))
            ordered_points = [fixed_8e_points[(start + offset) % len(fixed_8e_points)] for offset in range(len(fixed_8e_points))]

            # First pass: direct LOS and safe distance. This makes 8E primarily test
            # reload endurance instead of repeated obstacle re-acquisition.
            for idx, point in enumerate(ordered_points):
                dx = point.x - self.learner.position.x
                dy = point.y - self.learner.position.y
                if math.sqrt(dx * dx + dy * dy) >= 220.0 and self.world.has_line_of_sight(self.learner.position, point):
                    chosen_x = point.x
                    chosen_y = point.y
                    chosen_index = idx
                    chosen_heading = math.atan2(self.learner.position.y - chosen_y, self.learner.position.x - chosen_x)
                    break

            # Fallback: still use a fixed valid point with safe distance, even if the
            # current learner position temporarily blocks LOS through the obstacle.
            if chosen_x is None:
                for idx, point in enumerate(ordered_points):
                    dx = point.x - self.learner.position.x
                    dy = point.y - self.learner.position.y
                    if math.sqrt(dx * dx + dy * dy) >= 220.0:
                        chosen_x = point.x
                        chosen_y = point.y
                        chosen_index = idx
                        chosen_heading = math.atan2(self.learner.position.y - chosen_y, self.learner.position.x - chosen_x)
                        break

        if chosen_x is None:
            if controlled_respawn:
                # 8F and fallback: controlled rotating set of valid enemy respawns.
                # This gives variation across waves without fully randomizing the map.
                base_index = self.episode_spawn_index if 0 <= self.episode_spawn_index < spawn_count else chosen_index
                wave_offset = max(1, int(getattr(self, "stage8_current_wave", 1)) - 1)
                candidate_indices = [int((base_index + wave_offset + offset) % spawn_count) for offset in range(spawn_count)]
            else:
                candidate_indices = [int(self.np_random.integers(0, spawn_count)) for _ in range(12)]

            chosen_spawn = self.layout.spawns[candidate_indices[0]]
            chosen_index = candidate_indices[0]
            for candidate_index in candidate_indices:
                candidate = self.layout.spawns[candidate_index]
                dx = candidate.enemy_x - self.learner.position.x
                dy = candidate.enemy_y - self.learner.position.y
                if math.sqrt(dx * dx + dy * dy) >= 220.0:
                    chosen_spawn = candidate
                    chosen_index = candidate_index
                    break
            chosen_x = float(chosen_spawn.enemy_x)
            chosen_y = float(chosen_spawn.enemy_y)
            chosen_heading = float(chosen_spawn.enemy_heading)

        self.stage8_last_enemy_respawn_index = int(chosen_index)
        self.stage8_controlled_respawn_used = bool(controlled_respawn)

        self.enemy.reset_state()
        self.enemy.can_move = bool(self.stage8_enemy_can_move)
        self.enemy.move_speed = self._episode_enemy_move_speed()
        self.enemy.fire_rate = self._episode_enemy_fire_rate()
        self.enemy.max_ammo = int(STAGE8_MAX_AMMO)
        self.enemy.ammo = int(STAGE8_STARTING_AMMO)
        self.enemy.max_reload_charges = STAGE8_RELOAD_CHARGES
        self.enemy.reload_charges = STAGE8_RELOAD_CHARGES
        self.enemy.position.x = float(chosen_x)
        self.enemy.position.y = float(chosen_y)
        if chosen_heading is None:
            chosen_heading = math.atan2(self.learner.position.y - self.enemy.position.y, self.learner.position.x - self.enemy.position.x)
        self.enemy.heading = float(chosen_heading)

        # Start each wave clean.  The learner keeps HP, ammo, reload timer, and
        # reload charges, but old bullets are removed so the next wave is not
        # decided by projectiles fired at a dead opponent.
        self.world.projectiles.clear()
        self.world._visibility_state.clear()
        if self.enemy_bot is not None and hasattr(self.enemy_bot, "reset_memory"):
            self.enemy_bot.reset_memory()
        if self.reward_manager is not None:
            self.reward_manager.prev_distance = self.learner.position.distance_to(self.enemy.position)
            self.reward_manager.prev_angle_error = self.reward_manager._angle_error_to_enemy(self.learner, self.enemy)
            self.reward_manager.prev_line_of_sight = self.world.has_line_of_sight(self.learner.position, self.enemy.position)
            self.reward_manager.prev_clear_shot = (
                self.reward_manager._stage2_shot_ray_hits_enemy(self.learner, self.enemy, self.world)
                if self.reward_manager.prev_line_of_sight else False
            )

    def _advance_stage8_wave_if_needed(self) -> bool:
        if self.stage_id != "stage8" or self.learner is None or self.enemy is None:
            return False
        if self.reward_manager is None:
            return False
        if self.enemy.is_alive or not self.learner.is_alive:
            return False
        waves_cleared = min(int(getattr(self.reward_manager, "kills", 0)), int(self.stage8_enemy_waves))
        self.stage8_waves_completed = waves_cleared
        if waves_cleared >= int(self.stage8_enemy_waves):
            return False
        self.stage8_current_wave = waves_cleared + 1
        self._stage8_respawn_enemy_for_next_wave()
        return True

    def _reload_charge_text(self, agent: Agent) -> str:
        charges = getattr(agent, "reload_charges", None)
        max_charges = getattr(agent, "max_reload_charges", None)
        if charges is None or max_charges is None:
            return "unlimited"
        return f"{int(charges)}/{int(max_charges)}"

    def _draw_frame(self) -> None:
        if self.world is None or self.learner is None or self.enemy is None:
            return

        self._ensure_renderer()

        learner_reload_text = self._reload_charge_text(self.learner)
        enemy_reload_text = self._reload_charge_text(self.enemy)

        hud_lines = [
            f"Stage: {self.stage.name}",
            f"Learner HP {self.learner.health:.0f} | Ammo {self.learner.ammo}/{self.learner.max_ammo} | Reloads {learner_reload_text}",
            f"Enemy HP {self.enemy.health:.0f} | Ammo {self.enemy.ammo}/{self.enemy.max_ammo} | Reloads {enemy_reload_text}",
            f"Step {self._step_count}/{self._effective_max_steps()}",
            f"Mode: {'validation' if self.validation_mode else 'training'} | Frame stack: {self.frame_stack}",
            f"Observation schema: {self.observation_stage_id}",
            "P = pause/resume | F3 = rays | F4 = A* path | F5 = save checkpoint | ESC = stop current mode",
        ]

        self.renderer.draw_world(
            self.world,
            hud_lines=hud_lines,
            title="PPO shooter environment",
        )

    def _ensure_renderer(self) -> None:
        if self.renderer is None:
            screen = pygame.display.get_surface()

            self.renderer = Renderer(
                width=self.layout.width,
                height=self.layout.height,
                caption=f"RL Shooter - {self.stage.name}",
                target_fps=120,
                screen=screen,
                show_raycast_debug=self.show_raycast_debug,
            )

            self.renderer.reset_runtime_flags()

    def _place_agents(self) -> None:
        if self.world is None or self.learner is None or self.enemy is None:
            return

        if self.stage_id in {"stage1", "stage2"}:
            self._place_stage1_agents()
            return

        spawn_index = int(self.np_random.integers(0, len(self.layout.spawns)))
        spawn = self.layout.spawns[spawn_index]
        spawn_pair_ids = getattr(self.layout, "spawn_pair_ids", None) or []
        self.episode_spawn_index = spawn_index
        self.episode_spawn_pair_id = int(spawn_pair_ids[spawn_index]) if spawn_index < len(spawn_pair_ids) else spawn_index
        self.episode_obstacle_variant_id = int(getattr(self.layout, "obstacle_variant_id", -1))

        self.learner.position.x = spawn.learner_x
        self.learner.position.y = spawn.learner_y
        self.learner.heading = spawn.learner_heading

        self.enemy.position.x = spawn.enemy_x
        self.enemy.position.y = spawn.enemy_y
        self.enemy.heading = spawn.enemy_heading

    def _place_stage1_agents(self) -> None:
        if self.world is None or self.learner is None or self.enemy is None:
            return

        center_x = self.world.width * 0.5
        center_y = self.world.height * 0.5

        self.learner.position.x = center_x
        self.learner.position.y = center_y
        self.learner.heading = float(self.np_random.uniform(-math.pi, math.pi))

        min_radius, max_radius = (235.0, 345.0) if self.validation_mode else (225.0, 335.0)
        radius = float(self.np_random.uniform(min_radius, max_radius))
        angle = float(self.np_random.uniform(-math.pi, math.pi))

        self.enemy.position.x = center_x + math.cos(angle) * radius
        self.enemy.position.y = center_y + math.sin(angle) * radius
        self.enemy.heading = wrap_angle(
            angle + math.pi + float(self.np_random.uniform(-0.7, 0.7))
        )

    def close(self):
        self.renderer = None