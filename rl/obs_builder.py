from __future__ import annotations

import math

import numpy as np

from config.stages import normalize_stage_id
from game import detection
from game.agent import Agent, wrap_angle
from game.world import World
from training.skill_table import FULL_SKILL_ORDER, build_skill_table, skill_table_feature_vector


class ObsBuilder:


    RAY_COUNT = 32
    MAX_RAY_LENGTH = 380.0
    RAY_STEP = 8.0
    HIT_TYPE_COUNT = detection.HIT_TYPE_COUNT

    RAY_FEATURES_PER_RAY = 1 + HIT_TYPE_COUNT
    SELF_FEATURES = 7
    ENEMY_FEATURES = 10
    PROJECTILE_FEATURES = 8
    WALL_FEATURES = 1
    PICKUP_FEATURES = 6
    LOS_FEATURES = 1
    SKILL_FEATURES = len(FULL_SKILL_ORDER) * 2

    OBS_DIM = (
        RAY_COUNT * RAY_FEATURES_PER_RAY
        + SELF_FEATURES
        + ENEMY_FEATURES
        + PROJECTILE_FEATURES
        + WALL_FEATURES
        + PICKUP_FEATURES
        + LOS_FEATURES
        + SKILL_FEATURES
    )

    
    STAGE1_OBS_DIM = 15

    def __init__(self):
        self.last_debug_rays: list[detection.DetectionDebugLine] = []

    @classmethod
    def obs_dim_for_stage(cls, stage_id: str) -> int:
        
        return cls.STAGE1_OBS_DIM if normalize_stage_id(stage_id) in {"stage1", "stage2"} else cls.OBS_DIM

    def build(
        self,
        observer: Agent,
        target: Agent,
        world: World,
        skill_table: dict | None = None,
        stage_id: str | None = None,
    ) -> np.ndarray:
        if stage_id is not None and normalize_stage_id(stage_id) in {"stage1", "stage2"}:
            return self._build_stage1(observer, target, world)
        return self._build_full(observer, target, world, skill_table, normalize_stage_id(stage_id or "stage1"))

    def _build_stage1(self, observer: Agent, target: Agent, world: World) -> np.ndarray:
        
        self.last_debug_rays = []
        dx = target.position.x - observer.position.x
        dy = target.position.y - observer.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        relative_angle = wrap_angle(math.atan2(dy, dx) - observer.heading)
        abs_error = abs(relative_angle)
        velocity_scale = max(1.0, target.move_speed + observer.move_speed)
        rel_vx = target.velocity.x - observer.velocity.x
        rel_vy = target.velocity.y - observer.velocity.y
        visible = target.is_alive and world.has_line_of_sight(observer.position, target.position)
        aligned = visible and abs_error < 0.12

        features = [
            math.sin(relative_angle),
            math.cos(relative_angle),
            self._clip_signed(relative_angle / math.pi),
            self._clip_signed(abs_error / math.pi),
            self._clip_signed(dx / self.MAX_RAY_LENGTH),
            self._clip_signed(dy / self.MAX_RAY_LENGTH),
            self._norm_distance(distance, self.MAX_RAY_LENGTH),
            self._clip_signed(rel_vx / velocity_scale),
            self._clip_signed(rel_vy / velocity_scale),
            math.sin(observer.heading),
            math.cos(observer.heading),
            math.sin(target.heading),
            math.cos(target.heading),
            1.0 if visible else 0.0,
            1.0 if aligned else 0.0,
        ]

        obs = np.asarray(features, dtype=np.float32)
        if obs.shape != (self.STAGE1_OBS_DIM,):
            raise RuntimeError(f"Stage 1 observation dimension mismatch: got {obs.shape}, expected {(self.STAGE1_OBS_DIM,)}")
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)

    def _build_full(
        self,
        observer: Agent,
        target: Agent,
        world: World,
        skill_table: dict | None = None,
        stage_id: str = "stage1",
    ) -> np.ndarray:
        features: list[float] = []
        self.last_debug_rays = []

        hits, debug_lines = detection.trace_many_rays(
            world=world,
            origin=observer.position,
            heading=observer.heading,
            ray_count=self.RAY_COUNT,
            max_length=self.MAX_RAY_LENGTH,
            step_size=self.RAY_STEP,
            target_agent_id=target.agent_id,
            observer_agent_id=observer.agent_id,
            include_pickups=True,
            include_projectiles=True,
        )
        self.last_debug_rays = debug_lines

        for hit in hits:
            features.append(self._norm_distance(hit.distance, self.MAX_RAY_LENGTH))
            one_hot = [0.0] * self.HIT_TYPE_COUNT
            if 0 <= hit.hit_type < self.HIT_TYPE_COUNT:
                one_hot[hit.hit_type] = 1.0
            features.extend(one_hot)

        features.extend(self._self_features(observer))
        features.extend(self._enemy_features(observer, target, world))
        features.extend(self._projectile_features(observer, world))
        features.append(self._norm_distance(detection.nearest_wall_distance(world, observer.position, self.MAX_RAY_LENGTH), self.MAX_RAY_LENGTH))
        if stage_id == "stage8":
            features.extend(self._stage8_resource_features(observer))
        else:
            features.extend(self._pickup_features(observer, world))
        features.append(1.0 if self._enemy_visible(observer, target, world) else 0.0)

        if skill_table is None:
            skill_table = build_skill_table("stage1")
        features.extend(skill_table_feature_vector(skill_table))

        obs = np.asarray(features, dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Observation dimension mismatch: got {obs.shape}, expected {(self.OBS_DIM,)}")
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _self_features(self, observer: Agent) -> list[float]:
        speed_scale = max(1.0, observer.move_speed)
        return [
            observer.health_ratio,
            observer.ammo_ratio,
            1.0 if observer.reloading else 0.0,
            observer.reload_progress,
            observer.fire_ready,
            self._clip_signed(observer.velocity.x / speed_scale),
            self._clip_signed(observer.velocity.y / speed_scale),
        ]

    def _enemy_features(self, observer: Agent, target: Agent, world: World) -> list[float]:
        visible = self._enemy_visible(observer, target, world)
        if not visible:
            return [0.0] * self.ENEMY_FEATURES

        dx = target.position.x - observer.position.x
        dy = target.position.y - observer.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        angle = wrap_angle(math.atan2(dy, dx) - observer.heading)
        rel_vx = target.velocity.x - observer.velocity.x
        rel_vy = target.velocity.y - observer.velocity.y
        velocity_scale = max(1.0, observer.move_speed + target.move_speed)

        return [
            1.0,
            target.health_ratio,
            target.ammo_ratio,
            1.0 if target.reloading else 0.0,
            self._norm_distance(distance, self.MAX_RAY_LENGTH),
            math.sin(angle),
            math.cos(angle),
            self._clip_signed(rel_vx / velocity_scale),
            self._clip_signed(rel_vy / velocity_scale),
            target.fire_ready,
        ]

    def _projectile_features(self, observer: Agent, world: World) -> list[float]:
        threat = detection.projectile_threat(world, observer, max_distance=self.MAX_RAY_LENGTH)
        if not threat.exists:
            return [0.0] * self.PROJECTILE_FEATURES
        return [
            1.0,
            self._norm_distance(threat.distance, self.MAX_RAY_LENGTH),
            math.sin(threat.relative_angle),
            math.cos(threat.relative_angle),
            math.sin(threat.direction_angle),
            math.cos(threat.direction_angle),
            1.0 if threat.danger else 0.0,
            threat.time_to_impact,
        ]

    def _stage8_resource_features(self, observer: Agent) -> list[float]:
        charges = getattr(observer, "reload_charges", None)
        max_charges = getattr(observer, "max_reload_charges", None)
        if charges is None or max_charges in (None, 0):
            charge_ratio = 1.0
            charges_norm = 1.0
        else:
            charge_ratio = max(0.0, min(1.0, float(charges) / max(1.0, float(max_charges))))
            charges_norm = max(0.0, min(1.0, float(charges) / 3.0))
        target = 1.0 / 3.0
        ammo_ratio = observer.ammo_ratio
        return [
            charge_ratio,
            charges_norm,
            1.0 if ammo_ratio <= target else 0.0,
            1.0 if ammo_ratio <= 0.16 else 0.0,
            self._clip_signed((ammo_ratio - target) / max(1e-6, target)),
            1.0 if observer.reloading else 0.0,
        ]

    def _pickup_features(self, observer: Agent, world: World) -> list[float]:
        visible = [pickup for pickup in detection.visible_pickups(world, observer) if observer.position.distance_to(pickup.position) <= self.MAX_RAY_LENGTH]
        if not visible:
            return [0.0] * self.PICKUP_FEATURES
        pickup = min(visible, key=lambda p: observer.position.distance_to(p.position))
        dx = pickup.position.x - observer.position.x
        dy = pickup.position.y - observer.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        angle = wrap_angle(math.atan2(dy, dx) - observer.heading)
        return [
            1.0 if pickup.kind == "health" else 0.0,
            1.0 if pickup.kind == "ammo" else 0.0,
            self._norm_distance(distance, self.MAX_RAY_LENGTH),
            math.sin(angle),
            math.cos(angle),
            1.0,
        ]

    def _enemy_visible(self, observer: Agent, target: Agent, world: World) -> bool:
        if not target.is_alive:
            return False
        return observer.position.distance_to(target.position) <= self.MAX_RAY_LENGTH and world.has_line_of_sight(observer.position, target.position)

    @staticmethod
    def _norm_distance(distance: float, max_distance: float) -> float:
        return max(0.0, min(1.0, distance / max(1.0, max_distance)))

    @staticmethod
    def _clip_signed(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))
