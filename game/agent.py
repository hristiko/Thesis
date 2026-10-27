from __future__ import annotations

import math
from dataclasses import dataclass

from config.game_constants import (
    GLOCK_DAMAGE_HP,
    GLOCK_FIRE_RATE,
    GLOCK_MAX_AMMO,
    GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
    GLOCK_RELOAD_TIME,
    PLAYER_MAX_HEALTH,
    PLAYER_MOVE_SPEED_PX_PER_SEC,
    PLAYER_RADIUS_PX,
)
from game.world import Event, Vec2, World


@dataclass
class AgentCommand:
    move_idx: int = 0      # 0 none, 1 forward, 2 backward
    strafe_idx: int = 0    # 0 none, 1 left, 2 right
    rot_idx: int = 0       # 0 none, 1 left, 2 right
    shoot: bool = False
    reload: bool = False

    use_world_movement: bool = False
    world_move_x: float = 0.0
    world_move_y: float = 0.0


class Agent:
    TURN_SPEED = 3.4
    RADIUS = PLAYER_RADIUS_PX

    def __init__(
        self,
        agent_id: str,
        can_move: bool = True,
        move_speed: float = PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health: float = PLAYER_MAX_HEALTH,
        max_ammo: int = GLOCK_MAX_AMMO,
        fire_rate: float = GLOCK_FIRE_RATE,
        reload_time: float = GLOCK_RELOAD_TIME,
        projectile_damage: float = GLOCK_DAMAGE_HP,
        projectile_speed: float = GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        auto_reload_when_empty: bool = False,
        reload_charges: int | None = None,
    ):
        self.agent_id = agent_id
        self.can_move = can_move
        self.move_speed = move_speed
        self.max_health = max_health
        self.max_ammo = max_ammo
        self.fire_rate = fire_rate
        self.reload_time = reload_time
        self.projectile_damage = projectile_damage
        self.projectile_speed = projectile_speed
        self.auto_reload_when_empty = auto_reload_when_empty
        self.max_reload_charges = reload_charges
        self.reload_charges = reload_charges

        self.radius = self.RADIUS
        self.position = Vec2(0.0, 0.0)
        self.heading = 0.0
        self.health = max_health
        self.ammo = max_ammo
        self.reloading = False
        self.reload_timer = 0.0
        self.fire_timer = 0.0
        self.reload_charges = self.max_reload_charges
        self.velocity = Vec2(0.0, 0.0)
        self.blocked_last_step = False

    def reset_state(self) -> None:
        self.health = self.max_health
        self.ammo = self.max_ammo
        self.reloading = False
        self.reload_timer = 0.0
        self.fire_timer = 0.0
        self.reload_charges = self.max_reload_charges
        self.velocity = Vec2(0.0, 0.0)
        self.blocked_last_step = False

    def before_movement_step(self) -> None:
        self.blocked_last_step = False

    def _start_reload(self, events: list[Event]) -> bool:
        reload_data = {
            "ammo_before": self.ammo,
            "max_ammo": self.max_ammo,
            "ammo_ratio_before": self.ammo_ratio,
            "reload_charges_before": self.reload_charges,
        }
        if self.reloading:
            events.append(Event("reload_failed_already_reloading", self.agent_id, data=reload_data))
            return False
        if self.ammo >= self.max_ammo:
            events.append(Event("reload_failed_full_ammo", self.agent_id, data=reload_data))
            return False
        if self.reload_charges is not None and self.reload_charges <= 0:
            events.append(Event("reload_failed_no_reserve", self.agent_id, data=reload_data))
            return False
        self.reloading = True
        self.reload_timer = self.reload_time
        if self.reload_charges is not None:
            self.reload_charges -= 1
        reload_data["reload_charges_after"] = self.reload_charges
        events.append(Event("reload_started", self.agent_id, data=reload_data))
        return True

    def apply_command(self, command: AgentCommand, world: World) -> list[Event]:
        events: list[Event] = []
        dt = world.dt

        if command.rot_idx == 1:
            self.heading -= self.TURN_SPEED * dt
        elif command.rot_idx == 2:
            self.heading += self.TURN_SPEED * dt
        self.heading = wrap_angle(self.heading)

        if self.can_move:
            if command.use_world_movement:
                move_vector = Vec2(command.world_move_x, command.world_move_y)
                if move_vector.length() > 0.0:
                    move_vector = move_vector.normalize() * self.move_speed
                self.velocity = move_vector
            else:
                forward = Vec2(math.cos(self.heading), math.sin(self.heading))
                right = Vec2(-forward.y, forward.x)

                move_vector = Vec2(0.0, 0.0)
                if command.move_idx == 1:
                    move_vector = move_vector + forward
                elif command.move_idx == 2:
                    move_vector = move_vector + (forward * -1.0)

                if command.strafe_idx == 1:
                    move_vector = move_vector + (right * -1.0)
                elif command.strafe_idx == 2:
                    move_vector = move_vector + right

                if move_vector.length() > 0.0:
                    move_vector = move_vector.normalize() * self.move_speed
                self.velocity = move_vector
        else:
            self.velocity = Vec2(0.0, 0.0)

        if command.reload:
            self._start_reload(events)

        can_fire_now = self.fire_timer <= 0.0 and not self.reloading and self.ammo > 0
        if command.shoot and can_fire_now:
            direction = Vec2(math.cos(self.heading), math.sin(self.heading))
            moving = self.velocity.length() > max(5.0, self.move_speed * 0.10)
            shot_data = {
                "moving_shot": moving,
                "owner_heading": self.heading,
                "owner_position": (self.position.x, self.position.y),
                "ammo_before": self.ammo,
            }
            world.fire_projectile(
                owner_id=self.agent_id,
                start_position=self.position,
                direction=direction,
                damage=self.projectile_damage,
                speed=self.projectile_speed,
                data=shot_data,
            )
            self.ammo -= 1
            self.fire_timer = 1.0 / self.fire_rate
            events.append(Event("shot_fired", self.agent_id, data=shot_data))
            if self.ammo < 0:
                self.ammo = 0

            if self.auto_reload_when_empty and self.ammo == 0:
                self._start_reload(events)

        elif command.shoot:
            if self.reloading:
                events.append(Event("shot_failed_reloading", self.agent_id))
            elif self.fire_timer > 0.0:
                events.append(Event("shot_failed_cooldown", self.agent_id, value=self.fire_timer))
            elif self.ammo <= 0:
                events.append(Event("shot_failed_no_ammo", self.agent_id))
                if self.auto_reload_when_empty:
                    self._start_reload(events)

        if self.auto_reload_when_empty and self.ammo == 0 and not self.reloading:
            self._start_reload(events)

        return events

    def update_timers(self, dt: float) -> None:
        if self.reloading:
            self.reload_timer -= dt
            if self.reload_timer <= 0.0:
                self.reloading = False
                self.reload_timer = 0.0
                self.ammo = self.max_ammo
        if self.fire_timer > 0.0:
            self.fire_timer = max(0.0, self.fire_timer - dt)

    def take_damage(self, amount: float) -> bool:
        self.health = max(0.0, self.health - amount)
        return self.health <= 0.0

    @property
    def is_alive(self) -> bool:
        return self.health > 0.0

    @property
    def health_ratio(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0.0

    @property
    def ammo_ratio(self) -> float:
        return self.ammo / self.max_ammo if self.max_ammo > 0 else 0.0

    @property
    def reload_progress(self) -> float:
        if not self.reloading or self.reload_time <= 0.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.reload_timer / self.reload_time)))

    @property
    def fire_ready(self) -> float:
        return 1.0 if self.fire_timer <= 0.0 and not self.reloading and self.ammo > 0 else 0.0


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
