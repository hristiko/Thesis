from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from game import detection
from game.agent import Agent, AgentCommand, wrap_angle
from game.world import Vec2, World


BotKind = Literal["fsm", "behavior_tree", "cs2d_style"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _distance(a: Agent, b: Agent) -> float:
    dx = b.position.x - a.position.x
    dy = b.position.y - a.position.y
    return math.hypot(dx, dy)


def _angle_to(a: Agent, b: Agent) -> float:
    return math.atan2(b.position.y - a.position.y, b.position.x - a.position.x)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def _normalize_move(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-6:
        return 0.0, 0.0
    if length <= 1.0:
        return x, y
    return x / length, y / length


def _perpendicular(dx: float, dy: float, side: int) -> tuple[float, float]:
    if side >= 0:
        return -dy, dx
    return dy, -dx


def _has_los(world: World, agent: Agent, target: Agent) -> bool:
    if not agent.is_alive or not target.is_alive:
        return False
    return bool(world.has_line_of_sight(agent.position, target.position))


def _clear_shot(world: World, agent: Agent, target: Agent, heading: float | None = None) -> bool:
    if not agent.is_alive or not target.is_alive:
        return False

    shot_heading = agent.heading if heading is None else float(heading)
    direction = Vec2(math.cos(shot_heading), math.sin(shot_heading))
    muzzle = agent.position + direction * 20.0
    distance_to_target = muzzle.distance_to(target.position)

    ray = detection.trace_ray(
        world=world,
        origin=muzzle,
        direction=direction,
        max_length=distance_to_target + target.radius + 8.0,
        step_size=4.0,
        target_agent_id=target.agent_id,
        observer_agent_id=agent.agent_id,
        include_pickups=False,
        include_projectiles=False,
        enemy_margin=3.0,
    )
    return ray.hit_type == detection.HIT_ENEMY and ray.hit_agent_id == target.agent_id


def _is_reloading(agent: Agent) -> bool:
    return bool(getattr(agent, "reloading", False) or getattr(agent, "is_reloading", False))


def _can_shoot(agent: Agent) -> bool:
    if not agent.is_alive or agent.ammo <= 0 or _is_reloading(agent):
        return False

    fire_cooldown = getattr(agent, "fire_cooldown", 0.0)
    if fire_cooldown > 1e-6:
        return False

    return True


def _can_reload(agent: Agent) -> bool:
    if not agent.is_alive:
        return False
    if _is_reloading(agent):
        return False
    if agent.ammo >= agent.max_ammo:
        return False

    charges = getattr(agent, "reload_charges", None)
    if charges is not None and int(charges) <= 0:
        return False

    return True


@dataclass
class BotPlan:
    move_x: float = 0.0
    move_y: float = 0.0
    shoot: bool = False
    reload: bool = False
    desired_heading: float = 0.0
    label: str = "idle"


class EvaluationBotBase:
    def __init__(
        self,
        agent: Agent,
        world: World,
        target_id: str,
        rng: random.Random | None = None,
        reaction_delay_min: int = 10,
        reaction_delay_max: int = 18,
        aim_noise_rad: float = 0.10,
        max_turn_per_tick: float = 0.10,
        preferred_distance: float = 245.0,
        min_distance: float = 165.0,
        max_distance: float = 325.0,
        shoot_angle_threshold: float = 0.18,
    ) -> None:
        self.agent = agent
        self.world = world
        self.target_id = target_id
        self.rng = rng or random.Random()

        self.reaction_delay_min = int(reaction_delay_min)
        self.reaction_delay_max = int(reaction_delay_max)
        self.aim_noise_rad = float(aim_noise_rad)
        self.max_turn_per_tick = float(max_turn_per_tick)
        self.preferred_distance = float(preferred_distance)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.shoot_angle_threshold = float(shoot_angle_threshold)

        self._reaction_timer = 0
        self._plan = BotPlan(desired_heading=self.agent.heading)
        self._strafe_side = 1 if self.rng.random() < 0.5 else -1
        self._strafe_timer = self.rng.randint(30, 70)

    def reset_memory(self) -> None:
        self._reaction_timer = 0
        self._plan = BotPlan(desired_heading=self.agent.heading)
        self._strafe_side = 1 if self.rng.random() < 0.5 else -1
        self._strafe_timer = self.rng.randint(30, 70)

    def _target(self) -> Agent | None:
        agents_container = getattr(self.world, "agents", [])

        if isinstance(agents_container, dict):
            agents_iterable = agents_container.values()
        else:
            agents_iterable = agents_container

        for possible_agent in agents_iterable:
            if hasattr(possible_agent, "agent_id") and possible_agent.agent_id == self.target_id:
                return possible_agent

        return None

    def _new_reaction_delay(self) -> int:
        return self.rng.randint(self.reaction_delay_min, self.reaction_delay_max)

    def _maybe_flip_strafe(self) -> None:
        self._strafe_timer -= 1
        if self._strafe_timer <= 0:
            self._strafe_side *= -1
            self._strafe_timer = self.rng.randint(35, 80)

    def _aim_toward_plan(self) -> None:
        error = wrap_angle(self._plan.desired_heading - self.agent.heading)
        step = _clamp(error, -self.max_turn_per_tick, self.max_turn_per_tick)
        self.agent.heading = wrap_angle(self.agent.heading + step)

    def _aim_error_to_target(self, target: Agent) -> float:
        return abs(wrap_angle(_angle_to(self.agent, target) - self.agent.heading))

    def _target_heading_with_noise(self, target: Agent) -> float:
        return wrap_angle(_angle_to(self.agent, target) + self.rng.uniform(-self.aim_noise_rad, self.aim_noise_rad))

    def _range_move(self, target: Agent) -> tuple[float, float]:
        dx = target.position.x - self.agent.position.x
        dy = target.position.y - self.agent.position.y
        ux, uy = _unit(dx, dy)
        dist = math.hypot(dx, dy)

        if dist > self.max_distance:
            return ux, uy

        if dist < self.min_distance:
            return -ux, -uy

        sx, sy = _perpendicular(ux, uy, self._strafe_side)
        return sx * 0.75, sy * 0.75

    def _simple_los_recovery_move(self, target: Agent) -> tuple[float, float]:
        dx = target.position.x - self.agent.position.x
        dy = target.position.y - self.agent.position.y
        ux, uy = _unit(dx, dy)
        sx, sy = _perpendicular(ux, uy, self._strafe_side)

        move_x = sx * 0.85 + ux * 0.35
        move_y = sy * 0.85 + uy * 0.35
        return _normalize_move(move_x, move_y)

    def _make_command(self, plan: BotPlan) -> AgentCommand:
        move_x, move_y = _normalize_move(plan.move_x, plan.move_y)
        return AgentCommand(
            shoot=bool(plan.shoot),
            reload=bool(plan.reload),
            use_world_movement=True,
            world_move_x=move_x,
            world_move_y=move_y,
        )

    def _compute_plan(self, target: Agent) -> BotPlan:
        raise NotImplementedError

    def tick(self) -> AgentCommand:
        target = self._target()
        if target is None or not self.agent.is_alive:
            return AgentCommand(use_world_movement=True)

        self._maybe_flip_strafe()

        self._reaction_timer -= 1
        if self._reaction_timer <= 0:
            self._plan = self._compute_plan(target)
            self._reaction_timer = self._new_reaction_delay()

        self._aim_toward_plan()
        return self._make_command(self._plan)


class FSMEvaluationBot(EvaluationBotBase):
    def __init__(
        self,
        agent: Agent,
        world: World,
        target_id: str,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            world=world,
            target_id=target_id,
            rng=rng,
            reaction_delay_min=10,
            reaction_delay_max=18,
            aim_noise_rad=0.115,
            max_turn_per_tick=0.095,
            preferred_distance=245.0,
            min_distance=165.0,
            max_distance=330.0,
            shoot_angle_threshold=0.19,
        )
        self.state = "seek_los"

    def _compute_plan(self, target: Agent) -> BotPlan:
        los = _has_los(self.world, self.agent, target)
        clear = _clear_shot(self.world, self.agent, target)
        dist = _distance(self.agent, target)

        desired_heading = self._target_heading_with_noise(target)
        move_x, move_y = 0.0, 0.0
        shoot = False
        reload = False

        if self.agent.ammo <= 3 and _can_reload(self.agent):
            self.state = "reload"
        elif dist < 135.0:
            self.state = "retreat"
        elif not los:
            self.state = "seek_los"
        elif clear and dist <= 360.0:
            self.state = "attack"
        else:
            self.state = "strafe"

        if self.state == "reload":
            reload = True
            dx = target.position.x - self.agent.position.x
            dy = target.position.y - self.agent.position.y
            ux, uy = _unit(dx, dy)
            move_x, move_y = -ux * 0.45, -uy * 0.45

        elif self.state == "retreat":
            dx = target.position.x - self.agent.position.x
            dy = target.position.y - self.agent.position.y
            ux, uy = _unit(dx, dy)
            move_x, move_y = -ux, -uy

        elif self.state == "seek_los":
            move_x, move_y = self._simple_los_recovery_move(target)

        elif self.state == "attack":
            move_x, move_y = self._range_move(target)
            aim_error = self._aim_error_to_target(target)
            shoot = bool(
                clear
                and _can_shoot(self.agent)
                and aim_error <= self.shoot_angle_threshold
                and self.rng.random() > 0.12
            )

        elif self.state == "strafe":
            move_x, move_y = self._range_move(target)

        return BotPlan(
            move_x=move_x,
            move_y=move_y,
            shoot=shoot,
            reload=reload,
            desired_heading=desired_heading,
            label=self.state,
        )


class BehaviorTreeEvaluationBot(EvaluationBotBase):
    def __init__(
        self,
        agent: Agent,
        world: World,
        target_id: str,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            world=world,
            target_id=target_id,
            rng=rng,
            reaction_delay_min=8,
            reaction_delay_max=14,
            aim_noise_rad=0.085,
            max_turn_per_tick=0.115,
            preferred_distance=250.0,
            min_distance=170.0,
            max_distance=335.0,
            shoot_angle_threshold=0.165,
        )

    def _compute_plan(self, target: Agent) -> BotPlan:
        los = _has_los(self.world, self.agent, target)
        clear = _clear_shot(self.world, self.agent, target)
        dist = _distance(self.agent, target)
        desired_heading = self._target_heading_with_noise(target)

        dx = target.position.x - self.agent.position.x
        dy = target.position.y - self.agent.position.y
        ux, uy = _unit(dx, dy)

        if self.agent.ammo <= 5 and _can_reload(self.agent):
            return BotPlan(
                move_x=-ux * 0.35,
                move_y=-uy * 0.35,
                shoot=False,
                reload=True,
                desired_heading=desired_heading,
                label="bt_reload",
            )

        if self.agent.health <= 35 or dist < 145.0:
            sx, sy = _perpendicular(ux, uy, self._strafe_side)
            return BotPlan(
                move_x=-ux * 0.85 + sx * 0.25,
                move_y=-uy * 0.85 + sy * 0.25,
                shoot=False,
                reload=False,
                desired_heading=desired_heading,
                label="bt_retreat",
            )

        if not los:
            move_x, move_y = self._simple_los_recovery_move(target)
            return BotPlan(
                move_x=move_x,
                move_y=move_y,
                shoot=False,
                reload=False,
                desired_heading=desired_heading,
                label="bt_recover_los",
            )

        if clear:
            aim_error = self._aim_error_to_target(target)
            move_x, move_y = self._range_move(target)
            shoot = bool(
                _can_shoot(self.agent)
                and aim_error <= self.shoot_angle_threshold
                and self.rng.random() > 0.08
            )
            return BotPlan(
                move_x=move_x,
                move_y=move_y,
                shoot=shoot,
                reload=False,
                desired_heading=desired_heading,
                label="bt_attack",
            )

        move_x, move_y = self._range_move(target)
        return BotPlan(
            move_x=move_x,
            move_y=move_y,
            shoot=False,
            reload=False,
            desired_heading=desired_heading,
            label="bt_strafe",
        )


class CS2DStyleApproxBot(EvaluationBotBase):
    def __init__(
        self,
        agent: Agent,
        world: World,
        target_id: str,
        rng: random.Random | None = None,
        skill: float = 0.55,
    ) -> None:
        skill = _clamp(float(skill), 0.0, 1.0)

        reaction_min = int(round(16 - 7 * skill))
        reaction_max = int(round(26 - 12 * skill))
        aim_noise = 0.17 - 0.10 * skill
        turn_speed = 0.075 + 0.055 * skill
        shoot_threshold = 0.23 - 0.07 * skill

        super().__init__(
            agent=agent,
            world=world,
            target_id=target_id,
            rng=rng,
            reaction_delay_min=reaction_min,
            reaction_delay_max=reaction_max,
            aim_noise_rad=aim_noise,
            max_turn_per_tick=turn_speed,
            preferred_distance=245.0,
            min_distance=155.0,
            max_distance=340.0,
            shoot_angle_threshold=shoot_threshold,
        )

        self.skill = skill
        self._burst_remaining = 0
        self._burst_cooldown = 0

    def reset_memory(self) -> None:
        super().reset_memory()
        self._burst_remaining = 0
        self._burst_cooldown = 0

    def _maybe_start_burst(self, clear: bool, aim_error: float) -> None:
        if self._burst_cooldown > 0:
            self._burst_cooldown -= 1
            return

        if (
            clear
            and _can_shoot(self.agent)
            and aim_error <= self.shoot_angle_threshold
            and self.rng.random() < (0.45 + 0.25 * self.skill)
        ):
            self._burst_remaining = self.rng.randint(2, 4)
            self._burst_cooldown = self.rng.randint(12, 26)

    def _compute_plan(self, target: Agent) -> BotPlan:
        los = _has_los(self.world, self.agent, target)
        clear = _clear_shot(self.world, self.agent, target)
        dist = _distance(self.agent, target)
        desired_heading = self._target_heading_with_noise(target)

        dx = target.position.x - self.agent.position.x
        dy = target.position.y - self.agent.position.y
        ux, uy = _unit(dx, dy)
        sx, sy = _perpendicular(ux, uy, self._strafe_side)

        if _can_reload(self.agent):
            if self.agent.ammo <= 4:
                return BotPlan(
                    move_x=-ux * 0.30 + sx * 0.25,
                    move_y=-uy * 0.30 + sy * 0.25,
                    shoot=False,
                    reload=True,
                    desired_heading=desired_heading,
                    label="cs2d_reload_low",
                )
            if self.agent.ammo <= 8 and (not los or dist > 290.0) and self.rng.random() < 0.20:
                return BotPlan(
                    move_x=sx * 0.35,
                    move_y=sy * 0.35,
                    shoot=False,
                    reload=True,
                    desired_heading=desired_heading,
                    label="cs2d_reload_safe",
                )

        if not los:
            move_x, move_y = self._simple_los_recovery_move(target)
            return BotPlan(
                move_x=move_x,
                move_y=move_y,
                shoot=False,
                reload=False,
                desired_heading=desired_heading,
                label="cs2d_recover_los",
            )

        if dist < self.min_distance:
            move_x = -ux * 0.80 + sx * 0.35
            move_y = -uy * 0.80 + sy * 0.35
        elif dist > self.max_distance:
            move_x = ux * 0.65 + sx * 0.30
            move_y = uy * 0.65 + sy * 0.30
        else:
            move_x = sx * 0.75
            move_y = sy * 0.75

        aim_error = self._aim_error_to_target(target)
        self._maybe_start_burst(clear, aim_error)

        shoot = False
        if self._burst_remaining > 0 and clear and _can_shoot(self.agent):
            if self.rng.random() > 0.18:
                shoot = True
                self._burst_remaining -= 1

        return BotPlan(
            move_x=move_x,
            move_y=move_y,
            shoot=shoot,
            reload=False,
            desired_heading=desired_heading,
            label="cs2d_burst_strafe",
        )


def create_evaluation_bot(
    kind: BotKind,
    agent: Agent,
    world: World,
    target_id: str,
    rng: random.Random | None = None,
):
    if kind == "fsm":
        return FSMEvaluationBot(agent=agent, world=world, target_id=target_id, rng=rng)

    if kind == "behavior_tree":
        return BehaviorTreeEvaluationBot(agent=agent, world=world, target_id=target_id, rng=rng)

    if kind == "cs2d_style":
        return CS2DStyleApproxBot(agent=agent, world=world, target_id=target_id, rng=rng, skill=0.55)

    raise ValueError(f"Unknown evaluation bot kind: {kind}")