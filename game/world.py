from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from game import detection
from game.maps import MapLayout, RectObstacle

if TYPE_CHECKING:
    from game.agent import Agent


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()

    def normalize(self) -> "Vec2":
        length = self.length()
        if length <= 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)


@dataclass
class Event:
    kind: str
    agent_id: str
    other_id: str | None = None
    value: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.kind


@dataclass
class DebugPolyline:
    points: list[tuple[float, float]] = field(default_factory=list)
    kind: str = "generic"
    width: int = 2


@dataclass
class Projectile:
    owner_id: str
    position: Vec2
    velocity: Vec2
    damage: float
    radius: float = 3.0
    lifetime: float = 1.1
    data: dict[str, Any] = field(default_factory=dict)
    threatened_agent_ids: set[str] = field(default_factory=set)
    rewarded_agent_ids: set[str] = field(default_factory=set)


@dataclass
class Pickup:
    pickup_id: str
    kind: str
    position: Vec2
    amount: float
    radius: float = 14.0
    active: bool = True

    def reset(self) -> None:
        self.active = True


class World:
    """Engine truth for the shooter simulation.

    Exact geometry and hidden positions live here. RL observations are built by
    querying detection/raycast functions, so the policy receives only visible or
    detectable information rather than direct full-state coordinates.
    """

    def __init__(
        self,
        layout: MapLayout,
        dt: float = 1.0 / 30.0,
        stage_id: str | None = None,
    ):
        self.layout = layout
        self.stage_id = stage_id or getattr(layout, "stage_id", "")
        self.width = layout.width
        self.height = layout.height
        self.dt = dt
        self.step_count = 0
        self.agents: dict[str, "Agent"] = {}
        self.projectiles: list[Projectile] = []
        self.obstacles: list[RectObstacle] = list(layout.obstacles)
        self.pickups: list[Pickup] = [
            Pickup(
                pickup_id=f"pickup_{index}_{template.kind}",
                kind=template.kind,
                position=Vec2(template.x, template.y),
                amount=template.amount,
                radius=template.radius,
            )
            for index, template in enumerate(layout.pickups)
        ]
        self.debug_rays: list[detection.DetectionDebugLine] = []
        self.debug_points: list[tuple[float, float, str]] = []
        self.debug_polylines: list[DebugPolyline] = []
        self.last_events: list[Event] = []
        self.stage5_enemy_can_move = True
        self.stage5_enemy_can_shoot = True
        self.stage5_first_shot_delay_steps = 0
        self.stage5_max_active_enemy_projectiles = 99
        self.stage6_enemy_can_move = True
        self.stage6_enemy_can_shoot = True
        self.stage6_first_shot_delay_steps = 0
        self.stage6_max_active_enemy_projectiles = 99
        self._visibility_state: dict[tuple[str, str], bool] = {}
        self._next_projectile_uid = 1

    def add_agent(self, agent: "Agent") -> None:
        self.agents[agent.agent_id] = agent

    def reset_round(self) -> None:
        self.step_count = 0
        self.projectiles.clear()
        self.debug_rays.clear()
        self.debug_points.clear()
        self.debug_polylines.clear()
        self.last_events.clear()
        self._visibility_state.clear()
        self._next_projectile_uid = 1
        for agent in self.agents.values():
            agent.reset_state()
        for pickup in self.pickups:
            pickup.reset()

    def tick(self) -> list[Event]:
        self.step_count += 1
        events: list[Event] = []
        self.debug_points.clear()
        self.debug_polylines.clear()

        self._move_agents(events)
        self._collect_pickups(events)
        self._update_projectiles(events)
        self._update_visibility_events(events)

        for agent in self.agents.values():
            if agent.is_alive:
                agent.update_timers(self.dt)

        self.last_events = list(events)
        return events

    def fire_projectile(
        self,
        owner_id: str,
        start_position: Vec2,
        direction: Vec2,
        damage: float,
        speed: float,
        data: dict[str, Any] | None = None,
    ) -> None:
        unit = direction.normalize()
        muzzle = start_position + unit * 20.0
        projectile_data = dict(data or {})
        projectile_data.setdefault("projectile_uid", self._next_projectile_uid)
        self._next_projectile_uid += 1
        self.projectiles.append(
            Projectile(
                owner_id=owner_id,
                position=muzzle,
                velocity=unit * speed,
                damage=damage,
                data=projectile_data,
            )
        )

    # Compatibility wrappers. Internally they all delegate to game.detection.
    def has_line_of_sight(self, start: Vec2, end: Vec2, step: float = 6.0) -> bool:
        return detection.has_line_of_sight(self, start, end, step_size=step)

    def point_hits_wall(self, point: Vec2) -> bool:
        blocked, _ = detection.point_hits_wall_or_obstacle(self, point, purpose="movement")
        return blocked

    def is_circle_blocked(self, center: Vec2, radius: float) -> bool:
        blocked, _ = detection.circle_hits_wall_or_obstacle(self, center, radius, purpose="movement")
        return blocked

    def _move_agents(self, events: list[Event]) -> None:
        for agent in self.agents.values():
            if not agent.is_alive:
                continue

            agent.before_movement_step()

            if agent.velocity.length() <= 0.0:
                continue

            start = agent.position.copy()
            intended_end = Vec2(
                agent.position.x + agent.velocity.x * self.dt,
                agent.position.y + agent.velocity.y * self.dt,
            )

            check = detection.check_movement_path(
                world=self,
                start=start,
                intended_end=intended_end,
                radius=agent.radius,
                step_size=4.0,
            )

            agent.position.x, agent.position.y = check.final_position

            if check.blocked:
                agent.blocked_last_step = True

                if check.collision_point is not None:
                    self.debug_points.append(
                        (
                            check.collision_point[0],
                            check.collision_point[1],
                            "blocked_move",
                        )
                    )

                events.append(
                    Event(
                        kind="blocked_move",
                        agent_id=agent.agent_id,
                        data={
                            "blocker_type": check.blocker_type or "unknown",
                            "position": check.collision_point,
                            "attempted_end": (intended_end.x, intended_end.y),
                            "corrected_final_position": check.final_position,
                        },
                    )
                )

    def _collect_pickups(self, events: list[Event]) -> None:
        for agent in self.agents.values():
            if not agent.is_alive:
                continue

            overlap = detection.check_pickup_overlap(self, agent)
            if not overlap.overlaps or overlap.pickup is None:
                continue

            pickup = overlap.pickup
            useful = False
            actual_amount = 0.0

            if pickup.kind == "health":
                # Health regain is intentionally disabled for the final curriculum.
                # Legacy health pickup templates may still exist for old sanity checks,
                # but they no longer restore HP.
                useful = False
                actual_amount = 0.0

            elif pickup.kind == "ammo":
                useful = agent.ammo < agent.max_ammo
                before_ammo = agent.ammo
                agent.ammo = min(agent.max_ammo, agent.ammo + int(pickup.amount))
                actual_amount = float(agent.ammo - before_ammo)

            pickup.active = False

            base_data = {
                "pickup_kind": pickup.kind,
                "useful": useful,
                "pickup_id": pickup.pickup_id,
                "amount_applied": actual_amount,
            }

            events.append(
                Event(
                    "pickup_collected",
                    agent.agent_id,
                    value=actual_amount,
                    data=base_data,
                )
            )

            if useful:
                events.append(
                    Event(
                        "useful_pickup_collected",
                        agent.agent_id,
                        value=actual_amount,
                        data=base_data,
                    )
                )
            else:
                events.append(
                    Event(
                        "useless_pickup_collected",
                        agent.agent_id,
                        value=actual_amount,
                        data=base_data,
                    )
                )

    def _update_projectiles(self, events: list[Event]) -> None:
        alive_projectiles: list[Projectile] = []

        for projectile in self.projectiles:
            start = projectile.position.copy()
            intended_end = projectile.position + projectile.velocity * self.dt
            projectile.lifetime -= self.dt

            check = detection.check_projectile_path(
                world=self,
                owner_id=projectile.owner_id,
                start=start,
                intended_end=intended_end,
                projectile_radius=projectile.radius,
                step_size=4.0,
            )

            end_for_tracking = intended_end
            if check.hit_point is not None:
                end_for_tracking = Vec2(check.hit_point[0], check.hit_point[1])
            hit_agent_id = check.hit_agent_id if check.hit_type == "agent" else None
            self._track_projectile_dodge_events(projectile, start, end_for_tracking, events, hit_agent_id=hit_agent_id)

            if check.hit:
                if check.hit_point is not None:
                    projectile.position = Vec2(check.hit_point[0], check.hit_point[1])
                    self.debug_points.append((check.hit_point[0], check.hit_point[1], check.hit_type or "projectile_hit"))

                if check.hit_type == "agent" and check.hit_agent_id is not None:
                    target = self.agents.get(check.hit_agent_id)

                    if target is not None and target.is_alive:
                        killed = target.take_damage(projectile.damage)

                        event_data = dict(projectile.data)
                        event_data["hit_point"] = check.hit_point

                        events.append(Event("shot_hit", projectile.owner_id, other_id=target.agent_id, value=projectile.damage, data=event_data))
                        events.append(Event("damage_dealt", projectile.owner_id, other_id=target.agent_id, value=projectile.damage, data=event_data))
                        events.append(Event("damage_taken", target.agent_id, other_id=projectile.owner_id, value=projectile.damage, data=event_data))
                        events.append(Event("hit", projectile.owner_id, other_id=target.agent_id, value=projectile.damage, data=event_data))

                        if killed:
                            events.append(Event("kill", projectile.owner_id, other_id=target.agent_id))
                            events.append(Event("death", target.agent_id, other_id=projectile.owner_id))

                else:
                    blocker = check.blocker_type or "wall_or_obstacle"

                    event_data = dict(projectile.data)
                    event_data.update({"blocker_type": blocker, "position": check.hit_point})

                    if blocker == "arena_wall" and self.stage_id == "stage2":
                        events.append(Event("shot_missed", projectile.owner_id, data=event_data))
                    else:
                        events.append(Event("shot_blocked", projectile.owner_id, data=event_data))

                    if blocker == "arena_wall":
                        events.append(Event("projectile_blocked_by_wall", projectile.owner_id, data=event_data))
                    elif blocker == "obstacle":
                        events.append(Event("projectile_blocked_by_obstacle", projectile.owner_id, data=event_data))

                    self._maybe_emit_projectile_safe_events(projectile, end_for_tracking, events)

                continue

            projectile.position = intended_end

            if projectile.lifetime <= 0.0:
                events.append(Event("shot_missed", projectile.owner_id, data=dict(projectile.data)))
                self._maybe_emit_projectile_safe_events(projectile, intended_end, events)
                continue

            alive_projectiles.append(projectile)

        self.projectiles = alive_projectiles

    def _track_projectile_dodge_events(
        self,
        projectile: Projectile,
        start: Vec2,
        end: Vec2,
        events: list[Event],
        hit_agent_id: str | None = None,
    ) -> None:
        """Track same-tick projectile pass events for Stage 4 dodge learning.

        A projectile can threaten an agent before it either hits, expires, or is
        blocked.  We mark agents whose center corridor is threatened and emit an
        immediate event when the projectile crosses behind that agent's center on
        the same tick without hitting it.
        """
        if projectile.velocity.length() <= 1e-9:
            return

        projectile_dir = projectile.velocity.normalize()
        segment = end - start
        if segment.length() <= 1e-9:
            return

        for agent_id, agent in self.agents.items():
            if agent_id == projectile.owner_id or not agent.is_alive:
                continue
            if hit_agent_id is not None and agent_id == hit_agent_id:
                continue
            if agent_id in projectile.rewarded_agent_ids:
                continue

            to_agent_start = agent.position - start
            prev_forward = to_agent_start.x * projectile_dir.x + to_agent_start.y * projectile_dir.y
            closest_x = start.x + projectile_dir.x * prev_forward
            closest_y = start.y + projectile_dir.y * prev_forward
            lateral_dx = agent.position.x - closest_x
            lateral_dy = agent.position.y - closest_y
            lateral_distance = math.sqrt(lateral_dx * lateral_dx + lateral_dy * lateral_dy)
            danger_radius = agent.radius + projectile.radius + 14.0

            if lateral_distance <= danger_radius:
                projectile.threatened_agent_ids.add(agent_id)

            to_agent_end = agent.position - end
            curr_forward = to_agent_end.x * projectile_dir.x + to_agent_end.y * projectile_dir.y
            passed_agent = prev_forward >= 0.0 and curr_forward < 0.0

            if not passed_agent:
                continue
            if agent_id not in projectile.threatened_agent_ids:
                continue

            projectile.rewarded_agent_ids.add(agent_id)
            event_data = dict(projectile.data)
            event_data.update({
                'projectile_position': (end.x, end.y),
                'agent_position': (agent.position.x, agent.position.y),
                'lateral_distance': lateral_distance,
                'danger_radius': danger_radius,
                'prev_forward': prev_forward,
                'curr_forward': curr_forward,
            })
            events.append(Event('projectile_passed_agent', agent_id, other_id=projectile.owner_id, data=event_data))

    def _maybe_emit_projectile_safe_events(
        self,
        projectile: Projectile,
        end_position: Vec2,
        events: list[Event],
    ) -> None:
        """Emit a safe-resolution event for threatened agents not yet rewarded."""
        for agent_id in sorted(projectile.threatened_agent_ids):
            if agent_id in projectile.rewarded_agent_ids:
                continue
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            projectile.rewarded_agent_ids.add(agent_id)
            event_data = dict(projectile.data)
            event_data.update({
                'projectile_position': (end_position.x, end_position.y),
                'agent_position': (agent.position.x, agent.position.y),
                'safe_resolution': True,
            })
            events.append(Event('projectile_resolved_safe', agent_id, other_id=projectile.owner_id, data=event_data))


    def _update_visibility_events(self, events: list[Event]) -> None:
        agent_items = list(self.agents.items())

        for observer_id, observer in agent_items:
            if not observer.is_alive:
                continue

            for target_id, target in agent_items:
                if observer_id == target_id or not target.is_alive:
                    continue

                key = (observer_id, target_id)
                visible = self.has_line_of_sight(observer.position, target.position)
                previous = self._visibility_state.get(key)
                self._visibility_state[key] = visible

                if previous is None or previous == visible:
                    continue

                if visible:
                    events.append(
                        Event(
                            "line_of_sight_gained",
                            observer_id,
                            other_id=target_id,
                        )
                    )
                    events.append(
                        Event(
                            "enemy_visible",
                            observer_id,
                            other_id=target_id,
                        )
                    )
                else:
                    events.append(
                        Event(
                            "line_of_sight_lost",
                            observer_id,
                            other_id=target_id,
                        )
                    )
                    events.append(
                        Event(
                            "enemy_lost",
                            observer_id,
                            other_id=target_id,
                        )
                    )