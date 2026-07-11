from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


HIT_NOTHING = 0
HIT_WALL_OR_OBSTACLE = 1
HIT_ENEMY = 2
HIT_PICKUP = 3
HIT_PROJECTILE = 4
HIT_TYPE_COUNT = 5

HIT_TYPE_NAMES = {
    HIT_NOTHING: "nothing",
    HIT_WALL_OR_OBSTACLE: "wall_or_obstacle",
    HIT_ENEMY: "enemy",
    HIT_PICKUP: "pickup",
    HIT_PROJECTILE: "projectile",
}


@dataclass(frozen=True)
class DetectionDebugLine:
    start: tuple[float, float]
    end: tuple[float, float]
    hit_type: int
    hit_point: tuple[float, float] | None = None
    blocker_type: str | None = None


@dataclass(frozen=True)
class MovementCheck:
    blocked: bool
    final_position: tuple[float, float]
    collision_point: tuple[float, float] | None = None
    blocker_type: str | None = None


@dataclass(frozen=True)
class ProjectileCheck:
    hit: bool
    hit_type: str | None = None
    hit_agent_id: str | None = None
    hit_point: tuple[float, float] | None = None
    blocker_type: str | None = None


@dataclass(frozen=True)
class RayHit:
    hit_type: int
    distance: float
    hit_point: tuple[float, float]
    hit_agent_id: str | None = None
    hit_pickup_id: str | None = None
    hit_projectile_owner_id: str | None = None
    blocker_type: str | None = None


@dataclass(frozen=True)
class ProjectileThreat:
    exists: bool
    distance: float = 0.0
    relative_angle: float = 0.0
    direction_angle: float = 0.0
    time_to_impact: float = 1.0
    danger: bool = False
    projectile_owner_id: str | None = None


@dataclass(frozen=True)
class PickupOverlap:
    overlaps: bool
    pickup: Any | None = None


@dataclass(frozen=True)
class _Point:
    x: float
    y: float


def _xy(point: Any) -> tuple[float, float]:
    if isinstance(point, tuple) or isinstance(point, list):
        return float(point[0]), float(point[1])
    return float(point.x), float(point.y)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _lerp(a: tuple[float, float], b: tuple[float, float], ratio: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _rect_allows(rect: Any, purpose: str) -> bool:
    if purpose == "movement":
        return bool(getattr(rect, "blocks_movement", True))
    if purpose == "sight":
        return bool(getattr(rect, "blocks_sight", True))
    if purpose == "projectile":
        return bool(getattr(rect, "blocks_projectiles", True))
    return True


def point_hits_wall_or_obstacle(world: Any, point: Any, purpose: str = "movement") -> tuple[bool, str | None]:
    """Point-vs-geometry query shared by sight, projectile, and debug code."""
    x, y = _xy(point)
    if x <= 0.0 or y <= 0.0 or x >= float(world.width) or y >= float(world.height):
        return True, "arena_wall"

    for obstacle in getattr(world, "obstacles", []):
        if not _rect_allows(obstacle, purpose):
            continue
        if obstacle.left <= x <= obstacle.right and obstacle.top <= y <= obstacle.bottom:
            return True, "obstacle"
    return False, None


def circle_intersects_rect(center: Any, radius: float, rect: Any) -> bool:
    x, y = _xy(center)
    nearest_x = max(rect.left, min(rect.right, x))
    nearest_y = max(rect.top, min(rect.bottom, y))
    dx = x - nearest_x
    dy = y - nearest_y
    return (dx * dx + dy * dy) <= radius * radius


def circle_hits_wall_or_obstacle(world: Any, center: Any, radius: float, purpose: str = "movement") -> tuple[bool, str | None]:
    """Circle-vs-geometry query used for agent movement and bot wall avoidance."""
    x, y = _xy(center)
    if x - radius < 0.0 or y - radius < 0.0 or x + radius > world.width or y + radius > world.height:
        return True, "arena_wall"

    for obstacle in getattr(world, "obstacles", []):
        if not _rect_allows(obstacle, purpose):
            continue
        if circle_intersects_rect(center, radius, obstacle):
            return True, "obstacle"
    return False, None


def check_movement_path(
    world: Any,
    start: Any,
    intended_end: Any,
    radius: float,
    step_size: float = 4.0,
) -> MovementCheck:
    """Swept movement: sample from start to intended end and stop at first block."""
    start_xy = _xy(start)
    end_xy = _xy(intended_end)
    distance = _distance_xy(start_xy, end_xy)
    if distance <= 1e-9:
        return MovementCheck(blocked=False, final_position=start_xy)

    samples = max(1, int(math.ceil(distance / max(1.0, step_size))))
    last_clear = start_xy

    for index in range(1, samples + 1):
        sample = _lerp(start_xy, end_xy, index / samples)
        blocked, blocker = circle_hits_wall_or_obstacle(world, _Point(sample[0], sample[1]), radius, purpose="movement")
        if blocked:
            return MovementCheck(True, last_clear, sample, blocker)
        last_clear = sample

    return MovementCheck(False, end_xy)


def check_projectile_path(
    world: Any,
    owner_id: str,
    start: Any,
    intended_end: Any,
    projectile_radius: float,
    step_size: float = 4.0,
) -> ProjectileCheck:
    """Swept projectile collision so fast bullets cannot tunnel through objects."""
    start_xy = _xy(start)
    end_xy = _xy(intended_end)
    distance = _distance_xy(start_xy, end_xy)
    samples = max(1, int(math.ceil(distance / max(1.0, step_size))))

    for index in range(1, samples + 1):
        sample = _lerp(start_xy, end_xy, index / samples)
        sample_obj = _Point(sample[0], sample[1])

        blocked, blocker = point_hits_wall_or_obstacle(world, sample_obj, purpose="projectile")
        if blocked:
            return ProjectileCheck(True, "wall_or_obstacle", None, sample, blocker)

        for agent in getattr(world, "agents", {}).values():
            if agent.agent_id == owner_id or not agent.is_alive:
                continue
            if _distance_xy(sample, _xy(agent.position)) <= projectile_radius + agent.radius:
                return ProjectileCheck(True, "agent", agent.agent_id, sample, "agent")

    return ProjectileCheck(False, hit_point=end_xy)


def has_line_of_sight(world: Any, start: Any, end: Any, step_size: float = 6.0) -> bool:
    """Line-of-sight query used by bots, rewards, UI, and observations."""
    start_xy = _xy(start)
    end_xy = _xy(end)
    distance = _distance_xy(start_xy, end_xy)
    if distance <= 1e-9:
        return True

    samples = max(1, int(math.ceil(distance / max(1.0, step_size))))
    for index in range(1, samples):
        sample = _lerp(start_xy, end_xy, index / samples)
        blocked, _ = point_hits_wall_or_obstacle(world, _Point(sample[0], sample[1]), purpose="sight")
        if blocked:
            return False
    return True


def check_pickup_overlap(world: Any, agent: Any) -> PickupOverlap:
    """Pickup collision query shared by world updates and tests."""
    for pickup in getattr(world, "pickups", []):
        if not getattr(pickup, "active", False):
            continue
        if _distance_xy(_xy(agent.position), _xy(pickup.position)) <= agent.radius + pickup.radius:
            return PickupOverlap(True, pickup)
    return PickupOverlap(False, None)


def trace_ray(
    world: Any,
    origin: Any,
    direction: Any,
    max_length: float,
    step_size: float,
    target_agent_id: str | None = None,
    observer_agent_id: str | None = None,
    include_pickups: bool = True,
    include_projectiles: bool = True,
    enemy_margin: float = 6.0,
    pickup_margin: float = 4.0,
    projectile_margin: float = 5.0,
) -> RayHit:
    """Trace one observation ray and return the first object hit.

    Per step the query order is wall/obstacle, enemy, pickup, projectile.  A wall
    or sight-blocking obstacle therefore hides objects behind it.
    """
    ox, oy = _xy(origin)
    dx, dy = _xy(direction)
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return RayHit(HIT_NOTHING, max_length, (ox, oy))
    dx /= length
    dy /= length

    distance = step_size
    last_point = (ox + dx * max_length, oy + dy * max_length)
    while distance <= max_length:
        sample = (ox + dx * distance, oy + dy * distance)
        sample_obj = _Point(sample[0], sample[1])

        blocked, blocker = point_hits_wall_or_obstacle(world, sample_obj, purpose="sight")
        if blocked:
            return RayHit(HIT_WALL_OR_OBSTACLE, distance, sample, blocker_type=blocker)

        if target_agent_id is None:
            agents = [
                a for a in getattr(world, "agents", {}).values()
                if getattr(a, "is_alive", False) and a.agent_id != observer_agent_id
            ]
        else:
            agent = getattr(world, "agents", {}).get(target_agent_id)
            agents = [agent] if agent is not None and agent.is_alive else []

        for agent in agents:
            if _distance_xy(sample, _xy(agent.position)) <= agent.radius + enemy_margin:
                return RayHit(HIT_ENEMY, distance, sample, hit_agent_id=agent.agent_id)

        if include_pickups:
            for pickup in getattr(world, "pickups", []):
                if not pickup.active:
                    continue
                if _distance_xy(sample, _xy(pickup.position)) <= pickup.radius + pickup_margin:
                    return RayHit(HIT_PICKUP, distance, sample, hit_pickup_id=pickup.pickup_id)

        if include_projectiles:
            for projectile in getattr(world, "projectiles", []):
                if projectile.owner_id == observer_agent_id:
                    continue
                if _distance_xy(sample, _xy(projectile.position)) <= projectile.radius + projectile_margin:
                    return RayHit(
                        HIT_PROJECTILE,
                        distance,
                        sample,
                        hit_projectile_owner_id=projectile.owner_id,
                    )

        last_point = sample
        distance += step_size

    return RayHit(HIT_NOTHING, max_length, last_point)


def trace_many_rays(
    world: Any,
    origin: Any,
    heading: float,
    ray_count: int,
    max_length: float,
    step_size: float,
    target_agent_id: str | None = None,
    observer_agent_id: str | None = None,
    include_pickups: bool = True,
    include_projectiles: bool = True,
) -> tuple[list[RayHit], list[DetectionDebugLine]]:
    hits: list[RayHit] = []
    debug_lines: list[DetectionDebugLine] = []
    ox, oy = _xy(origin)

    for ray_index in range(ray_count):
        relative_angle = -math.pi + ((2.0 * math.pi) * ray_index / ray_count)
        ray_angle = heading + relative_angle
        direction = _Point(math.cos(ray_angle), math.sin(ray_angle))
        hit = trace_ray(
            world=world,
            origin=origin,
            direction=direction,
            max_length=max_length,
            step_size=step_size,
            target_agent_id=target_agent_id,
            observer_agent_id=observer_agent_id,
            include_pickups=include_pickups,
            include_projectiles=include_projectiles,
        )
        hits.append(hit)
        debug_lines.append(
            DetectionDebugLine(
                start=(ox, oy),
                end=hit.hit_point,
                hit_type=hit.hit_type,
                hit_point=hit.hit_point if hit.hit_type != HIT_NOTHING else None,
                blocker_type=hit.blocker_type,
            )
        )

    return hits, debug_lines


def nearest_visible_enemy(world: Any, observer: Any, enemy_ids: list[str] | None = None) -> Any | None:
    ids = enemy_ids or [agent_id for agent_id in getattr(world, "agents", {}) if agent_id != observer.agent_id]
    best_agent = None
    best_distance = float("inf")
    for agent_id in ids:
        agent = world.agents.get(agent_id)
        if agent is None or not agent.is_alive:
            continue
        if not has_line_of_sight(world, observer.position, agent.position):
            continue
        distance = _distance_xy(_xy(observer.position), _xy(agent.position))
        if distance < best_distance:
            best_distance = distance
            best_agent = agent
    return best_agent


def visible_pickups(world: Any, observer: Any) -> list[Any]:
    visible = []
    for pickup in getattr(world, "pickups", []):
        if pickup.active and has_line_of_sight(world, observer.position, pickup.position):
            visible.append(pickup)
    return visible


def nearest_wall_distance(world: Any, origin: Any, max_distance: float = 360.0, ray_count: int = 16, step_size: float = 8.0) -> float:
    """Approximate distance to nearest sight-blocking wall/obstacle by radial probes."""
    best = max_distance
    for index in range(ray_count):
        angle = (2.0 * math.pi) * index / ray_count
        hit = trace_ray(
            world=world,
            origin=origin,
            direction=_Point(math.cos(angle), math.sin(angle)),
            max_length=max_distance,
            step_size=step_size,
            target_agent_id="__none__",
            observer_agent_id=None,
            include_pickups=False,
            include_projectiles=False,
        )
        if hit.hit_type == HIT_WALL_OR_OBSTACLE:
            best = min(best, hit.distance)
    return best


def projectile_threat(world: Any, observer: Any, max_distance: float = 420.0, max_tti: float = 1.25) -> ProjectileThreat:
    """Return the nearest enemy projectile and whether it is dangerous.

    The danger flag is based on estimated closest approach, not hidden intent.  It
    gives PPO enough partial information to learn dodging without receiving full
    global state.
    """
    observer_xy = _xy(observer.position)
    best: ProjectileThreat | None = None
    best_key = float("inf")

    for projectile in getattr(world, "projectiles", []):
        if projectile.owner_id == observer.agent_id:
            continue
        pos_xy = _xy(projectile.position)
        distance = _distance_xy(observer_xy, pos_xy)
        if distance > max_distance:
            continue

        vx, vy = _xy(projectile.velocity)
        speed_sq = vx * vx + vy * vy
        if speed_sq <= 1e-9:
            continue

        to_observer_x = observer_xy[0] - pos_xy[0]
        to_observer_y = observer_xy[1] - pos_xy[1]
        tti = (to_observer_x * vx + to_observer_y * vy) / speed_sq
        closest_x = pos_xy[0] + vx * max(0.0, tti)
        closest_y = pos_xy[1] + vy * max(0.0, tti)
        closest_distance = _distance_xy(observer_xy, (closest_x, closest_y))
        danger_radius = observer.radius + projectile.radius + 14.0
        danger = 0.0 <= tti <= max_tti and closest_distance <= danger_radius

        relative_angle = math.atan2(pos_xy[1] - observer_xy[1], pos_xy[0] - observer_xy[0]) - observer.heading
        direction_angle = math.atan2(vy, vx) - observer.heading
        threat = ProjectileThreat(
            exists=True,
            distance=distance,
            relative_angle=_wrap_angle(relative_angle),
            direction_angle=_wrap_angle(direction_angle),
            time_to_impact=max(0.0, min(max_tti, tti)) / max_tti,
            danger=danger,
            projectile_owner_id=projectile.owner_id,
        )
        key = threat.time_to_impact if danger else 1.0 + distance / max_distance
        if key < best_key:
            best = threat
            best_key = key

    return best if best is not None else ProjectileThreat(False)
