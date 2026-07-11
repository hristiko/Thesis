from __future__ import annotations

import math

from config.game_constants import PLAYER_RADIUS_PX, STAGE3_WALL_CLEARANCE_PX
from game.world import Vec2, World


def clamp_point_for_player(point: Vec2, world: World, radius: float = PLAYER_RADIUS_PX, margin: float = 0.0) -> Vec2:
    safe = radius + margin
    return Vec2(
        max(safe, min(float(world.width) - safe, point.x)),
        max(safe, min(float(world.height) - safe, point.y)),
    )


def is_walkable_point(point: Vec2, world: World, radius: float = PLAYER_RADIUS_PX, margin: float = 0.0) -> bool:
    p = clamp_point_for_player(point, world, radius, margin)
    return not world.is_circle_blocked(p, radius + margin)


def desired_combat_position(
    *,
    learner_pos: Vec2,
    enemy_pos: Vec2,
    world: World,
    ideal_range: float,
    agent_radius: float = PLAYER_RADIUS_PX,
) -> Vec2:
    """Choose a reachable point near the enemy at approximately combat range.

    This is not a controller. It produces only a reward/diagnostic target. The
    implementation is deliberately grid/A*-friendly and no longer depends on a
    separate navigation-mesh abstraction. Near corners it prefers the best visible feasible point
    instead of forcing a large wall buffer that can trap the learner.
    """
    dx = learner_pos.x - enemy_pos.x
    dy = learner_pos.y - enemy_pos.y
    base_angle = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0

    offsets = [0.0, 0.20, -0.20, 0.45, -0.45, 0.75, -0.75, 1.05, -1.05, 1.45, -1.45, math.pi]
    offsets.extend((2.0 * math.pi * i / 20.0) for i in range(20))

    best_point: Vec2 | None = None
    best_score = float("inf")
    for offset in offsets:
        angle = base_angle + offset
        raw = Vec2(enemy_pos.x + math.cos(angle) * ideal_range, enemy_pos.y + math.sin(angle) * ideal_range)
        candidate = clamp_point_for_player(raw, world, agent_radius)
        if not is_walkable_point(candidate, world, agent_radius):
            continue

        distance_from_learner = learner_pos.distance_to(candidate)
        line_of_sight_penalty = 0.0 if world.has_line_of_sight(candidate, enemy_pos) else 220.0
        boundary_penalty = _boundary_penalty(candidate, world, agent_radius)
        angle_penalty = min(abs(offset), math.pi) * 12.0
        range_error = abs(candidate.distance_to(enemy_pos) - ideal_range)
        # Range is important, but near map edges exact ideal range can be
        # impossible. Do not let range error dominate feasibility and visibility.
        score = distance_from_learner + line_of_sight_penalty + boundary_penalty + angle_penalty + range_error * 0.18
        if score < best_score:
            best_score = score
            best_point = candidate

    if best_point is not None:
        return best_point

    fallback = Vec2(
        enemy_pos.x + math.cos(base_angle) * ideal_range,
        enemy_pos.y + math.sin(base_angle) * ideal_range,
    )
    return clamp_point_for_player(fallback, world, agent_radius)


def _boundary_penalty(point: Vec2, world: World, radius: float) -> float:
    clearance = min(point.x, point.y, float(world.width) - point.x, float(world.height) - point.y) - radius
    if clearance >= STAGE3_WALL_CLEARANCE_PX:
        return 0.0
    return (STAGE3_WALL_CLEARANCE_PX - clearance) * 1.20
