from __future__ import annotations

import math
from dataclasses import dataclass, field

from config.game_constants import PLAYER_RADIUS_PX
from game.world import Vec2, World
from navigation.astar import GridCell, astar
from navigation.combat_positioning import clamp_point_for_player, desired_combat_position, is_walkable_point


@dataclass(frozen=True)
class NavigationReference:
    

    valid: bool = False
    desired_position: Vec2 = field(default_factory=Vec2)
    waypoint: Vec2 = field(default_factory=Vec2)
    path: tuple[Vec2, ...] = field(default_factory=tuple)
    desired_distance: float = 0.0
    waypoint_distance: float = 0.0
    path_distance: float = 0.0
    path_deviation: float = 0.0
    progress_delta: float = 0.0
    desired_progress_delta: float = 0.0
    waypoint_reached: bool = False
    recomputed: bool = False


class NavigationReferenceTracker:
    

    def __init__(
        self,
        *,
        ideal_range: float,
        update_interval_steps: int = 8,
        enemy_repath_distance: float = 28.0,
        target_repath_distance: float = 28.0,
        waypoint_radius: float = 34.0,
        cell_size: float = 32.0,
    ):
        self.ideal_range = float(ideal_range)
        self.update_interval_steps = int(max(1, update_interval_steps))
        self.enemy_repath_distance = float(enemy_repath_distance)
        self.target_repath_distance = float(target_repath_distance)
        self.waypoint_radius = float(waypoint_radius)
        self.cell_size = float(cell_size)
        self._step = 0
        self._last_enemy_pos: Vec2 | None = None
        self._last_desired_pos: Vec2 | None = None
        self._path: list[Vec2] = []
        self._previous_path_distance: float | None = None
        self._previous_desired_distance: float | None = None
        self.current = NavigationReference()

    def reset(self) -> None:
        self._step = 0
        self._last_enemy_pos = None
        self._last_desired_pos = None
        self._path = []
        self._previous_path_distance = None
        self._previous_desired_distance = None
        self.current = NavigationReference()

    def update(self, *, learner: object, enemy: object, world: World, force: bool = False) -> NavigationReference:
        self._step += 1
        learner_pos = learner.position
        enemy_pos = enemy.position
        agent_radius = float(getattr(learner, "radius", PLAYER_RADIUS_PX))

        desired = desired_combat_position(
            learner_pos=learner_pos,
            enemy_pos=enemy_pos,
            world=world,
            ideal_range=self.ideal_range,
            agent_radius=agent_radius,
        )

        waypoint_reached = bool(self.current.valid and learner_pos.distance_to(self.current.waypoint) <= self.waypoint_radius)
        needs_repath = force or not self._path or waypoint_reached
        if self._step % self.update_interval_steps == 0:
            needs_repath = True
        if self._last_enemy_pos is None or enemy_pos.distance_to(self._last_enemy_pos) >= self.enemy_repath_distance:
            needs_repath = True
        if self._last_desired_pos is None or desired.distance_to(self._last_desired_pos) >= self.target_repath_distance:
            needs_repath = True

        recomputed = False
        if needs_repath:
            self._path = self._find_astar_path(world, learner_pos, desired, agent_radius)
            self._last_enemy_pos = Vec2(enemy_pos.x, enemy_pos.y)
            self._last_desired_pos = Vec2(desired.x, desired.y)
            recomputed = True

        if not self._path:
            self.current = NavigationReference(valid=False)
            return self.current

        waypoint = self._select_waypoint(learner_pos, self._path)
        desired_distance = learner_pos.distance_to(desired)
        waypoint_distance = learner_pos.distance_to(waypoint)
        path_distance = self._path_distance_from(learner_pos, self._path, waypoint)
        path_deviation = self._distance_to_path(learner_pos, self._path)

        progress_delta = 0.0 if self._previous_path_distance is None else self._previous_path_distance - path_distance
        desired_progress_delta = 0.0 if self._previous_desired_distance is None else self._previous_desired_distance - desired_distance
        self._previous_path_distance = path_distance
        self._previous_desired_distance = desired_distance

        self.current = NavigationReference(
            valid=True,
            desired_position=desired,
            waypoint=waypoint,
            path=tuple(self._path),
            desired_distance=desired_distance,
            waypoint_distance=waypoint_distance,
            path_distance=path_distance,
            path_deviation=path_deviation,
            progress_delta=progress_delta,
            desired_progress_delta=desired_progress_delta,
            waypoint_reached=waypoint_reached,
            recomputed=recomputed,
        )
        return self.current

    def _find_astar_path(self, world: World, start: Vec2, goal: Vec2, agent_radius: float) -> list[Vec2]:
        start_cell = self._cell_for_point(start, world, agent_radius)
        goal_cell = self._cell_for_point(goal, world, agent_radius)

        cells = astar(
            start_cell,
            goal_cell,
            is_walkable=lambda cell: self._is_walkable_cell(cell, world, agent_radius),
            neighbors=lambda cell: self._neighbors(cell, world, agent_radius),
        )
        if not cells:
            return [clamp_point_for_player(start, world, agent_radius), clamp_point_for_player(goal, world, agent_radius)]
        path = [self._point_for_cell(cell, world, agent_radius) for cell in cells]
        # Preserve the exact desired combat position as the reward target.
        if path[-1].distance_to(goal) > 1.0:
            path.append(clamp_point_for_player(goal, world, agent_radius))
        return path

    def _columns(self, world: World) -> int:
        return max(1, int(math.ceil(world.width / self.cell_size)))

    def _rows(self, world: World) -> int:
        return max(1, int(math.ceil(world.height / self.cell_size)))

    def _cell_for_point(self, point: Vec2, world: World, agent_radius: float) -> GridCell:
        p = clamp_point_for_player(point, world, agent_radius)
        col = int(max(0, min(self._columns(world) - 1, p.x // self.cell_size)))
        row = int(max(0, min(self._rows(world) - 1, p.y // self.cell_size)))
        return col, row

    def _point_for_cell(self, cell: GridCell, world: World, agent_radius: float) -> Vec2:
        col, row = cell
        return clamp_point_for_player(Vec2((col + 0.5) * self.cell_size, (row + 0.5) * self.cell_size), world, agent_radius)

    def _is_walkable_cell(self, cell: GridCell, world: World, agent_radius: float) -> bool:
        col, row = cell
        if col < 0 or row < 0 or col >= self._columns(world) or row >= self._rows(world):
            return False
        return is_walkable_point(self._point_for_cell(cell, world, agent_radius), world, agent_radius)

    def _neighbors(self, cell: GridCell, world: World, agent_radius: float) -> list[GridCell]:
        col, row = cell
        out: list[GridCell] = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                nxt = (col + dc, row + dr)
                if not self._is_walkable_cell(nxt, world, agent_radius):
                    continue
                if dc != 0 and dr != 0:
                    if not self._is_walkable_cell((col + dc, row), world, agent_radius):
                        continue
                    if not self._is_walkable_cell((col, row + dr), world, agent_radius):
                        continue
                out.append(nxt)
        return out

    def _select_waypoint(self, learner_pos: Vec2, path: list[Vec2]) -> Vec2:
        for point in path[1:]:
            if learner_pos.distance_to(point) > self.waypoint_radius:
                return point
        return path[-1]

    @staticmethod
    def _path_distance_from(learner_pos: Vec2, path: list[Vec2], waypoint: Vec2) -> float:
        if not path:
            return 0.0
        try:
            start_index = path.index(waypoint)
        except ValueError:
            start_index = 1 if len(path) > 1 else 0
        total = learner_pos.distance_to(waypoint)
        for index in range(start_index, len(path) - 1):
            total += path[index].distance_to(path[index + 1])
        return total

    @staticmethod
    def _distance_to_path(point: Vec2, path: list[Vec2]) -> float:
        if len(path) < 2:
            return point.distance_to(path[0]) if path else 0.0
        best = float("inf")
        for index in range(len(path) - 1):
            best = min(best, _point_segment_distance(point, path[index], path[index + 1]))
        return best


def _point_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> float:
    ab_x = b.x - a.x
    ab_y = b.y - a.y
    ap_x = point.x - a.x
    ap_y = point.y - a.y
    denom = ab_x * ab_x + ab_y * ab_y
    if denom <= 1e-9:
        return point.distance_to(a)
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / denom))
    closest = Vec2(a.x + ab_x * t, a.y + ab_y * t)
    return point.distance_to(closest)
