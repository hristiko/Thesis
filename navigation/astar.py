from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable

GridCell = tuple[int, int]


def astar(
    start: GridCell,
    goal: GridCell,
    *,
    is_walkable: Callable[[GridCell], bool],
    neighbors: Callable[[GridCell], Iterable[GridCell]],
) -> list[GridCell]:
    
    if start == goal:
        return [start]
    if not is_walkable(start) or not is_walkable(goal):
        return []

    def heuristic(cell: GridCell) -> float:
        return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    open_heap: list[tuple[float, int, GridCell]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, start))
    came_from: dict[GridCell, GridCell] = {}
    g_score: dict[GridCell, float] = {start: 0.0}
    visited: set[GridCell] = set()
    counter = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)
        visited.add(current)

        cx, cy = current
        for nxt in neighbors(current):
            if nxt in visited or not is_walkable(nxt):
                continue
            nx, ny = nxt
            edge_cost = math.hypot(nx - cx, ny - cy)
            tentative_g = g_score[current] + edge_cost
            if tentative_g + 1e-9 >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative_g
            counter += 1
            heapq.heappush(open_heap, (tentative_g + heuristic(nxt), counter, nxt))

    return []


def _reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
