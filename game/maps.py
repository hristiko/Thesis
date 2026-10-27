from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.stages import normalize_stage_id


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float
    blocks_movement: bool = True
    blocks_sight: bool = True
    blocks_projectiles: bool = True

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass(frozen=True)
class PickupTemplate:
    kind: str
    x: float
    y: float
    amount: float
    radius: float = 14.0


@dataclass(frozen=True)
class SpawnPair:
    learner_x: float
    learner_y: float
    learner_heading: float
    enemy_x: float
    enemy_y: float
    enemy_heading: float


@dataclass(frozen=True)
class MapLayout:
    width: int
    height: int
    obstacles: list[RectObstacle] = field(default_factory=list)
    pickups: list[PickupTemplate] = field(default_factory=list)
    spawns: list[SpawnPair] = field(default_factory=list)
    # Diagnostic metadata. These IDs are logged by Stage 7 so a bad validation
    # result can be traced back to the exact obstacle variant and spawn pair.
    obstacle_variant_id: int = -1
    spawn_pair_ids: list[int] = field(default_factory=list)


ARENA_WIDTH = 1200
ARENA_HEIGHT = 800

OPEN_COMBAT_SPAWNS = [
    SpawnPair(450, 320, 0.0, 700, 320, 3.14159),
    SpawnPair(450, 320, 0.0, 200, 320, 0.0),
    SpawnPair(450, 130, 1.5708, 450, 560, -1.5708),
    SpawnPair(450, 560, -1.5708, 450, 130, 1.5708),
    SpawnPair(450, 330, 0.0, 690, 170, 2.35619),
    SpawnPair(450, 330, 0.0, 210, 500, -0.78539),
    SpawnPair(220, 220, 0.2, 840, 580, -2.6),
    SpawnPair(820, 210, 2.7, 240, 580, -0.4),
]

OPEN_VALIDATION_SPAWNS = [
    SpawnPair(260, 170, 0.35, 880, 490, -2.65),
    SpawnPair(890, 180, 2.75, 250, 505, -0.25),
    SpawnPair(320, 610, -0.55, 850, 235, 2.45),
]

OPEN_MOVEMENT_SPAWNS = [
    SpawnPair(180, 320, 0.0, 720, 320, 3.14159),
    SpawnPair(180, 170, 0.0, 720, 500, 3.14159),
    SpawnPair(180, 500, 0.0, 720, 170, 3.14159),
    SpawnPair(450, 95, 1.5708, 450, 570, -1.5708),
    SpawnPair(920, 580, 3.14159, 360, 235, 0.0),
]

OBSTACLE_STAGE_SPAWNS = [
    SpawnPair(160, 320, 0.0, 740, 320, 3.14159),
    SpawnPair(140, 170, 0.0, 760, 500, 3.14159),
    SpawnPair(140, 500, 0.0, 760, 170, 3.14159),
    SpawnPair(990, 200, 3.0, 260, 610, -0.45),
    SpawnPair(980, 620, -2.8, 240, 195, 0.4),
]

OBSTACLE_VALIDATION_SPAWNS = [
    SpawnPair(210, 660, -0.35, 950, 200, 2.7),
    SpawnPair(970, 645, -2.9, 220, 180, 0.25),
    SpawnPair(200, 245, 0.1, 965, 565, -2.85),
]

CENTER_BLOCK = RectObstacle(360, 220, 180, 260)
TOP_LEFT_BLOCK = RectObstacle(190, 120, 90, 90)
BOTTOM_RIGHT_BLOCK = RectObstacle(620, 490, 90, 90)

# Stage 7 is phased. Early phases use fewer obstacles so the policy learns
# line-of-sight recovery and wall-safe movement before full obstacle combat.
STAGE7_ONE_OBSTACLE_VARIANTS: list[list[RectObstacle]] = [
    [CENTER_BLOCK],
    [RectObstacle(430, 170, 140, 320)],
]


STAGE8_OBSTACLE_BRIDGE_SPAWNS = [
    OBSTACLE_STAGE_SPAWNS[0],
    OBSTACLE_STAGE_SPAWNS[1],
    OBSTACLE_STAGE_SPAWNS[2],
]

STAGE8_OBSTACLE_BRIDGE_VALIDATION_SPAWNS = [
    OBSTACLE_VALIDATION_SPAWNS[0],
    OBSTACLE_VALIDATION_SPAWNS[2],
]

STAGE8_OBSTACLE_BRIDGE_VARIANTS: list[list[RectObstacle]] = [
    STAGE7_ONE_OBSTACLE_VARIANTS[0],
]

STAGE8_OBSTACLE_BRIDGE_VALIDATION_VARIANTS: list[list[RectObstacle]] = [
    STAGE7_ONE_OBSTACLE_VARIANTS[1],
]

STAGE7_TWO_OBSTACLE_VARIANTS: list[list[RectObstacle]] = [
    [CENTER_BLOCK, TOP_LEFT_BLOCK],
    [RectObstacle(430, 170, 140, 320), RectObstacle(210, 560, 150, 80)],
    [RectObstacle(330, 320, 250, 85), RectObstacle(670, 415, 190, 90)],
]

OBSTACLE_VARIANTS: list[list[RectObstacle]] = [
    [CENTER_BLOCK, TOP_LEFT_BLOCK, BOTTOM_RIGHT_BLOCK],
    [RectObstacle(430, 170, 140, 320), RectObstacle(210, 560, 150, 80), RectObstacle(750, 110, 95, 150)],
    [RectObstacle(330, 320, 250, 85), RectObstacle(670, 415, 190, 90), RectObstacle(525, 125, 80, 125)],
    [RectObstacle(280, 185, 100, 220), RectObstacle(590, 240, 120, 230), RectObstacle(820, 560, 120, 80)],
]

STAGE7_FOUR_OBSTACLE_VARIANTS: list[list[RectObstacle]] = [
    [CENTER_BLOCK, TOP_LEFT_BLOCK, BOTTOM_RIGHT_BLOCK, RectObstacle(860, 260, 95, 170)],
    [RectObstacle(430, 170, 140, 320), RectObstacle(210, 560, 150, 80), RectObstacle(750, 110, 95, 150), RectObstacle(835, 500, 130, 90)],
    [RectObstacle(330, 320, 250, 85), RectObstacle(670, 415, 190, 90), RectObstacle(525, 125, 80, 125), RectObstacle(900, 220, 90, 150)],
    [RectObstacle(280, 185, 100, 220), RectObstacle(590, 240, 120, 230), RectObstacle(820, 560, 120, 80), RectObstacle(165, 510, 110, 95)],
]

OBSTACLE_VALIDATION_VARIANTS: list[list[RectObstacle]] = [
    [RectObstacle(405, 245, 120, 300), RectObstacle(740, 225, 130, 85), RectObstacle(230, 560, 130, 90)],
    [RectObstacle(305, 165, 95, 160), RectObstacle(535, 390, 250, 90), RectObstacle(865, 510, 90, 155)],
]

STAGE7_FOUR_OBSTACLE_VALIDATION_VARIANTS: list[list[RectObstacle]] = [
    [RectObstacle(405, 245, 120, 300), RectObstacle(740, 225, 130, 85), RectObstacle(230, 560, 130, 90), RectObstacle(905, 360, 95, 170)],
    [RectObstacle(305, 165, 95, 160), RectObstacle(535, 390, 250, 90), RectObstacle(865, 510, 90, 155), RectObstacle(165, 445, 130, 85)],
]


def _stage7_obstacle_variants(phase_index: int, validation: bool) -> list[list[RectObstacle]]:
    phase_index = max(1, min(5, int(phase_index)))
    if phase_index <= 1:
        return STAGE7_ONE_OBSTACLE_VARIANTS
    if phase_index == 2:
        return STAGE7_TWO_OBSTACLE_VARIANTS
    if phase_index == 3:
        return OBSTACLE_VALIDATION_VARIANTS if validation else OBSTACLE_VARIANTS
    # 7D and 7E both use four obstacles. 7D changes pressure, not geometry;
    # 7E is the full final validation on the held-out four-obstacle variants.
    return STAGE7_FOUR_OBSTACLE_VALIDATION_VARIANTS if validation else STAGE7_FOUR_OBSTACLE_VARIANTS

PICKUP_VARIANTS: list[list[PickupTemplate]] = [
    [
        PickupTemplate("health", 260, 350, 30),
        PickupTemplate("ammo", 640, 350, 8),
        PickupTemplate("ammo", 450, 140, 6),
        PickupTemplate("health", 450, 560, 25),
    ],
    [
        PickupTemplate("health", 305, 610, 25),
        PickupTemplate("ammo", 810, 275, 8),
        PickupTemplate("ammo", 525, 705, 6),
        PickupTemplate("health", 905, 520, 25),
    ],
    [
        PickupTemplate("health", 225, 230, 25),
        PickupTemplate("ammo", 930, 660, 7),
        PickupTemplate("ammo", 565, 170, 6),
        PickupTemplate("health", 680, 675, 30),
    ],
]

PICKUP_VALIDATION_VARIANTS: list[list[PickupTemplate]] = [
    [
        PickupTemplate("health", 350, 720, 25),
        PickupTemplate("ammo", 880, 165, 8),
        PickupTemplate("ammo", 495, 520, 6),
        PickupTemplate("health", 1010, 395, 30),
    ]
]


def _choice_index(options: list[Any], rng: Any | None, fallback_index: int = 0) -> int:
    if not options:
        raise ValueError("Cannot choose from an empty list")
    if rng is None:
        return int(fallback_index % len(options))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, len(options)))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(len(options)))
    return int(fallback_index % len(options))


def _choice(options: list[Any], rng: Any | None, fallback_index: int = 0) -> Any:
    return options[_choice_index(options, rng, fallback_index=fallback_index)]


STAGE7_SPAWN_OBSTACLE_CLEARANCE = 24.0


def _point_rect_distance(x: float, y: float, obstacle: RectObstacle) -> float:
    dx = max(obstacle.left - x, 0.0, x - obstacle.right)
    dy = max(obstacle.top - y, 0.0, y - obstacle.bottom)
    return (dx * dx + dy * dy) ** 0.5


def _point_clear_of_obstacles(
    x: float,
    y: float,
    obstacles: list[RectObstacle],
    clearance: float = STAGE7_SPAWN_OBSTACLE_CLEARANCE,
) -> bool:
    return all(_point_rect_distance(x, y, obstacle) >= clearance for obstacle in obstacles)


def _spawn_clear_of_obstacles(
    spawn: SpawnPair,
    obstacles: list[RectObstacle],
    clearance: float = STAGE7_SPAWN_OBSTACLE_CLEARANCE,
) -> bool:
    return (
        _point_clear_of_obstacles(spawn.learner_x, spawn.learner_y, obstacles, clearance)
        and _point_clear_of_obstacles(spawn.enemy_x, spawn.enemy_y, obstacles, clearance)
    )


def _valid_spawns_for_obstacles(
    spawns: list[SpawnPair],
    obstacles: list[RectObstacle],
    clearance: float = STAGE7_SPAWN_OBSTACLE_CLEARANCE,
) -> tuple[list[SpawnPair], list[int]]:
    valid_pairs = [
        (index, spawn) for index, spawn in enumerate(spawns)
        if _spawn_clear_of_obstacles(spawn, obstacles, clearance)
    ]
    if not valid_pairs:
        raise ValueError("No Stage 7 spawns are clear of the selected obstacles")
    return [spawn for _, spawn in valid_pairs], [index for index, _ in valid_pairs]


def build_layout(
    stage_id: str,
    rng: Any | None = None,
    validation: bool = False,
    stage7_phase_index: int | None = None,
    stage8_phase_index: int | None = None,
) -> MapLayout:
    """Build a map layout for training, validation, play, or sanity checks.

    The returned layout remains simple and thesis-readable. Stage 7 and Stage 8
    can sample obstacle/pickup variants; validation uses held-out variants when
    possible so a best model is not promoted from one memorized map only.
    """
    stage_id = normalize_stage_id(stage_id)

    if stage_id == "stage0":
        obstacles = _choice(OBSTACLE_VALIDATION_VARIANTS if validation else OBSTACLE_VARIANTS, rng)
        pickups = _choice(PICKUP_VALIDATION_VARIANTS if validation else PICKUP_VARIANTS, rng)
        spawns = OBSTACLE_VALIDATION_SPAWNS if validation else OBSTACLE_STAGE_SPAWNS
        return MapLayout(ARENA_WIDTH, ARENA_HEIGHT, obstacles=list(obstacles), pickups=list(pickups), spawns=list(spawns), spawn_pair_ids=list(range(len(spawns))))

    if stage_id == "stage8":
        
        phase_index = 5 if validation and stage8_phase_index is None else int(stage8_phase_index or 1)
        phase_index = max(1, min(6, phase_index))

        if phase_index <= 2 or phase_index == 5:
            spawns = OPEN_VALIDATION_SPAWNS if validation else OPEN_COMBAT_SPAWNS
            return MapLayout(
                ARENA_WIDTH,
                ARENA_HEIGHT,
                obstacles=[],
                pickups=[],
                spawns=list(spawns),
                spawn_pair_ids=list(range(len(spawns))),
            )

        obstacle_variants = (
            STAGE8_OBSTACLE_BRIDGE_VALIDATION_VARIANTS
            if validation
            else STAGE8_OBSTACLE_BRIDGE_VARIANTS
        )
        obstacle_variant_id = _choice_index(obstacle_variants, rng)
        obstacles = list(obstacle_variants[obstacle_variant_id])

        spawn_pool = (
            STAGE8_OBSTACLE_BRIDGE_VALIDATION_SPAWNS
            if validation
            else STAGE8_OBSTACLE_BRIDGE_SPAWNS
        )
        valid_spawns, spawn_pair_ids = _valid_spawns_for_obstacles(list(spawn_pool), obstacles)

        return MapLayout(
            ARENA_WIDTH,
            ARENA_HEIGHT,
            obstacles=obstacles,
            pickups=[],
            spawns=valid_spawns,
            obstacle_variant_id=obstacle_variant_id,
            spawn_pair_ids=spawn_pair_ids,
        )

    if stage_id in {"stage1", "stage2", "stage4", "stage5", "stage6"}:
        spawns = OPEN_VALIDATION_SPAWNS if validation else OPEN_COMBAT_SPAWNS
        return MapLayout(ARENA_WIDTH, ARENA_HEIGHT, obstacles=[], pickups=[], spawns=list(spawns), spawn_pair_ids=list(range(len(spawns))))

    if stage_id == "stage3":
        spawns = OPEN_VALIDATION_SPAWNS if validation else OPEN_MOVEMENT_SPAWNS
        return MapLayout(ARENA_WIDTH, ARENA_HEIGHT, obstacles=[], pickups=[], spawns=list(spawns), spawn_pair_ids=list(range(len(spawns))))

    if stage_id == "stage7":
        phase_index = 5 if validation and stage7_phase_index is None else int(stage7_phase_index or 1)
        obstacle_variants = _stage7_obstacle_variants(phase_index, validation=validation)
        obstacle_variant_id = _choice_index(obstacle_variants, rng)
        obstacles = list(obstacle_variants[obstacle_variant_id])
        spawn_pool = OBSTACLE_VALIDATION_SPAWNS if validation else OBSTACLE_STAGE_SPAWNS
        spawns, spawn_pair_ids = _valid_spawns_for_obstacles(list(spawn_pool), obstacles)
        return MapLayout(
            ARENA_WIDTH,
            ARENA_HEIGHT,
            obstacles=obstacles,
            pickups=[],
            spawns=spawns,
            obstacle_variant_id=obstacle_variant_id,
            spawn_pair_ids=spawn_pair_ids,
        )

    raise ValueError(f"Unknown stage_id: {stage_id}")