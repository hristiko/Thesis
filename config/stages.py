from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.game_constants import (
    GLOCK_DAMAGE_HP,
    GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
    GLOCK_FIRE_RATE,
    PLAYER_MOVE_SPEED_PX_PER_SEC,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

RUN_FAMILY = "curriculum_masked_ppo_v3"

CHECKPOINTS_ROOT = ROOT_DIR / "checkpoints"
CHECKPOINTS_DIR = CHECKPOINTS_ROOT / RUN_FAMILY
LOGS_DIR = ROOT_DIR / "logs" / RUN_FAMILY
PLOTS_DIR = ROOT_DIR / "plots" / RUN_FAMILY
STATS_PATH = ROOT_DIR / f"stage_stats_{RUN_FAMILY}.json"
GLOBAL_LATEST_TRAINING_STATE = CHECKPOINTS_ROOT / "latest.pt"


@dataclass(frozen=True)
class StageConfig:
    stage_id: str
    stage_number: int
    name: str
    description: str

    trainable: bool
    required_previous_stage: str | None

    learner_can_move: bool
    learner_can_strafe: bool
    learner_can_rotate: bool
    learner_can_shoot: bool
    learner_can_reload: bool
    learner_auto_reload: bool

    # Scenario controls.
    enemy_can_move: bool
    enemy_can_shoot: bool
    use_obstacles: bool
    use_pickups: bool
    randomize_spawns: bool
    randomize_obstacles: bool
    randomize_pickups: bool

    enemy_move_speed: float
    enemy_fire_rate: float
    enemy_reload_time: float
    enemy_projectile_damage: float
    enemy_projectile_speed: float
    enemy_behavior_profile: str

    learning_skill: str
    active_skills: tuple[str, ...]
    active_actions: tuple[str, ...]
    locked_actions: tuple[str, ...]
    observation_inputs: tuple[str, ...]
    reward_components: tuple[str, ...]
    success_criteria: str
    promotion_criteria: str
    validation_criteria: str

    previous_skill_eval_stage_ids: tuple[str, ...] = field(default_factory=tuple)
    previous_skill_min_success: float = 0.55
    saves: tuple[str, ...] = field(default_factory=tuple)
    carries_over: tuple[str, ...] = field(default_factory=tuple)

    allow_same_stage_resume: bool = True
    end_episode_on_success: bool = False
    max_steps: int = 420
    success_threshold: float = 0.80
    success_window_size: int = 20
    required_consecutive_checks: int = 5
    frame_stack: int = 4

    @property
    def stage_code(self) -> str:
        return str(self.stage_number)

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINTS_DIR / self.stage_id

    @property
    def log_dir(self) -> Path:
        return LOGS_DIR / self.stage_id

    @property
    def best_model_filename(self) -> str:
        return f"bestBotStage{self.stage_code}.zip"

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / self.best_model_filename

    @property
    def latest_model_path(self) -> Path:
        return self.checkpoint_dir / f"latestStage{self.stage_code}.zip"

    @property
    def interrupted_model_path(self) -> Path:
        return self.checkpoint_dir / f"interruptedStage{self.stage_code}.zip"

    @property
    def latest_training_state_path(self) -> Path:
        return self.checkpoint_dir / f"latestTrainingStateStage{self.stage_code}.pt"

    @property
    def interrupted_training_state_path(self) -> Path:
        return self.checkpoint_dir / f"interruptedTrainingStateStage{self.stage_code}.pt"

    @property
    def best_training_state_path(self) -> Path:
        return self.checkpoint_dir / f"bestTrainingStateStage{self.stage_code}.pt"

    @property
    def config_snapshot_path(self) -> Path:
        return self.log_dir / "training_config.json"

    @property
    def skill_table_snapshot_path(self) -> Path:
        return self.log_dir / "skill_table_latest.json"

    @property
    def train_log_path(self) -> Path:
        return self.log_dir / "train_log.csv"

    @property
    def eval_log_path(self) -> Path:
        return self.log_dir / "eval_log.csv"

    @property
    def actions_enabled(self) -> dict[str, bool]:
        return {
            "move": self.learner_can_move,
            "strafe": self.learner_can_move and self.learner_can_strafe,
            "rotate": self.learner_can_rotate,
            "shoot": self.learner_can_shoot,
            "reload": self.learner_can_reload,
        }


COMMON_OBS = (
    "partial raycast perception with one-hot hit types: none, wall, enemy, pickup, projectile",
    "own health, ammo, reload, fire-ready, and velocity features",
    "visible enemy features only when line of sight exists",
    "nearest enemy projectile distance, angle, direction, danger, and time-to-impact estimate",
    "nearest wall and visible pickup features",
    "line-of-sight flag",
    "full skill-table status vector",
    "frame-stacked observations for short-term memory",
)


STAGE1_OBS = (
    "relative angle to enemy as sin, cos, signed error, and absolute aim error",
    "enemy relative position and distance",
    "enemy relative velocity / movement direction",
    "learner and enemy heading",
    "line-of-sight and alignment flags",
    "frame-stacked compact aim-tracking observations",
)

COMMON_SAVES = (
    "MaskablePPO/PPO model",
    "training resume state",
    "validated best checkpoint",
    "skill table snapshot",
    "CSV logs",
)

STAGES: dict[str, StageConfig] = {
    "stage0": StageConfig(
        stage_id="stage0",
        stage_number=0,
        name="Stage 0 - Environment Sanity Check",
        description="Scripted/manual validation of movement, shooting, collisions, rewards, observations, and resets before PPO.",
        trainable=False,
        required_previous_stage=None,
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=True,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=True,
        use_obstacles=True,
        use_pickups=False,
        randomize_spawns=False,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="standard",
        learning_skill="environment_sanity",
        active_skills=("environment_sanity",),
        active_actions=("move", "strafe", "rotate", "shoot", "reload"),
        locked_actions=(),
        observation_inputs=COMMON_OBS,
        reward_components=("event creation checks", "observation normalization checks"),
        success_criteria="all scripted assertions pass",
        promotion_criteria="not a PPO stage; run tools/sanity_check.py or the UI sanity option",
        validation_criteria="mechanics assertions pass on open, obstacle, and pickup layouts",
        success_threshold=1.0,
        success_window_size=1,
    ),
    "stage1": StageConfig(
        stage_id="stage1",
        stage_number=1,
        name="Stage 1 - Aim Tracking",
        description="Stationary learner rotates toward randomized moving enemies for a full episode. Only rotation actions are available through action masking.",
        trainable=True,
        required_previous_stage=None,
        learner_can_move=False,
        learner_can_strafe=False,
        learner_can_rotate=True,
        learner_can_shoot=False,
        learner_can_reload=False,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=False,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=0.0,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="orbit_no_shoot",
        learning_skill="aim_tracking",
        active_skills=("aim_tracking",),
        active_actions=("rotate",),
        locked_actions=("move", "strafe", "shoot", "reload"),
        observation_inputs=STAGE1_OBS,
        reward_components=("dense aim error", "aim-error improvement", "alignment streak", "smooth tracking", "invalid-action penalty", "full-episode completion"),
        success_criteria="full 300-step episode with mean_aim_error <= 0.35, time_on_target_ratio >= 0.60, max_aligned_steps >= 60, and invalid_actions == 0",
        promotion_criteria="rolling randomized evaluation success >= 95% for 10 consecutive checks",
        validation_criteria="same strict aim criteria on randomized seeded target spawns, speeds, and orbit directions",
        saves=COMMON_SAVES,
        carries_over=("aim_tracking",),
        allow_same_stage_resume=True,
        end_episode_on_success=False,
        max_steps=300,
        success_threshold=0.95,
        success_window_size=50,
        required_consecutive_checks=10,
    ),
    "stage2": StageConfig(
        stage_id="stage2",
        stage_number=2,
        name="Stage 2 - Shooting Timing",
        description="Stationary learner keeps the Stage 1 compact aim input and learns to shoot at high-confidence alignment moments. Movement remains disabled.",
        trainable=True,
        required_previous_stage="stage1",
        learner_can_move=False,
        learner_can_strafe=False,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=False,
        learner_auto_reload=True,
        enemy_can_move=True,
        enemy_can_shoot=False,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=0.0,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="orbit_no_shoot",
        learning_skill="shooting_timing",
        active_skills=("aim_tracking", "shooting_timing"),
        active_actions=("rotate", "shoot"),
        locked_actions=("move", "strafe", "manual_reload"),
        observation_inputs=STAGE1_OBS,
        reward_components=("light aim-retention shaping", "precise shot timing", "strong hit reward", "miss/blocked/bad-shot penalties", "tiny high-confidence hesitation penalty"),
        success_criteria="retained aim quality plus enough hits/damage, high hit rate, low bad-shot rate, and no wasted low-confidence shooting",
        promotion_criteria="validated rolling shooting-timing success and retained aim tracking",
        validation_criteria="hit-rate and bad-shot thresholds on randomized seeded aim-shoot scenarios",
        previous_skill_eval_stage_ids=("stage1",),
        previous_skill_min_success=0.60,
        saves=COMMON_SAVES,
        carries_over=("aim_tracking", "shooting_timing"),
        end_episode_on_success=True,
        max_steps=380,
        success_threshold=0.75,
        success_window_size=20,
        required_consecutive_checks=5,
    ),
    "stage3": StageConfig(
        stage_id="stage3",
        stage_number=3,
        name="Stage 3 - Basic Movement",
        description="Learner moves, strafes, and rotates while shooting is disabled. It learns reward-guided A* movement references, range control, basic combat positioning, visibility, aim stability, and boundary avoidance.",
        trainable=True,
        required_previous_stage="stage2",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=False,
        learner_can_reload=False,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=False,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=0.0,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="passive_mover",
        learning_skill="basic_movement",
        active_skills=("aim_tracking", "basic_movement"),
        active_actions=("move", "strafe", "rotate"),
        locked_actions=("shoot", "reload"),
        observation_inputs=COMMON_OBS,
        reward_components=("closed-loop chase/withdraw/stabilize distance control", "reduced A* waypoint guidance", "additive range/aim/visibility rewards", "small combined tracking bonus", "range-streak reward", "line-of-sight reward", "reduced wall-clearance penalty", "anti-freeze and anti-jitter penalties"),
        success_criteria="navigation/path-following quality plus closed-loop range control, useful movement, useful combat range, line of sight, usable aim, low blocked movement, boundary safety, and smoothness",
        promotion_criteria="validated controlled-positioning success and retained aim tracking",
        validation_criteria="A*-guided range-control, aim/visibility, and movement-quality criteria on randomized unseen spawns",
        previous_skill_eval_stage_ids=("stage1",),
        previous_skill_min_success=0.55,
        saves=COMMON_SAVES,
        carries_over=("aim_tracking", "basic_movement"),
        end_episode_on_success=True,
        max_steps=420,
        success_threshold=0.75,
        success_window_size=20,
        required_consecutive_checks=5,
    ),
    "stage4": StageConfig(
        stage_id="stage4",
        stage_number=4,
        name="Stage 4 - Dodging and Survival",
        description="Learner observes incoming slow projectiles and survives by dodging. Shooting is initially disabled.",
        trainable=True,
        required_previous_stage="stage3",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=False,
        learner_can_reload=False,
        learner_auto_reload=False,
        enemy_can_move=False,
        enemy_can_shoot=True,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="slow_shooter",
        learning_skill="dodging_survival",
        active_skills=("aim_tracking", "basic_movement", "dodging_survival"),
        active_actions=("move", "strafe", "rotate"),
        locked_actions=("shoot", "reload"),
        observation_inputs=COMMON_OBS,
        reward_components=("damage-taken penalty", "projectile dodge reward", "survival with movement", "passive hiding penalty"),
        success_criteria="low damage taken, high survival time, and sufficient projectile avoidance rate",
        promotion_criteria="validated dodging success and retained basic movement/aim",
        validation_criteria="survival and avoidance criteria on unseen projectile lanes/spawns",
        previous_skill_eval_stage_ids=("stage1", "stage3"),
        previous_skill_min_success=0.50,
        saves=COMMON_SAVES,
        carries_over=("aim_tracking", "basic_movement", "dodging_survival"),
        max_steps=460,
        success_threshold=0.70,
        success_window_size=20,
        required_consecutive_checks=5,
    ),
    "stage5": StageConfig(
        stage_id="stage5",
        stage_number=5,
        name="Stage 5 - Moving Shooting",
        description="Learner combines movement, rotation, and shooting through a 5A/5B/5C moving-shooting curriculum. Reload is simplified with auto-reload so Stage 5 focuses on hitting while moving, not resource timing.",
        trainable=True,
        required_previous_stage="stage4",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=False,
        learner_auto_reload=True,
        enemy_can_move=True,
        enemy_can_shoot=True,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="easy_duel",
        learning_skill="moving_shooting",
        active_skills=("aim_tracking", "shooting_timing", "basic_movement", "dodging_survival", "moving_shooting"),
        active_actions=("move", "strafe", "rotate", "shoot"),
        locked_actions=("reload",),
        observation_inputs=COMMON_OBS,
        reward_components=("moving-hit reward", "damage dealt", "moving hit rate", "small gated aim support", "stationary-shot penalty", "damage/death penalty"),
        success_criteria="minimum moving hits, damage dealt while moving, hit quality, movement activity, and phase-dependent survival",
        promotion_criteria="5A moving-shot foundation, 5B light-pressure moving shooting, then 5C final validation",
        validation_criteria="phase-aware moving-hit and damage metrics on randomized unseen spawns; official pass only at final 5C phase",
        previous_skill_eval_stage_ids=("stage1", "stage2"),
        previous_skill_min_success=0.50,
        saves=COMMON_SAVES,
        carries_over=("aim_tracking", "shooting_timing", "basic_movement", "dodging_survival", "moving_shooting"),
        max_steps=500,
        success_threshold=0.70,
        success_window_size=20,
        required_consecutive_checks=5,
    ),
    "stage6": StageConfig(
        stage_id="stage6",
        stage_number=6,
        name="Stage 6 - Open-Arena Combat",
        description="Full open-arena duel without obstacles or pickups, trained through 6A/6B/6C/6D: light pressure, medium pressure, hard bridge, then final full open-arena combat.",
        trainable=True,
        required_previous_stage="stage5",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=True,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=True,
        use_obstacles=False,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=False,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="moderate_duel",
        learning_skill="open_arena_combat",
        active_skills=("aim_tracking", "shooting_timing", "basic_movement", "dodging_survival", "moving_shooting", "open_arena_combat"),
        active_actions=("move", "strafe", "rotate", "shoot", "reload"),
        locked_actions=(),
        observation_inputs=COMMON_OBS,
        reward_components=("kill reward", "death penalty", "damage dealt/taken ratio", "manual reload sanity", "projectile-hit control", "A* combat-position reference", "shot economy", "anti-passivity"),
        success_criteria="phase-aware open-arena win, damage ratio, hit rate, low damage taken, reload sanity, and no severe passive hiding",
        promotion_criteria="6A light-pressure duel, 6B medium pressure, 6C hard bridge, then 6D final full open-arena combat",
        validation_criteria="phase-aware open-arena combat metrics on unseen spawn positions and enemy movement seeds; official pass only at final 6D phase",
        previous_skill_eval_stage_ids=("stage5",),
        previous_skill_min_success=0.50,
        saves=COMMON_SAVES,
        carries_over=("all open-arena combat skills",),
        max_steps=560,
        success_threshold=0.68,
        success_window_size=20,
        required_consecutive_checks=5,
    ),
    "stage7": StageConfig(
        stage_id="stage7",
        stage_number=7,
        name="Stage 7 - Obstacle and Cover Combat",
        description="Phased obstacle-cover combat through 7A/7B/7C/7D/7E: one obstacle stationary enemy, two obstacles stationary enemy, three obstacles with moving enemy, four-obstacle medium-pressure bridge, then four-obstacle final validation. No pickups yet.",
        trainable=True,
        required_previous_stage="stage6",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=True,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=True,
        use_obstacles=True,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=True,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="standard",
        learning_skill="obstacle_cover_combat",
        active_skills=("aim_tracking", "shooting_timing", "basic_movement", "dodging_survival", "moving_shooting", "open_arena_combat", "obstacle_cover_combat"),
        active_actions=("move", "strafe", "rotate", "shoot", "reload"),
        locked_actions=(),
        observation_inputs=COMMON_OBS,
        reward_components=("blocked-shot penalty", "blocked-movement penalty", "A* obstacle path reference", "limited line-of-sight recovery reward", "cover survival", "phase-aware obstacle count"),
        success_criteria="phase-aware obstacle win while limiting blocked movement, blocked shots, damage taken, and passive hiding",
        promotion_criteria="7A one obstacle stationary enemy, 7B two obstacles stationary enemy, 7C three obstacles with moving enemy, 7D four-obstacle medium-pressure bridge, then 7E four-obstacle final validation",
        validation_criteria="official pass only at final 7E on unseen four-obstacle and spawn variants",
        previous_skill_eval_stage_ids=("stage5", "stage6"),
        previous_skill_min_success=0.45,
        saves=COMMON_SAVES,
        carries_over=("all combat and cover skills",),
        max_steps=1200,
        success_threshold=0.35,
        success_window_size=20,
        required_consecutive_checks=3,
    ),
    "stage8": StageConfig(
        stage_id="stage8",
        stage_number=8,
        name="Stage 8 - Reload Endurance Combat",
        description="Simplified final curriculum stage. Health regain and map pickups are disabled. The normal Glock magazine is preserved. Reload pressure is created naturally by sequential enemy waves: only one opponent is active at a time, and the learner keeps its HP, ammo, and reload charges across waves.",
        trainable=True,
        required_previous_stage="stage7",
        learner_can_move=True,
        learner_can_strafe=True,
        learner_can_rotate=True,
        learner_can_shoot=True,
        learner_can_reload=True,
        learner_auto_reload=False,
        enemy_can_move=True,
        enemy_can_shoot=True,
        use_obstacles=True,
        use_pickups=False,
        randomize_spawns=True,
        randomize_obstacles=True,
        randomize_pickups=False,
        enemy_move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        enemy_fire_rate=GLOCK_FIRE_RATE,
        enemy_reload_time=1.2,
        enemy_projectile_damage=GLOCK_DAMAGE_HP,
        enemy_projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        enemy_behavior_profile="standard",
        learning_skill="tactical_reload_management",
        active_skills=("aim_tracking", "shooting_timing", "basic_movement", "dodging_survival", "moving_shooting", "open_arena_combat", "obstacle_cover_combat", "tactical_reload_management"),
        active_actions=("move", "strafe", "rotate", "shoot", "reload"),
        locked_actions=(),
        observation_inputs=COMMON_OBS,
        reward_components=("shared combat reward", "A* obstacle path reference", "shooting discipline penalties", "reload-quality reward", "limited reload reserves", "sequential enemy waves"),
        success_criteria="phase-aware endurance combat: clear enough sequential enemy waves, survive, reload at least once when useful, and avoid spam shooting/reloading or wall-stuck behavior",
        promotion_criteria="8A open-arena stationary waves, 8B open-arena light-fire waves, 8C one-obstacle waves, 8D multi-obstacle waves, then 8E final four-obstacle endurance validation",
        validation_criteria="official pass only at final 8E with sequential waves, obstacle combat, survival, reload quality, clean shooting, and reasonable movement/aim checks",
        previous_skill_eval_stage_ids=("stage1", "stage2", "stage5", "stage6", "stage7"),
        previous_skill_min_success=0.45,
        saves=COMMON_SAVES,
        carries_over=("complete PPO duel bot",),
        max_steps=2200,
        success_threshold=0.35,
        success_window_size=20,
        required_consecutive_checks=3,
    ),
}

STAGE_ORDER = [f"stage{i}" for i in range(0, 9)]
TRAINABLE_STAGE_ORDER = [stage_id for stage_id in STAGE_ORDER if STAGES[stage_id].trainable]

STAGE_ALIASES = {
    "stageA": "stage1",
    "stageB": "stage2",
    "stageC": "stage6",
    "stageD": "stage3",
    "stageE": "stage5",
    "stageF": "stage7",
    "stageG": "stage8",
}


def normalize_stage_id(stage_id: str) -> str:
    return STAGE_ALIASES.get(stage_id, stage_id)


def get_stage(stage_id: str) -> StageConfig:
    canonical = normalize_stage_id(stage_id)
    if canonical not in STAGES:
        raise ValueError(f"Unknown stage_id: {stage_id}")
    return STAGES[canonical]


def stage_index(stage_id: str) -> int:
    return STAGE_ORDER.index(normalize_stage_id(stage_id))


def next_stage_id(stage_id: str) -> str | None:
    index = stage_index(stage_id)
    if index + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[index + 1]


def ensure_runtime_directories() -> None:
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for stage_id in STAGE_ORDER:
        stage = STAGES[stage_id]
        stage.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stage.log_dir.mkdir(parents=True, exist_ok=True)
