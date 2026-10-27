from __future__ import annotations

import math
import random

import pygame

from config.game_constants import GLOCK_DAMAGE_HP, GLOCK_PROJECTILE_SPEED_PX_PER_SEC, PLAYER_MAX_HEALTH, PLAYER_MOVE_SPEED_PX_PER_SEC
from config.stages import get_stage
from game.agent import Agent, AgentCommand
from game.bot import ModelBot, StockBot
from game.evaluation_bots import create_evaluation_bot
from game.maps import RectObstacle, build_layout    
from game.world import World
from rendering.renderer import Renderer


PLAY_STAGE_ID = "stage8"


def _spawn_pair(world: World, learner: Agent, enemy: Agent) -> None:
    spawn = random.choice(world.layout.spawns)
    learner.position.x = spawn.learner_x
    learner.position.y = spawn.learner_y
    learner.heading = spawn.learner_heading
    enemy.position.x = spawn.enemy_x
    enemy.position.y = spawn.enemy_y
    enemy.heading = spawn.enemy_heading


def _read_human_command(player: Agent, renderer: Renderer | None = None) -> AgentCommand:
    keys = pygame.key.get_pressed()
    mouse_x, mouse_y = pygame.mouse.get_pos()

    use_center_aim = (
        renderer is not None
        and getattr(renderer, "follow_camera_enabled", False)
        and getattr(renderer, "camera_target_agent_id", None) == player.agent_id
    )

    if use_center_aim:
        view_rect = getattr(
            renderer,
            "camera_view_rect",
            pygame.Rect(0, 0, renderer.screen.get_width(), renderer.screen.get_height()),
        )

        
        mouse_x = max(view_rect.left, min(mouse_x, view_rect.right - 1))
        mouse_y = max(view_rect.top, min(mouse_y, view_rect.bottom - 1))

        
        dx = mouse_x - view_rect.centerx
        dy = mouse_y - view_rect.centery
    else:
        if renderer is not None and hasattr(renderer, "screen_to_world"):
            mouse_x, mouse_y = renderer.screen_to_world(mouse_x, mouse_y)
        dx = mouse_x - player.position.x
        dy = mouse_y - player.position.y

    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        player.heading = math.atan2(dy, dx)

    move_x = 0.0
    move_y = 0.0

    if keys[pygame.K_w]:
        move_y -= 1.0
    if keys[pygame.K_s]:
        move_y += 1.0
    if keys[pygame.K_a]:
        move_x -= 1.0
    if keys[pygame.K_d]:
        move_x += 1.0

    mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)

    return AgentCommand(
        shoot=bool(mouse_buttons[0]),
        reload=keys[pygame.K_r],
        use_world_movement=True,
        world_move_x=move_x,
        world_move_y=move_y,
    )


def _reloads_label(agent: Agent) -> str:
    charges = getattr(agent, "reload_charges", None)
    if charges is None:
        return "inf"
    return str(int(charges))


def _is_rl_model_key(bot_key: str) -> bool:
    return bot_key == "stage8_8d"


def _is_scripted_eval_key(bot_key: str) -> bool:
    return bot_key in {"fsm", "behavior_tree", "cs2d_style"}


def _bot_key_to_stage_id(bot_key: str) -> str:
    if bot_key in {"stage8_8d", "fsm", "behavior_tree", "cs2d_style"}:
        return "stage8"
    raise ValueError(f"Unknown bot key: {bot_key}")


def _bot_key_to_label(bot_key: str) -> str:
    if bot_key == "stage8_8d":
        return "Stage 8D RL bot"
    if bot_key == "fsm":
        return "FSM bot"
    if bot_key == "behavior_tree":
        return "Behavior Tree bot"
    if bot_key == "cs2d_style":
        return "CS2D-style bot"
    return bot_key


def _stage8_8d_model_path(stage):
    return stage.checkpoint_dir / "best_stage8_8d_level_04.zip"


def _bot_key_to_model_path(bot_key: str):
    stage_id = _bot_key_to_stage_id(bot_key)
    stage = get_stage(stage_id)

    if bot_key == "stage8_8d":
        return _stage8_8d_model_path(stage)

    return stage.best_model_path


def _build_layout_for_bot_key(bot_key: str):
    stage_id = _bot_key_to_stage_id(bot_key)

    if bot_key in {"stage8_8d", "fsm", "behavior_tree", "cs2d_style"}:
        return build_layout(stage_id, validation=True, stage8_phase_index=4)

    return build_layout(stage_id)

def _distance_point_to_obstacle(px: float, py: float, obstacle: RectObstacle) -> float:
    closest_x = max(obstacle.x, min(px, obstacle.x + obstacle.w))
    closest_y = max(obstacle.y, min(py, obstacle.y + obstacle.h))
    return math.hypot(px - closest_x, py - closest_y)


def _obstacles_overlap_with_margin(a: RectObstacle, b: RectObstacle, margin: float = 48.0) -> bool:
    m = int(margin)
    rect_a = pygame.Rect(
        int(a.x) - m,
        int(a.y) - m,
        int(a.w) + 2 * m,
        int(a.h) + 2 * m,
    )
    rect_b = pygame.Rect(int(b.x), int(b.y), int(b.w), int(b.h))
    return rect_a.colliderect(rect_b)


def _add_final_eval_obstacles(layout) -> None:
    
    base_obstacles = list(getattr(layout, "obstacles", []))

    spawn_points: list[tuple[float, float]] = []
    for spawn in layout.spawns:
        spawn_points.append((spawn.learner_x, spawn.learner_y))
        spawn_points.append((spawn.enemy_x, spawn.enemy_y))

    candidate_obstacles = [
        # Safe upper/lower cover on the right half.
        RectObstacle(625, 85, 95, 85),
        RectObstacle(625, 625, 95, 85),

        # Mid-map side cover. These are far enough from the Stage 8D
        # validation spawns but still add LOS decisions around the center.
        RectObstacle(680, 340, 85, 90),
        RectObstacle(300, 360, 80, 90),

        # Backup candidates. The filter below keeps only safe ones.
        RectObstacle(760, 455, 80, 90),
        RectObstacle(265, 500, 80, 80),
        RectObstacle(790, 260, 75, 85),
    ]

    safe_extras: list[RectObstacle] = []

    for obstacle in candidate_obstacles:
        too_close_to_spawn = any(
            _distance_point_to_obstacle(spawn_x, spawn_y, obstacle) < 175.0
            for spawn_x, spawn_y in spawn_points
        )
        if too_close_to_spawn:
            continue

        overlaps_existing = any(
            _obstacles_overlap_with_margin(obstacle, existing, margin=45.0)
            for existing in base_obstacles + safe_extras
        )
        if overlaps_existing:
            continue

        safe_extras.append(obstacle)
        if len(safe_extras) >= 4:
            break

    layout.obstacles.clear()
    layout.obstacles.extend(base_obstacles + safe_extras)


def _reset_agent_for_respawn(agent: Agent, reload_charges: int | None = None) -> None:
    agent.health = agent.max_health
    agent.ammo = agent.max_ammo

    if reload_charges is not None and hasattr(agent, "reload_charges"):
        agent.reload_charges = reload_charges

    if hasattr(agent, "is_reloading"):
        agent.is_reloading = False
    if hasattr(agent, "reloading"):
        agent.reloading = False
    if hasattr(agent, "reload_timer"):
        agent.reload_timer = 0.0
    if hasattr(agent, "fire_cooldown"):
        agent.fire_cooldown = 0.0
    if hasattr(agent, "alive"):
        agent.alive = True


def _respawn_single_agent(
    world: World,
    agent: Agent,
    other: Agent,
    as_learner_spawn: bool,
    reload_charges: int | None = None,
    avoid_position: tuple[float, float] | None = None,
) -> None:
    spawn_options = list(world.layout.spawns)
    random.shuffle(spawn_options)

    best_spawn = spawn_options[0]
    best_score = -1e9

    chosen = spawn_options[0]

    for spawn in spawn_options:
        if as_learner_spawn:
            x = spawn.learner_x
            y = spawn.learner_y
        else:
            x = spawn.enemy_x
            y = spawn.enemy_y

        dist_from_other = math.hypot(x - other.position.x, y - other.position.y)

        if avoid_position is None:
            dist_from_death = 9999.0
        else:
            dist_from_death = math.hypot(x - avoid_position[0], y - avoid_position[1])

        score = dist_from_other + 0.5 * dist_from_death

        if dist_from_other < 220.0:
            score -= 500.0

        if dist_from_death < 160.0:
            score -= 1000.0

        if score > best_score:
            best_score = score
            best_spawn = spawn

        if dist_from_other >= 220.0 and dist_from_death >= 160.0:
            chosen = spawn
            break
    else:
        chosen = best_spawn

    if as_learner_spawn:
        agent.position.x = chosen.learner_x
        agent.position.y = chosen.learner_y
        agent.heading = chosen.learner_heading
    else:
        agent.position.x = chosen.enemy_x
        agent.position.y = chosen.enemy_y
        agent.heading = chosen.enemy_heading

    _reset_agent_for_respawn(agent, reload_charges=reload_charges)


def _make_controller(bot_key: str, agent: Agent, world: World, target_id: str):
    if _is_rl_model_key(bot_key):
        model_path = _bot_key_to_model_path(bot_key)
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found for {bot_key}: {model_path}")

        return ModelBot(
            model_path=model_path,
            stage_id=_bot_key_to_stage_id(bot_key),
            agent=agent,
            world=world,
            target_id=target_id,
        )

    if _is_scripted_eval_key(bot_key):
        return create_evaluation_bot(
            kind=bot_key,
            agent=agent,
            world=world,
            target_id=target_id,
            rng=random.Random(),
        )

    raise ValueError(f"Unknown bot key: {bot_key}")


def _run_round_loop(
    title: str,
    world: World,
    learner: Agent,
    enemy: Agent,
    enemy_controller,
    renderer: Renderer,
    mode_label: str,
) -> None:
    round_index = 1
    player_score = 0
    enemy_score = 0
    last_result = ""
    countdown = 0

    while not renderer.quit_requested and not renderer.back_requested:
        score_text = f"Score Player:{player_score} Enemy:{enemy_score}"
        status_text = f"{last_result} | {score_text}" if last_result else score_text

        if not renderer.sync_runtime(
            lambda: _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, status_text)
        ):
            break

        if countdown > 0:
            countdown -= 1
            if countdown == 0:
                world.reset_round()
                _spawn_pair(world, learner, enemy)
                if hasattr(enemy_controller, "reset_memory"):
                    enemy_controller.reset_memory()
                last_result = ""
                round_index += 1

            score_text = f"Score Player:{player_score} Enemy:{enemy_score}"
            status_text = f"{last_result} | {score_text}" if last_result else score_text
            _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, status_text)
            continue

        player_command = _read_human_command(learner, renderer=renderer)
        events = learner.apply_command(player_command, world)

        enemy_command = enemy_controller.tick()
        events.extend(enemy.apply_command(enemy_command, world))
        events.extend(world.tick())

        for event in events:
            if event.kind == "kill":
                if event.agent_id == learner.agent_id:
                    player_score += 1
                    last_result = "Player killed enemy"
                elif event.agent_id == enemy.agent_id:
                    enemy_score += 1
                    last_result = "Enemy killed player"
                countdown = 45

        score_text = f"Score Player:{player_score} Enemy:{enemy_score}"
        status_text = f"{last_result} | {score_text}" if last_result else score_text
        _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, status_text)


def _run_bot_vs_bot_round_loop(
    title: str,
    world: World,
    blue_agent: Agent,
    red_agent: Agent,
    blue_controller,
    red_controller,
    renderer: Renderer,
    mode_label: str,
    blue_reload_charges: int | None = None,
    red_reload_charges: int | None = None,
) -> None:
    round_index = 1
    blue_score = 0
    red_score = 0
    last_result = ""
    respawn_countdown = 0
    pending_respawns: set[str] = set()
    pending_avoid_positions: dict[str, tuple[float, float]] = {}

    while not renderer.quit_requested and not renderer.back_requested:
        score_text = f"Score Blue:{blue_score} Red:{red_score}"
        status_text = f"{last_result} | {score_text}" if last_result else score_text

        if not renderer.sync_runtime(
            lambda: _draw_match_frame(
                renderer,
                world,
                blue_agent,
                red_agent,
                mode_label,
                round_index,
                status_text,
                learner_label="Blue",
                enemy_label="Red",
            )
        ):
            break

        if respawn_countdown > 0:
            respawn_countdown -= 1

            if respawn_countdown == 0:
                respawned_blue = "blue" in pending_respawns
                respawned_red = "red" in pending_respawns

                if respawned_blue:
                    _respawn_single_agent(
                        world,
                        blue_agent,
                        red_agent,
                        as_learner_spawn=True,
                        reload_charges=blue_reload_charges,
                        avoid_position=pending_avoid_positions.get("blue"),
                    )
                    if hasattr(blue_controller, "reset_memory"):
                        blue_controller.reset_memory()

                if respawned_red:
                    _respawn_single_agent(
                        world,
                        red_agent,
                        blue_agent,
                        as_learner_spawn=False,
                        reload_charges=red_reload_charges,
                        avoid_position=pending_avoid_positions.get("red"),
                    )
                    if hasattr(red_controller, "reset_memory"):
                        red_controller.reset_memory()

                # Safety correction: if both respawned too close, move Red again.
                if respawned_blue and respawned_red:
                    dist = math.hypot(
                        blue_agent.position.x - red_agent.position.x,
                        blue_agent.position.y - red_agent.position.y,
                    )
                    if dist < 220.0:
                        _respawn_single_agent(
                            world,
                            red_agent,
                            blue_agent,
                            as_learner_spawn=False,
                            reload_charges=red_reload_charges,
                            avoid_position=pending_avoid_positions.get("red"),
                        )
                        if hasattr(red_controller, "reset_memory"):
                            red_controller.reset_memory()

                pending_respawns.clear()
                pending_avoid_positions.clear()
                last_result = ""

            score_text = f"Score Blue:{blue_score} Red:{red_score}"
            status_text = f"{last_result} | {score_text}" if last_result else score_text

            _draw_match_frame(
                renderer,
                world,
                blue_agent,
                red_agent,
                mode_label,
                round_index,
                status_text,
                learner_label="Blue",
                enemy_label="Red",
            )
            continue

        blue_command = blue_controller.tick()
        red_command = red_controller.tick()

        events = blue_agent.apply_command(blue_command, world)
        events.extend(red_agent.apply_command(red_command, world))
        events.extend(world.tick())

        blue_killed_red = False
        red_killed_blue = False

        for event in events:
            if event.kind == "kill":
                # In this project, event.agent_id refers to the killer agent.
                if event.agent_id == blue_agent.agent_id:
                    blue_killed_red = True
                elif event.agent_id == red_agent.agent_id:
                    red_killed_blue = True

        
        blue_dead = not blue_agent.is_alive
        red_dead = not red_agent.is_alive

        if blue_killed_red:
            blue_score += 1
            pending_respawns.add("red")
            pending_avoid_positions["red"] = (red_agent.position.x, red_agent.position.y)

        if red_killed_blue:
            red_score += 1
            pending_respawns.add("blue")
            pending_avoid_positions["blue"] = (blue_agent.position.x, blue_agent.position.y)

        if blue_dead and "blue" not in pending_respawns:
            pending_respawns.add("blue")
            pending_avoid_positions["blue"] = (blue_agent.position.x, blue_agent.position.y)

        if red_dead and "red" not in pending_respawns:
            pending_respawns.add("red")
            pending_avoid_positions["red"] = (red_agent.position.x, red_agent.position.y)

        if blue_killed_red and red_killed_blue:
            last_result = "Both bots killed each other"
            respawn_countdown = 45
        elif blue_killed_red:
            last_result = "Blue killed Red"
            respawn_countdown = 45
        elif red_killed_blue:
            last_result = "Red killed Blue"
            respawn_countdown = 45
        elif pending_respawns:
            last_result = "Bot death detected"
            respawn_countdown = 45

        score_text = f"Score Blue:{blue_score} Red:{red_score}"
        status_text = f"{last_result} | {score_text}" if last_result else score_text

        _draw_match_frame(
            renderer,
            world,
            blue_agent,
            red_agent,
            mode_label,
            round_index,
            status_text,
            learner_label="Blue",
            enemy_label="Red",
        )


def _run_bot_vs_bot_headless_loop(
    world: World,
    blue_agent: Agent,
    red_agent: Agent,
    blue_controller,
    red_controller,
    blue_reload_charges: int | None,
    red_reload_charges: int | None,
    max_steps: int = 9000,
) -> None:
    blue_score = 0
    red_score = 0
    simultaneous_kills = 0

    for step in range(1, max_steps + 1):
        blue_command = blue_controller.tick()
        red_command = red_controller.tick()

        events = blue_agent.apply_command(blue_command, world)
        events.extend(red_agent.apply_command(red_command, world))
        events.extend(world.tick())

        blue_killed_red = False
        red_killed_blue = False

        for event in events:
            if event.kind == "kill":
                # In this project, event.agent_id refers to the killer agent.
                if event.agent_id == blue_agent.agent_id:
                    blue_killed_red = True
                elif event.agent_id == red_agent.agent_id:
                    red_killed_blue = True

        blue_dead = not blue_agent.is_alive
        red_dead = not red_agent.is_alive

        respawn_blue = False
        respawn_red = False
        blue_death_position = (blue_agent.position.x, blue_agent.position.y)
        red_death_position = (red_agent.position.x, red_agent.position.y)

        if blue_killed_red:
            blue_score += 1
            respawn_red = True

        if red_killed_blue:
            red_score += 1
            respawn_blue = True

        if blue_dead:
            respawn_blue = True

        if red_dead:
            respawn_red = True

        if respawn_blue and respawn_red:
            simultaneous_kills += 1

        if respawn_blue:
            _respawn_single_agent(
                world,
                blue_agent,
                red_agent,
                as_learner_spawn=True,
                reload_charges=blue_reload_charges,
                avoid_position=blue_death_position,
            )
            if hasattr(blue_controller, "reset_memory"):
                blue_controller.reset_memory()

        if respawn_red:
            _respawn_single_agent(
                world,
                red_agent,
                blue_agent,
                as_learner_spawn=False,
                reload_charges=red_reload_charges,
                avoid_position=red_death_position,
            )
            if hasattr(red_controller, "reset_memory"):
                red_controller.reset_memory()

        # Safety correction: if both respawned too close, move Red again.
        if respawn_blue and respawn_red:
            dist = math.hypot(
                blue_agent.position.x - red_agent.position.x,
                blue_agent.position.y - red_agent.position.y,
            )
            if dist < 220.0:
                _respawn_single_agent(
                    world,
                    red_agent,
                    blue_agent,
                    as_learner_spawn=False,
                    reload_charges=red_reload_charges,
                    avoid_position=red_death_position,
                )
                if hasattr(red_controller, "reset_memory"):
                    red_controller.reset_memory()

    print()
    print("Headless evaluation finished")
    print(f"Steps: {max_steps}")
    print(f"Score Blue: {blue_score}")
    print(f"Score Red: {red_score}")
    print(f"Simultaneous kills: {simultaneous_kills}")
    print()


def _draw_match_frame(
    renderer: Renderer,
    world: World,
    learner: Agent,
    enemy: Agent,
    mode_label: str,
    round_index: int,
    last_result: str,
    learner_label: str = "Player",
    enemy_label: str = "Enemy",
) -> None:
    hud_lines = [
        f"Mode: {mode_label}",
        f"Round: {round_index}",
        f"{learner_label} HP {learner.health:.0f}/{learner.max_health:.0f} | Ammo {learner.ammo}/{learner.max_ammo} | Reloads {_reloads_label(learner)}",
        f"{enemy_label} HP {enemy.health:.0f}/{enemy.max_health:.0f} | Ammo {enemy.ammo}/{enemy.max_ammo} | Reloads {_reloads_label(enemy)}",
        "Move: WASD | Aim: mouse | Shoot: left mouse | Reload: R",
        "Pause: P | Back: ESC",
    ]
    if last_result:
        hud_lines.append(last_result)
    renderer.draw_world(world, hud_lines=hud_lines, title="Top-down shooter")


def run_human_vs_stock_bot() -> None:
    stage = get_stage(PLAY_STAGE_ID)
    layout = build_layout(PLAY_STAGE_ID)
    world = World(layout)

    learner = Agent(
        agent_id="learner",
        can_move=True,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
    )
    enemy = Agent(
        agent_id="enemy",
        can_move=True,
        move_speed=stage.enemy_move_speed,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=stage.enemy_fire_rate,
        reload_time=stage.enemy_reload_time,
        projectile_damage=stage.enemy_projectile_damage,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
    )

    world.add_agent(learner)
    world.add_agent(enemy)
    world.reset_round()
    _spawn_pair(world, learner, enemy)

    bot = StockBot(enemy, world, target_id="learner", profile=stage.enemy_behavior_profile)
    renderer = Renderer(layout.width, layout.height, "RL Shooter - Play 1v1", target_fps=60)
    renderer.reset_runtime_flags()

    _run_round_loop(
        title=stage.name,
        world=world,
        learner=learner,
        enemy=enemy,
        enemy_controller=bot,
        renderer=renderer,
        mode_label="Play 1v1 vs stock bot",
    )


def run_human_vs_model_bot(stage_id: str, use_follow_camera: bool = False) -> None:
    stage = get_stage(stage_id)
    test_stage8_8d = stage.stage_id == "stage8"

    model_path = (
        _stage8_8d_model_path(stage)
        if test_stage8_8d
        else stage.best_model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {stage_id}: {model_path}")

    layout = (
        build_layout(stage_id, validation=True, stage8_phase_index=4)
        if test_stage8_8d
        else build_layout(stage_id)
    )

    if test_stage8_8d:
        _add_final_eval_obstacles(layout)

    world = World(layout)

    learner = Agent(
        agent_id="learner",
        can_move=True,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        reload_charges=3 if test_stage8_8d else None,
    )
    enemy = Agent(
        agent_id="enemy",
        can_move=stage.learner_can_move,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        auto_reload_when_empty=stage.learner_auto_reload,
        reload_charges=3 if test_stage8_8d else None,
    )

    world.add_agent(learner)
    world.add_agent(enemy)
    world.reset_round()
    _spawn_pair(world, learner, enemy)

    bot = ModelBot(
        model_path=model_path,
        stage_id=stage_id,
        agent=enemy,
        world=world,
        target_id="learner",
    )
    renderer = Renderer(layout.width, layout.height, f"RL Shooter - Test {stage.name}", target_fps=60)
    renderer.reset_runtime_flags()

    if test_stage8_8d and use_follow_camera:
        renderer.enable_follow_camera(
            target_agent_id="learner",
            world_span=380.0,
            square_view=True,
        )
        pygame.mouse.set_pos(
            renderer.screen.get_width() // 2,
            renderer.screen.get_height() // 2,
        )
    else:
        renderer.disable_follow_camera()

    _run_round_loop(
        title=stage.name,
        world=world,
        learner=learner,
        enemy=enemy,
        enemy_controller=bot,
        renderer=renderer,
        mode_label="Human player vs Stage 8D RL bot" if test_stage8_8d else f"Test vs {model_path.name}",
    )


def run_model_vs_model_bot(
    bot_a_key: str = "stage8_8d",
    bot_b_key: str = "fsm",
    visible: bool = True,
) -> None:
    blue_stage_id = _bot_key_to_stage_id(bot_a_key)
    red_stage_id = _bot_key_to_stage_id(bot_b_key)

    if blue_stage_id != red_stage_id:
        raise ValueError("Evaluation testing requires both bots to use the same stage/layout.")

    stage = get_stage(blue_stage_id)
    layout = _build_layout_for_bot_key(bot_a_key)
    _add_final_eval_obstacles(layout)
    world = World(layout)

    blue_reload_charges = 3
    red_reload_charges = 3

    blue_agent = Agent(
        agent_id="learner",
        can_move=stage.learner_can_move,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        auto_reload_when_empty=stage.learner_auto_reload,
        reload_charges=blue_reload_charges,
    )

    red_agent = Agent(
        agent_id="enemy",
        can_move=True,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        auto_reload_when_empty=False,
        reload_charges=red_reload_charges,
    )

    world.add_agent(blue_agent)
    world.add_agent(red_agent)
    world.reset_round()
    _spawn_pair(world, blue_agent, red_agent)

    blue_controller = _make_controller(
        bot_key=bot_a_key,
        agent=blue_agent,
        world=world,
        target_id="enemy",
    )

    red_controller = _make_controller(
        bot_key=bot_b_key,
        agent=red_agent,
        world=world,
        target_id="learner",
    )

    blue_label = _bot_key_to_label(bot_a_key)
    red_label = _bot_key_to_label(bot_b_key)

    if not visible:
        print()
        print(f"Running headless evaluation: Blue {blue_label} vs Red {red_label}")
        _run_bot_vs_bot_headless_loop(
            world=world,
            blue_agent=blue_agent,
            red_agent=red_agent,
            blue_controller=blue_controller,
            red_controller=red_controller,
            blue_reload_charges=blue_reload_charges,
            red_reload_charges=red_reload_charges,
            max_steps=9000,
        )
        return

    renderer = Renderer(layout.width, layout.height, f"RL Shooter - Blue {blue_label} vs Red {red_label}", target_fps=60)
    renderer.reset_runtime_flags()

    _run_bot_vs_bot_round_loop(
        title=stage.name,
        world=world,
        blue_agent=blue_agent,
        red_agent=red_agent,
        blue_controller=blue_controller,
        red_controller=red_controller,
        renderer=renderer,
        mode_label=f"Blue: {blue_label} vs Red: {red_label}",
        blue_reload_charges=blue_reload_charges,
        red_reload_charges=red_reload_charges,
    )