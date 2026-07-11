from __future__ import annotations

import math
import random

import pygame

from config.game_constants import GLOCK_DAMAGE_HP, GLOCK_PROJECTILE_SPEED_PX_PER_SEC, PLAYER_MAX_HEALTH, PLAYER_MOVE_SPEED_PX_PER_SEC
from config.stages import get_stage
from game.agent import Agent, AgentCommand
from game.bot import ModelBot, StockBot
from game.maps import build_layout
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


def _read_human_command(player: Agent) -> AgentCommand:
    keys = pygame.key.get_pressed()
    mouse_x, mouse_y = pygame.mouse.get_pos()

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
    last_result = ""
    countdown = 0

    while not renderer.quit_requested and not renderer.back_requested:
        if not renderer.sync_runtime(
            lambda: _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, last_result)
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
            _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, last_result)
            continue

        player_command = _read_human_command(learner)
        events = learner.apply_command(player_command, world)
        enemy_command = enemy_controller.tick()
        events.extend(enemy.apply_command(enemy_command, world))
        events.extend(world.tick())

        for event in events:
            if event.kind == "kill":
                if event.agent_id == learner.agent_id:
                    last_result = "You won the round"
                elif event.agent_id == enemy.agent_id:
                    last_result = "Enemy won the round"
                countdown = 45

        _draw_match_frame(renderer, world, learner, enemy, mode_label, round_index, last_result)


def _draw_match_frame(
    renderer: Renderer,
    world: World,
    learner: Agent,
    enemy: Agent,
    mode_label: str,
    round_index: int,
    last_result: str,
) -> None:
    hud_lines = [
        f"Mode: {mode_label}",
        f"Round: {round_index}",
        f"Player HP {learner.health:.0f}/{learner.max_health:.0f} | Ammo {learner.ammo}/{learner.max_ammo}",
        f"Enemy  HP {enemy.health:.0f}/{enemy.max_health:.0f} | Ammo {enemy.ammo}/{enemy.max_ammo}",
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


def run_human_vs_model_bot(stage_id: str) -> None:
    stage = get_stage(stage_id)
    model_path = stage.best_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {stage_id}: {model_path}")

    layout = build_layout(stage_id)
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
        can_move=stage.learner_can_move,
        move_speed=PLAYER_MOVE_SPEED_PX_PER_SEC,
        max_health=PLAYER_MAX_HEALTH,
        max_ammo=20,
        fire_rate=2.0,
        reload_time=1.2,
        projectile_damage=GLOCK_DAMAGE_HP,
        projectile_speed=GLOCK_PROJECTILE_SPEED_PX_PER_SEC,
        auto_reload_when_empty=stage.learner_auto_reload,
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

    _run_round_loop(
        title=stage.name,
        world=world,
        learner=learner,
        enemy=enemy,
        enemy_controller=bot,
        renderer=renderer,
        mode_label=f"Test vs {stage.best_model_filename}",
    )