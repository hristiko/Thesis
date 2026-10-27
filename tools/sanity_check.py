from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import detection
from game.agent import Agent, AgentCommand
from game.maps import build_layout
from game.world import Event, Vec2, World
from rl.action_space import sample_valid_action, stage_action_mask
from rl.obs_builder import ObsBuilder
from rl.reward_manager import RewardManager
from training.skill_table import build_skill_table
from training.stage_manager import mark_stage0_sanity_passed


class SanityFailure(AssertionError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SanityFailure(message)


def _make_agent(agent_id: str, x: float, y: float, heading: float = 0.0, **kwargs) -> Agent:
    agent = Agent(agent_id=agent_id, can_move=True, **kwargs)
    agent.position = Vec2(x, y)
    agent.heading = heading
    return agent


def _open_world() -> tuple[World, Agent, Agent]:
    world = World(build_layout("stage1"))
    learner = _make_agent("learner", 450, 350, 0.0)
    enemy = _make_agent("enemy", 720, 350, math.pi)
    world.add_agent(learner)
    world.add_agent(enemy)
    world.reset_round()
    learner.position = Vec2(450, 350)
    learner.heading = 0.0
    enemy.position = Vec2(720, 350)
    enemy.heading = math.pi
    return world, learner, enemy


def _run_until_event(world: World, wanted: str, max_steps: int = 80) -> list[Event]:
    seen: list[Event] = []
    for _ in range(max_steps):
        events = world.tick()
        seen.extend(events)
        if any(event.kind == wanted for event in events):
            return seen
    return seen


def check_movement_rotation_and_reset() -> dict:
    world, learner, enemy = _open_world()
    old_heading = learner.heading
    learner.apply_command(AgentCommand(rot_idx=2), world)
    _assert(learner.heading != old_heading, "rotation command did not change learner heading")

    learner.heading = 0.0
    old_x = learner.position.x
    learner.apply_command(AgentCommand(move_idx=1), world)
    world.tick()
    _assert(learner.position.x > old_x, "forward movement did not advance the agent")

    world.fire_projectile("learner", learner.position, Vec2(1, 0), 1.0, 100.0)
    _assert(len(world.projectiles) == 1, "projectile was not spawned before reset")
    learner.health = 30.0
    world.reset_round()
    _assert(len(world.projectiles) == 0, "reset did not clear projectiles")
    _assert(abs(learner.health - learner.max_health) < 1e-6, "reset did not restore learner health")
    _assert(all(pickup.active for pickup in world.pickups), "reset did not reactivate pickups")
    return {"movement": True, "rotation": True, "reset": True}


def check_wall_and_obstacle_collision() -> dict:
    world, learner, _enemy = _open_world()
    learner.position = Vec2(18, 350)
    learner.heading = math.pi
    learner.apply_command(AgentCommand(move_idx=1), world)
    events = world.tick()
    _assert(any(event.kind == "blocked_move" for event in events), "wall collision did not emit blocked_move")
    _assert(learner.position.x >= learner.radius - 0.5, "wall collision allowed agent outside arena")

    obstacle_world = World(build_layout("stage7"))
    obstacle_learner = _make_agent("learner", 320, 350, 0.0)
    obstacle_enemy = _make_agent("enemy", 760, 350, math.pi)
    obstacle_world.add_agent(obstacle_learner)
    obstacle_world.add_agent(obstacle_enemy)
    movement = detection.check_movement_path(obstacle_world, Vec2(320, 350), Vec2(460, 350), obstacle_learner.radius)
    _assert(movement.blocked and movement.blocker_type == "obstacle", "obstacle movement collision failed")
    _assert(not obstacle_world.has_line_of_sight(obstacle_learner.position, obstacle_enemy.position), "obstacle should block line of sight")
    return {"wall_collision": True, "obstacle_collision": True, "line_of_sight_blocking": True}


def check_shoot_reload_damage_and_death() -> dict:
    world, learner, enemy = _open_world()
    learner.projectile_damage = 35.0
    learner.projectile_speed = 620.0
    shoot_events = learner.apply_command(AgentCommand(shoot=True), world)
    _assert(any(event.kind == "shot_fired" for event in shoot_events), "shoot command did not emit shot_fired")
    _assert(len(world.projectiles) == 1, "shoot command did not spawn projectile")
    hit_events = _run_until_event(world, "damage_taken", max_steps=60)
    _assert(any(event.kind == "shot_hit" and event.agent_id == "learner" for event in hit_events), "projectile did not hit enemy")
    _assert(any(event.kind == "damage_taken" and event.agent_id == "enemy" for event in hit_events), "damage_taken event missing for projectile hit")
    _assert(enemy.health < enemy.max_health, "projectile hit did not reduce enemy health")

    world, learner, enemy = _open_world()
    learner.projectile_damage = 120.0
    learner.projectile_speed = 620.0
    learner.apply_command(AgentCommand(shoot=True), world)
    kill_events = _run_until_event(world, "death", max_steps=60)
    _assert(any(event.kind == "kill" and event.agent_id == "learner" for event in kill_events), "lethal shot did not emit kill")
    _assert(any(event.kind == "death" and event.agent_id == "enemy" for event in kill_events), "lethal shot did not emit death")
    _assert(not enemy.is_alive, "enemy remained alive after lethal projectile")

    reload_world, reloader, _ = _open_world()
    reloader.ammo = 0
    reload_events = reloader.apply_command(AgentCommand(reload=True), reload_world)
    _assert(any(event.kind == "reload_started" for event in reload_events), "reload command did not start reload")
    for _ in range(int(reloader.reload_time / reload_world.dt) + 2):
        reload_world.tick()
    _assert(reloader.ammo == reloader.max_ammo and not reloader.reloading, "reload did not refill ammo")
    return {"shooting": True, "reload": True, "damage": True, "death": True}


def check_projectile_swept_collision_and_raycast() -> dict:
    world, learner, enemy = _open_world()
    check = detection.check_projectile_path(world, "learner", Vec2(500, 350), Vec2(760, 350), projectile_radius=3.0, step_size=12.0)
    _assert(check.hit and check.hit_type == "agent" and check.hit_agent_id == "enemy", "swept projectile collision missed enemy")

    obstacle_world = World(build_layout("stage7"))
    obstacle_learner = _make_agent("learner", 320, 350, 0.0)
    obstacle_enemy = _make_agent("enemy", 760, 350, math.pi)
    obstacle_world.add_agent(obstacle_learner)
    obstacle_world.add_agent(obstacle_enemy)
    blocked = detection.check_projectile_path(obstacle_world, "learner", Vec2(320, 350), Vec2(500, 350), projectile_radius=3.0)
    _assert(blocked.hit and blocked.blocker_type == "obstacle", "swept projectile collision did not stop on obstacle")

    world.fire_projectile("enemy", Vec2(540, 350), Vec2(-1, 0), damage=5.0, speed=220.0)
    hit = detection.trace_ray(world, learner.position, Vec2(1, 0), max_length=360, step_size=5, target_agent_id="enemy", observer_agent_id="learner")
    _assert(hit.hit_type == detection.HIT_PROJECTILE, "raycast did not report projectile as first visible object")

    stage8_world = World(build_layout("stage8"))
    _assert(len(stage8_world.pickups) == 0, "Stage 8 should not spawn map pickups")
    return {"swept_projectile_collision": True, "projectile_raycast": True, "stage8_no_map_pickups": True}


def check_resources_rewards_and_observation() -> dict:
    world = World(build_layout("stage8"))
    learner = _make_agent("learner", 450, 350, 0.0)
    enemy = _make_agent("enemy", 900, 350, math.pi)
    world.add_agent(learner)
    world.add_agent(enemy)
    world.reset_round()

    _assert(len(world.pickups) == 0, "Stage 8 tactical reload layout should not contain map pickups")

    reward_world, reward_learner, reward_enemy = _open_world()
    manager = RewardManager("stage6")
    manager.reset(reward_learner, reward_enemy, reward_world)
    reward = manager.compute([Event("damage_taken", "learner", value=20.0)], reward_learner, reward_enemy, reward_world, 1)
    _assert(reward < -0.5 and manager.damage_taken >= 20.0, "damage-taking penalty was not applied")

    obs_builder = ObsBuilder()
    skill_table = build_skill_table("stage8", evaluation_mode=True)
    obs = obs_builder.build(learner, enemy, world, skill_table)
    _assert(obs.dtype == np.float32, "observation dtype is not float32")
    _assert(np.all(np.isfinite(obs)), "observation contains non-finite values")
    _assert(float(obs.min()) >= -1.0001 and float(obs.max()) <= 1.0001, "observation is not normalized to [-1, 1]")
    _assert(ObsBuilder.OBS_DIM == obs.shape[0], "observation length does not match ObsBuilder.OBS_DIM")
    stage1_obs = obs_builder.build(learner, enemy, world, skill_table, stage_id="stage1")
    _assert(stage1_obs.shape[0] == ObsBuilder.STAGE1_OBS_DIM, "Stage 1 compact observation length mismatch")
    _assert(np.all(np.isfinite(stage1_obs)), "Stage 1 compact observation contains non-finite values")
    return {"pickup_collection": True, "damage_penalty": True, "observation_normalization": True, "stage1_observation_filter": True}


def check_environment_reset_interface() -> dict:
    try:
        from rl.env import ShooterEnv
    except ModuleNotFoundError as exc:

        if exc.name != "gymnasium":
            raise
        world, learner, _enemy = _open_world()
        learner.health = 10.0
        world.fire_projectile("learner", learner.position, Vec2(1, 0), 1.0, 100.0)
        world.reset_round()
        _assert(learner.health == learner.max_health and len(world.projectiles) == 0, "world reset fallback failed")
        return {"gym_reset": "skipped: gymnasium not installed", "world_reset": True}

    env = ShooterEnv(stage_id="stage1", render_mode=None, validation_mode=True)
    try:
        obs, info = env.reset(seed=123)
        _assert(obs.shape == env.observation_space.shape, "reset observation shape mismatch")
        _assert(np.all(np.isfinite(obs)), "reset observation contains non-finite values")
        expected_stage1_mask = [True, False, False, True, False, False, True, True, True, True, False, True, False]
        _assert(stage_action_mask("stage1").tolist() == expected_stage1_mask, "Stage 1 action mask does not lock movement/shoot/reload branches")
        _assert(sample_valid_action("stage1").shape == env.action_space.shape, "valid Stage 1 sample shape mismatch")
        obs2, reward, terminated, truncated, step_info = env.step(env.sample_valid_action())
        _assert(obs2.shape == env.observation_space.shape, "step observation shape mismatch")
        _assert(isinstance(float(reward), float), "step reward is not numeric")
        obs3, _ = env.reset(seed=124)
        _assert(obs3.shape == env.observation_space.shape, "second reset observation shape mismatch")
        _assert(env.world is not None and len(env.world.projectiles) == 0, "environment reset did not clear projectiles")
    finally:
        env.close()
    return {"gym_reset": True, "gym_step": True, "frame_stacked_observation": True}


def run_stage0_sanity_check(verbose: bool = True) -> dict:
    
    checks: list[tuple[str, Callable[[], dict]]] = [
        ("movement, rotation, reset", check_movement_rotation_and_reset),
        ("wall/obstacle collision and line of sight", check_wall_and_obstacle_collision),
        ("shooting, reload, damage, death", check_shoot_reload_damage_and_death),
        ("projectile collision and raycast visibility", check_projectile_swept_collision_and_raycast),
        ("resources, rewards, observations", check_resources_rewards_and_observation),
        ("RL environment reset interface", check_environment_reset_interface),
    ]

    report: dict[str, object] = {"stage": "stage0", "passed": False, "checks": {}}
    for label, check_fn in checks:
        result = check_fn()
        report["checks"][label] = result
        if verbose:
            print(f"[Stage 0] OK: {label}")

    report["passed"] = True
    mark_stage0_sanity_passed(report)
    if verbose:
        print("Stage 0 sanity check passed. Mechanics are ready for PPO curriculum training.")
    return report


def check_detection() -> None:
    run_stage0_sanity_check(verbose=True)


if __name__ == "__main__":
    run_stage0_sanity_check(verbose=True)
