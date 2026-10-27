# Reinforcement Learning for Skill-Based Bot Development in a Top-Down Shooter Game

A Python project that trains a reinforcement learning (RL) agent to play a custom 2D top-down shooter inspired by Counter-Strike 2D. The agent learns gameplay skills progressively using **curriculum learning** and **Maskable Proximal Policy Optimization (Maskable PPO)** rather than relying on a fully scripted combat policy.

This project was developed for my bachelor's thesis at the **University of Primorska, FAMNIT (2026)**.

## Overview

Training a shooter bot requires several related abilities: tracking a moving target, shooting at the right moment, maintaining a useful position, dodging projectiles, fighting around obstacles, and managing ammunition. Trying to learn all of these behaviors simultaneously makes it difficult to identify which skills have actually been acquired.

The project addresses this by using a staged training curriculum. Each stage introduces a more demanding task, with its own reward design and validation criteria. The agent trains against a configurable scripted opponent (the *stock bot*) in a custom Pygame environment exposed through the Gymnasium API.

The focus is both on **developing the agent** and on **measuring how its skills emerge and combine during training**.

## Key features

- **Custom 2D shooter:** movement, projectiles, health, ammunition, reloading, walls, obstacles, and line-of-sight checks.
- **Gymnasium environment:** a modular interface connecting gameplay to RL training.
- **Maskable PPO:** policy optimization with valid-action masks in the stages that use masking.
- **Curriculum learning:** eight stages, from basic aiming to extended combat and reload management.
- **Reward shaping and separate success metrics:** rewards guide learning; independent behavioral checks determine whether a stage is passed.
- **Partial observations and frame stacking:** numerical state features instead of raw game screenshots.
- **Scripted opponents:** configurable stock-bot behavior for controlled training scenarios.
- **Training diagnostics:** evaluation logs, saved checkpoints, and stage-specific performance metrics.
- **Visualization and play modes:** interact with the game or watch a trained agent.

## Training curriculum

| Stage | Main objective |
| --- | --- |
| 1. Aiming | Rotate toward the opponent and maintain aim alignment. |
| 2. Shooting | Learn when to fire while maintaining accurate aim. |
| 3. Movement | Control distance and positioning without losing sight of the opponent. |
| 4. Dodging | Avoid incoming projectiles while maintaining active movement and survival. |
| 5. Moving shooting | Combine movement, aiming, and accurate shooting. |
| 6. Open-arena combat | Combine movement, shooting, dodging, survival, and manual reloading. |
| 7. Obstacle combat | Navigate around obstacles, recover line of sight, and find clear shots. |
| 8. Reload endurance | Survive successive enemy waves while managing limited reload resources. |

The curriculum includes intermediate phases and checks for previously acquired skills. Progression is based on validation criteria, not simply on accumulated reward.

## How the agent works

### Learning algorithm

The agent uses **Maskable PPO** from `sb3-contrib`, built on Stable-Baselines3 and PyTorch. Its policy receives numerical observations from a custom `gymnasium.Env` and chooses a multi-branch action:

```text
MultiDiscrete([3, 3, 3, 2, 2])

[ movement, dodge, rotation, shoot, reload ]
```

Action masking prevents selection of unavailable actions during the stages where masks are enabled. **Stage 8 intentionally tests decision-making without action masking**, making poor shooting and reload choices part of the learning challenge.

### Observation design

The agent learns from structured numerical features rather than pixel images:

- **Stages 1-2:** 15 aiming-focused features per observation.
- **Stages 3-8:** 243 features covering raycast perception, agent state, visible opponent state, projectile threats, obstacles, line of sight, reload information, and skill status.
- **Four-frame stacking:** the model receives 60 input values in Stages 1-2 or 972 input values in Stages 3-8, providing a short history of recent game states.

### Rewards and evaluation

Each stage has a reward function that encourages relevant behavior. Separately, stage-specific validation metrics check whether the intended skill was actually learned. These include measures such as aiming accuracy, hit rate, movement quality, projectile avoidance, survival, combat performance, and reload quality.

This separation matters because a policy can accumulate high reward without completing the actual stage objective. Checkpoints and evaluation logs are used to inspect progression and select trained policies.

## Results

The bachelor's thesis reports the following results from **one main curriculum training run**:

| Stage | Best stage-specific validation score | Status |
| --- | ---: | --- |
| 1. Aiming | 0.8500 | Passed |
| 2. Shooting | 0.7650 | Passed |
| 3. Movement | 0.8000 | Passed |
| 4. Dodging | 0.8000 | Passed |
| 5. Moving shooting | 0.8000 | Passed |
| 6. Open-arena combat | 0.7500 | Passed |
| 7. Obstacle combat | 0.7333 | Passed |
| 8. Reload endurance | Final validation success: 0.0 | Not passed |

**Important:** the validation scores use stage-specific criteria. They are **not win rates** and should not be compared as if every stage used the same test.

The agent successfully completed the first seven curriculum stages under their tested conditions. Stage 8 exposed an unresolved issue with reload timing and resource management across longer combat sequences, even though the agent could still deal damage and clear some waves.

These results demonstrate what this particular training configuration achieved. They do not establish that curriculum learning outperforms non-curriculum training, because no direct non-curriculum baseline or multi-seed comparison was performed.

## Technology stack

- **Python 3.12**
- **Pygame** for game simulation and rendering
- **Gymnasium** for the RL environment interface
- **Stable-Baselines3** and **sb3-contrib** for Maskable PPO
- **PyTorch** for neural-network training
- **NumPy** for numerical operations
- **Matplotlib** for training analysis and plots

## Project structure

```text
config/             Game constants and curriculum stage configurations
game/              Game world, agents, scripted opponent, and maps
rl/                Gymnasium environment, observations, actions, and rewards
navigation/        A* navigation and path-reference utilities
training/          Training, evaluation, callbacks, and checkpoint management
rendering/         Pygame rendering
ui/                Game menus and interface
tools/             Sanity checks and utilities
main.py            Launch the game
train.py           Start curriculum training
watch.py           Watch a trained policy
requirements.txt   Python dependencies
```

## Limitations and future work

- The evaluation is based on one main curriculum training run, not an extensive multi-seed study.
- There is no direct comparison with training without a curriculum or with alternative RL algorithms.
- Performance on unseen opponents and maps has not been established.
- The final reload-endurance stage did not satisfy its validation criteria.

Possible extensions include broader opponent randomization, multi-seed experiments, curriculum ablations, improved reload-resource training, and a more automated evaluation and checkpoint-selection pipeline.

## Academic context

**Author:** Hristijan Kochovski  
**Thesis:** *Reinforcement Learning for Skill-Based Bot Development in a Top-Down Shooter Game*  
**Institution:** University of Primorska, Faculty of Mathematics, Natural Sciences and Information Technologies (FAMNIT)  
**Year:** 2026

The project is a research and educational prototype, not a production-ready game AI system.
