from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from config.stages import PLOTS_DIR, get_stage, normalize_stage_id


def _read_train_log(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _numeric_column(rows: list[dict[str, str]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            values.append(0.0)
    return values


def _moving_average(values: list[float], window: int = 20) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        result.append(sum(segment) / len(segment))
    return result


def _plot_series(episodes: list[int], values: list[float], title: str, ylabel: str, path: Path, window: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, values, label=ylabel)
    plt.plot(episodes, _moving_average(values, window=window), label=f"{ylabel} MA({window})")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_stage(stage_id: str, window: int = 20) -> dict[str, Path]:
    stage_id = normalize_stage_id(stage_id)
    stage = get_stage(stage_id)
    csv_path = stage.train_log_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    rows = _read_train_log(csv_path)
    if not rows:
        raise RuntimeError(f"Training log is empty: {csv_path}")

    episodes = [int(float(row["episode"])) for row in rows]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "reward": PLOTS_DIR / f"{stage.stage_id}_reward_per_episode.png",
        "success_metric": PLOTS_DIR / f"{stage.stage_id}_success_metric.png",
        "damage_ratio": PLOTS_DIR / f"{stage.stage_id}_damage_ratio.png",
        "hit_rate": PLOTS_DIR / f"{stage.stage_id}_hit_rate.png",
        "damage_taken": PLOTS_DIR / f"{stage.stage_id}_damage_taken.png",
    }

    _plot_series(episodes, _numeric_column(rows, "total_reward"), f"{stage.name} - Reward", "Total reward", outputs["reward"], window)
    _plot_series(episodes, _numeric_column(rows, "stage_success_metric"), f"{stage.name} - Stage success metric", "Stage success metric", outputs["success_metric"], window)
    _plot_series(episodes, _numeric_column(rows, "damage_dealt_taken_ratio"), f"{stage.name} - Damage dealt/taken ratio", "Damage ratio", outputs["damage_ratio"], window)
    _plot_series(episodes, _numeric_column(rows, "hit_rate"), f"{stage.name} - Hit rate", "Hit rate", outputs["hit_rate"], window)
    _plot_series(episodes, _numeric_column(rows, "damage_taken"), f"{stage.name} - Damage taken", "Damage taken", outputs["damage_taken"], window)

    for name, path in outputs.items():
        print(f"Saved {name} plot to: {path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot curriculum training metrics.")
    parser.add_argument("--stage", type=str, default="stage1", help="stage0 through stage8; old aliases still work")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    plot_stage(args.stage, window=args.window)


if __name__ == "__main__":
    main()
