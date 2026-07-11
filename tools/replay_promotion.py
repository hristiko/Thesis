from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.callbacks import STAGE8_CURRICULUM_PHASES
from training.promotion import evaluate_stage8_phase


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def replay_stage8(eval_log_path: Path) -> int:
    with eval_log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("eval_check", ""))].append(row)

    phase_idx = 0
    good_checks = 0
    promotions = 0

    for eval_check in sorted(grouped.keys(), key=lambda x: int(float(x or 0))):
        batch = grouped[eval_check]
        phase = STAGE8_CURRICULUM_PHASES[phase_idx]
        episodes = len(batch)
        mean_metrics: dict[str, float] = {}
        metric_keys = [
            "success",
            "deaths",
            "stage8_reload_quality",
            "stage8_early_reloads",
            "stage8_empty_ammo_steps",
        ]
        for key in metric_keys:
            vals = [_f(r.get(key, 0.0)) for r in batch]
            mean_metrics[key] = sum(vals) / max(1, len(vals))
        summary = {
            "episodes": episodes,
            "success_rate": mean_metrics.get("success", 0.0),
            "validation_score": _f(batch[0].get("validation_score", 0.0)),
            "mean_metrics": mean_metrics,
        }
        decision = evaluate_stage8_phase(phase, summary, default_required_checks=3)
        good_checks = good_checks + 1 if decision.passed else 0
        print(
            f"check {eval_check:>4} | phase {phase['phase']} | passed={decision.passed} | "
            f"checks {good_checks}/{decision.checks_required} | score {decision.validation_score:.3f} | "
            f"success {decision.success_rate:.2%}"
        )
        if good_checks >= decision.checks_required and phase_idx < len(STAGE8_CURRICULUM_PHASES) - 1:
            old = phase["phase"]
            phase_idx += 1
            promotions += 1
            good_checks = 0
            print(f"PROMOTE {old} -> {STAGE8_CURRICULUM_PHASES[phase_idx]['phase']} at eval_check {eval_check}")
    print(f"Final inferred phase: {STAGE8_CURRICULUM_PHASES[phase_idx]['phase']}")
    return promotions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_log", type=Path)
    args = parser.parse_args()
    replay_stage8(args.eval_log)


if __name__ == "__main__":
    main()
