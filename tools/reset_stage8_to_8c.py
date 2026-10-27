from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


PHASE_INDEX_8C = 2  # zero-based index in STAGE8_CURRICULUM_PHASES
PHASE_NAME = "8C"
DEFAULT_STATE_PATH = Path("checkpoints/curriculum_masked_ppo_v3/stage8/interruptedTrainingStateStage8.pt")
STATS_PATH = Path("stage_stats_curriculum_masked_ppo_v3.json")

WINDOW_KEYS = (
    "stage8_phase_success_window",
    "stage8_phase_metric_window",
    "stage8_phase_death_window",
    "stage8_phase_reload_quality_window",
    "stage8_phase_early_reload_window",
    "stage8_phase_empty_ammo_window",
)


def reset_callback_state(path: Path) -> bool:
    if torch is None:
        print(f"Cannot update .pt callback state because torch import failed: {TORCH_IMPORT_ERROR}")
        return False
    if not path.exists():
        print(f"Checkpoint state not found: {path}")
        return False

    data = torch.load(path, map_location="cpu", weights_only=False)
    state = data.get("callback_state", data)

    old_phase = state.get("stage8_curriculum_phase")
    old_gate_phase = state.get("stage8_eval_gate_phase")
    old_checks = state.get("stage8_eval_gate_good_checks")

    state["stage8_curriculum_phase"] = PHASE_INDEX_8C
    state["stage8_episodes_at_phase"] = 0
    state["stage8_eval_gate_phase"] = PHASE_INDEX_8C
    state["stage8_eval_gate_good_checks"] = 0
    state["stage8_last_eval_gate_summary"] = {}
    state["stage8_last_promotion_event"] = (
        "MANUAL RESET TO 8C after obstacle/range stabilization patch"
    )

    for key in WINDOW_KEYS:
        state[key] = []

    if "callback_state" in data:
        data["callback_state"] = state
    else:
        data = state

    backup = path.with_suffix(path.suffix + ".before_8c_stabilization_reset_backup")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

    torch.save(data, path)
    print(f"Updated checkpoint state: {path}")
    print(f"Backup: {backup}")
    print(f"stage8_curriculum_phase: {old_phase} -> {PHASE_INDEX_8C} ({PHASE_NAME})")
    print(f"stage8_eval_gate_phase: {old_gate_phase} -> {PHASE_INDEX_8C}")
    print(f"stage8_eval_gate_good_checks: {old_checks} -> 0")
    return True


def reset_stats(path: Path) -> bool:
    if not path.exists():
        print(f"Stats file not found, skipping: {path}")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))

    # The project has used more than one stats layout during development. Update
    # both the top-level stage8 object and stages.stage8 when present.
    targets = []
    if isinstance(data.get("stage8"), dict):
        targets.append(data["stage8"])
    stages = data.get("stages")
    if isinstance(stages, dict) and isinstance(stages.get("stage8"), dict):
        targets.append(stages["stage8"])

    for stage8 in targets:
        stage8["stage8_curriculum_phase"] = PHASE_INDEX_8C
        stage8["stage8_eval_gate_phase"] = PHASE_INDEX_8C
        stage8["stage8_phase_name"] = PHASE_NAME
        stage8["good_checks_count"] = 0
        stage8["required_good_checks"] = 2
        stage8["stage8_last_promotion_event"] = (
            "manually reset to 8C after obstacle/range stabilization patch"
        )

    backup = path.with_suffix(path.suffix + ".before_8c_stabilization_reset_backup")
    if not backup.exists():
        backup.write_text(json.dumps(data, indent=2), encoding="utf-8")

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated stats metadata: {path}")
    return True


def main() -> int:
    state_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STATE_PATH
    ok_state = reset_callback_state(state_path)
    ok_stats = reset_stats(STATS_PATH)
    if not ok_state:
        print("No checkpoint state was updated. Check the path or pass it as an argument.")
        return 1
    if not ok_stats:
        print("Checkpoint reset was applied, but stats metadata was not changed.")
    print("Stage 8 reset to 8C. Model weights are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
