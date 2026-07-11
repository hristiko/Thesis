from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.stages import (
    RUN_FAMILY,
    STAGE_ORDER,
    STAGES,
    STATS_PATH,
    TRAINABLE_STAGE_ORDER,
    ensure_runtime_directories,
    get_stage,
    normalize_stage_id,
)
from training.skill_table import build_skill_table


MANUAL_UNLOCKED_STAGES = {"stage5"}
MANUAL_PREVIOUS_BEST_STAGES = {("stage5", "stage4")}


def _stage_default_entry(stage_id: str) -> dict[str, Any]:
    stage = STAGES[stage_id]
    return {
        "name": stage.name,
        "description": stage.description,
        "trainable": stage.trainable,
        "trained": False,
        "unlocked": stage_id in ("stage0", "stage1"),
        "sanity_passed": False,
        "best_success_rate": 0.0,
        "best_validation_score": 0.0,
        "validation_success_rate": 0.0,
        "last_rolling_success_rate": 0.0,
        "last_total_episodes": 0,
        "has_best_model": False,
        "official_best_model": False,
        "passed_validation": False,
        "best_model_path": str(stage.best_model_path),
        "last_checkpoint_path": "",
        "last_promotion_reason": "",
        "last_validation_report": {},
        "learning_skill": stage.learning_skill,
    }


def _default_stats() -> dict[str, Any]:
    return {
        "run_family": RUN_FAMILY,
        "algorithm": "MaskablePPO",
        "last_trained_stage": None,
        "skill_table": build_skill_table("stage1"),
        "stages": {stage_id: _stage_default_entry(stage_id) for stage_id in STAGE_ORDER},
    }


def ensure_stage_stats_file() -> None:
    ensure_runtime_directories()
    if not STATS_PATH.exists():
        save_stage_stats(_default_stats())
        return

    data = load_stage_stats()
    default = _default_stats()
    changed = False

    if data.get("run_family") != RUN_FAMILY:
        # Different run families are not observation/action compatible. Keep the
        # old file on disk but initialize this run family's stats cleanly.
        data = default
        changed = True
    else:
        for key, value in default.items():
            if key not in data:
                data[key] = value
                changed = True

        if "stages" not in data:
            data["stages"] = {}
            changed = True

        for stage_id in STAGE_ORDER:
            if stage_id not in data["stages"]:
                data["stages"][stage_id] = default["stages"][stage_id]
                changed = True
            else:
                for key, value in default["stages"][stage_id].items():
                    if key not in data["stages"][stage_id]:
                        data["stages"][stage_id][key] = value
                        changed = True

    if changed:
        save_stage_stats(data)
    refresh_stage_unlocks(save=True)


def load_stage_stats() -> dict[str, Any]:
    ensure_runtime_directories()
    if not STATS_PATH.exists():
        return _default_stats()
    with STATS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_stage_stats(data: dict[str, Any]) -> None:
    ensure_runtime_directories()
    with STATS_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _stage_has_any_checkpoint(stage) -> bool:
    return (
        stage.best_model_path.exists()
        or stage.latest_model_path.exists()
        or stage.interrupted_model_path.exists()
        or stage.best_training_state_path.exists()
        or stage.latest_training_state_path.exists()
        or stage.interrupted_training_state_path.exists()
    )


def _stage_has_official_best(data: dict[str, Any], stage_id: str) -> bool:
    stage = get_stage(stage_id)
    entry = data.get("stages", {}).get(stage.stage_id, {})
    return bool(
        stage.best_model_path.exists()
        and entry.get("official_best_model", False)
        and entry.get("passed_validation", False)
    )


def refresh_stage_unlocks(save: bool = False) -> dict[str, Any]:
    """Refresh display state without unlocking from arbitrary checkpoints.

    The old curriculum unlocked the next stage when any previous checkpoint
    existed. This was convenient during demos but unsafe for thesis evaluation.
    Now a trainable stage unlocks only when the required previous stage has an
    official validated best model. Interrupted/latest checkpoints remain useful
    for manual resume, but they do not advance the curriculum.
    """
    data = load_stage_stats()
    if data.get("run_family") != RUN_FAMILY:
        data = _default_stats()

    data.setdefault("stages", {})
    for stage_id in STAGE_ORDER:
        stage = get_stage(stage_id)
        entry = data["stages"].setdefault(stage_id, _stage_default_entry(stage_id))
        entry["name"] = stage.name
        entry["description"] = stage.description
        entry["trainable"] = stage.trainable
        entry["best_model_path"] = str(stage.best_model_path)
        entry["has_best_model"] = stage.best_model_path.exists()
        entry["learning_skill"] = stage.learning_skill

        entry["trained"] = bool(entry.get("trained", False) or _stage_has_any_checkpoint(stage))
        if stage_id == "stage0":
            entry["unlocked"] = True
        elif stage.required_previous_stage is None:
            entry["unlocked"] = True
        elif stage.stage_id in MANUAL_UNLOCKED_STAGES:
            entry["unlocked"] = True
        else:
            entry["unlocked"] = _stage_has_official_best(data, stage.required_previous_stage)

        # If a best model disappeared, do not keep claiming official completion.
        if not stage.best_model_path.exists() and stage.trainable:
            entry["official_best_model"] = False
            entry["passed_validation"] = False

    if save:
        save_stage_stats(data)
    return data


def update_training_stats(
    stage_id: str,
    best_success_rate: float,
    last_rolling_success_rate: float,
    total_episodes: int,
    skill_table: dict[str, Any] | None = None,
    last_checkpoint_path: str = "",
    promotion_reason: str = "",
    *,
    passed_validation: bool | None = None,
    official_best_model: bool | None = None,
    validation_success_rate: float | None = None,
    best_validation_score: float | None = None,
    last_validation_report: dict[str, Any] | None = None,
) -> None:
    canonical_stage_id = get_stage(normalize_stage_id(stage_id)).stage_id
    data = refresh_stage_unlocks(save=False)
    data["last_trained_stage"] = canonical_stage_id
    if skill_table is not None:
        data["skill_table"] = skill_table

    entry = data["stages"][canonical_stage_id]
    entry["trained"] = total_episodes > 0 or entry.get("trained", False)
    entry["best_success_rate"] = max(float(entry.get("best_success_rate", 0.0)), float(best_success_rate))
    entry["last_rolling_success_rate"] = float(last_rolling_success_rate)
    entry["last_total_episodes"] = int(total_episodes)
    entry["has_best_model"] = get_stage(canonical_stage_id).best_model_path.exists()
    entry["best_model_path"] = str(get_stage(canonical_stage_id).best_model_path)
    if validation_success_rate is not None:
        entry["validation_success_rate"] = float(validation_success_rate)
    if best_validation_score is not None:
        entry["best_validation_score"] = max(float(entry.get("best_validation_score", 0.0)), float(best_validation_score))
    if passed_validation is not None:
        entry["passed_validation"] = bool(passed_validation or entry.get("passed_validation", False))
    if official_best_model is not None:
        entry["official_best_model"] = bool(official_best_model or entry.get("official_best_model", False))
    if last_checkpoint_path:
        entry["last_checkpoint_path"] = str(last_checkpoint_path)
    if promotion_reason:
        entry["last_promotion_reason"] = str(promotion_reason)
    if last_validation_report is not None:
        entry["last_validation_report"] = last_validation_report

    save_stage_stats(data)
    refresh_stage_unlocks(save=True)


def mark_stage0_sanity_passed(report: dict[str, Any]) -> None:
    data = refresh_stage_unlocks(save=False)
    entry = data["stages"]["stage0"]
    entry["sanity_passed"] = True
    entry["passed_validation"] = True
    entry["last_validation_report"] = report
    entry["last_promotion_reason"] = "Stage 0 sanity checks passed."
    data["last_trained_stage"] = "stage0"
    save_stage_stats(data)


def get_global_skill_table() -> dict[str, Any]:
    data = load_stage_stats()
    return data.get("skill_table") or build_skill_table("stage1")


def get_previous_stage_best_model(stage_id: str) -> Path | None:
    stage = get_stage(stage_id)
    if stage.required_previous_stage is None:
        return None
    data = refresh_stage_unlocks(save=True)
    previous_stage = get_stage(stage.required_previous_stage)
    if _stage_has_official_best(data, previous_stage.stage_id):
        return previous_stage.best_model_path
    if (stage.stage_id, previous_stage.stage_id) in MANUAL_PREVIOUS_BEST_STAGES and previous_stage.best_model_path.exists():
        return previous_stage.best_model_path
    return None


def get_stage_display_rows() -> list[dict[str, Any]]:
    data = refresh_stage_unlocks(save=True)
    rows: list[dict[str, Any]] = []
    for stage_id in STAGE_ORDER:
        stage = get_stage(stage_id)
        entry = data["stages"][stage_id]
        rows.append(
            {
                "stage_id": stage_id,
                "name": stage.name,
                "description": stage.description,
                "trainable": stage.trainable,
                "unlocked": bool(entry.get("unlocked", False)),
                "trained": bool(entry.get("trained", False)),
                "passed_validation": bool(entry.get("passed_validation", False)),
                "official_best_model": bool(entry.get("official_best_model", False)),
                "best_success_rate": float(entry.get("best_success_rate", 0.0)),
                "validation_success_rate": float(entry.get("validation_success_rate", 0.0)),
                "last_rolling_success_rate": float(entry.get("last_rolling_success_rate", 0.0)),
                "last_total_episodes": int(entry.get("last_total_episodes", 0)),
                "has_best_model": bool(entry.get("has_best_model", False)),
                "best_model_path": str(entry.get("best_model_path", stage.best_model_path)),
                "latest_checkpoint_path": str(stage.latest_training_state_path) if stage.latest_training_state_path.exists() else "",
                "interrupted_checkpoint_path": str(stage.interrupted_training_state_path) if stage.interrupted_training_state_path.exists() else "",
                "learning_skill": stage.learning_skill,
                "last_checkpoint_path": str(entry.get("last_checkpoint_path", "")),
                "last_promotion_reason": str(entry.get("last_promotion_reason", "")),
            }
        )
    return rows


def get_trainable_stage_display_rows() -> list[dict[str, Any]]:
    return [row for row in get_stage_display_rows() if row["stage_id"] in TRAINABLE_STAGE_ORDER]


def get_last_trained_stage_label() -> str:
    data = refresh_stage_unlocks(save=True)
    stage_id = data.get("last_trained_stage")
    if not stage_id:
        return "Your bot was last trained in: none yet"
    stage = get_stage(stage_id)
    if stage.stage_id == "stage0":
        return "Last curriculum action: Stage 0 sanity check"
    return f"Your bot was last trained in: {stage.name} using MaskablePPO"


def get_testable_model_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = refresh_stage_unlocks(save=True)
    for stage_id in TRAINABLE_STAGE_ORDER:
        stage = get_stage(stage_id)
        entry = data["stages"][stage_id]
        if stage.best_model_path.exists():
            rows.append(
                {
                    "stage_id": stage_id,
                    "label": stage.best_model_filename,
                    "description": stage.name,
                    "official": bool(entry.get("official_best_model", False)),
                    "passed_validation": bool(entry.get("passed_validation", False)),
                }
            )
    return rows


def get_previous_stage_source_rows(stage_id: str) -> list[dict[str, Any]]:
    stage = get_stage(stage_id)
    if stage.required_previous_stage is None:
        return []
    previous_stage = get_stage(stage.required_previous_stage)
    data = refresh_stage_unlocks(save=True)
    previous_entry = data["stages"][previous_stage.stage_id]
    official_best_available = _stage_has_official_best(data, previous_stage.stage_id)
    if (stage.stage_id, previous_stage.stage_id) in MANUAL_PREVIOUS_BEST_STAGES and previous_stage.best_model_path.exists():
        official_best_available = True
    return [
        {
            "source": "best",
            "label": "validated best",
            "description": (
                f"Official validated MaskablePPO model from {previous_stage.name}. "
                f"Required for curriculum progression. File: {previous_stage.best_model_filename}"
            ),
            "enabled": official_best_available,
        },
        {
            "source": "latest",
            "label": "latest same-architecture checkpoint",
            "description": (
                f"Most recent MaskablePPO progress from {previous_stage.name}. Useful for manual experiments only; "
                "it does not unlock this stage."
            ),
            "enabled": previous_stage.latest_model_path.exists() and bool(previous_entry.get("trained", False)),
        },
        {
            "source": "interrupted",
            "label": "interrupted same-architecture checkpoint",
            "description": (
                f"Checkpoint saved when {previous_stage.name} was stopped manually. Useful for manual experiments only; "
                "it does not unlock this stage."
            ),
            "enabled": previous_stage.interrupted_model_path.exists(),
        },
    ]


def get_previous_stage_name(stage_id: str) -> str | None:
    stage = get_stage(stage_id)
    if stage.required_previous_stage is None:
        return None
    return get_stage(stage.required_previous_stage).name
