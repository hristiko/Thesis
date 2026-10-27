from __future__ import annotations

from copy import deepcopy
from typing import Any

from config.stages import get_stage, normalize_stage_id, stage_index

FULL_SKILL_ORDER = [
    "environment_sanity",
    "aim_tracking",
    "shooting_timing",
    "basic_movement",
    "dodging_survival",
    "moving_shooting",
    "open_arena_combat",
    "obstacle_cover_combat",
    "tactical_reload_management",
]

SKILL_FIRST_STAGE = {
    "environment_sanity": "stage0",
    "aim_tracking": "stage1",
    "shooting_timing": "stage2",
    "basic_movement": "stage3",
    "dodging_survival": "stage4",
    "moving_shooting": "stage5",
    "open_arena_combat": "stage6",
    "obstacle_cover_combat": "stage7",
    "pickup_resource_management": "stage8",
    "tactical_reload_management": "stage8",
}

STATUS_TO_FEATURE = {
    "not_learned": 0.0,
    "locked_future": 0.0,
    "inactive": 0.25,
    "learning": 0.50,
    "evaluating": 0.75,
    "learned": 1.0,
}


def build_skill_table(
    stage_id: str,
    previous_table: dict[str, dict[str, Any]] | None = None,
    evaluation_mode: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return a stable, complete curriculum skill table for one stage."""
    stage_id = normalize_stage_id(stage_id)
    previous_table = previous_table or {}
    stage = get_stage(stage_id)
    current_index = stage_index(stage_id)
    table: dict[str, dict[str, Any]] = {}

    for skill_name in FULL_SKILL_ORDER:
        first_stage_id = SKILL_FIRST_STAGE[skill_name]
        first_index = stage_index(first_stage_id)
        previous_entry = deepcopy(previous_table.get(skill_name, {}))
        previous_value = previous_entry.get("value")
        previous_best_metric = previous_entry.get("best_metric")

        if skill_name == stage.learning_skill:
            status = "evaluating" if evaluation_mode else "learning"
            value = previous_value if previous_value is not None else "active_placeholder"
        elif first_index < current_index:
            status = "learned"
            value = previous_value if previous_value is not None else "carried_by_previous_checkpoint"
        elif skill_name in stage.active_skills:
            status = "inactive"
            value = previous_value if previous_value is not None else None
        else:
            status = "locked_future" if first_index > current_index else "not_learned"
            value = None

        table[skill_name] = {
            "status": status,
            "value": value,
            "stage": first_stage_id,
            "active": skill_name in stage.active_skills,
            "currently_learning": skill_name == stage.learning_skill,
            "best_metric": previous_best_metric,
        }

    return table


def update_skill_table_after_stage(
    skill_table: dict[str, dict[str, Any]],
    stage_id: str,
    success_metric: float,
    checkpoint_path: str | None,
    learned: bool,
) -> dict[str, dict[str, Any]]:
   
    table = deepcopy(skill_table)
    stage = get_stage(stage_id)
    skill_name = stage.learning_skill
    if skill_name in table:
        table[skill_name]["status"] = "learned" if learned else "learning"
        table[skill_name]["value"] = {
            "checkpoint_path": checkpoint_path,
            "success_metric": float(success_metric),
        }
        table[skill_name]["best_metric"] = max(
            float(table[skill_name].get("best_metric") or 0.0),
            float(success_metric),
        )
    return table


def skill_table_feature_vector(skill_table: dict[str, dict[str, Any]]) -> list[float]:
    
    features: list[float] = []
    for skill_name in FULL_SKILL_ORDER:
        entry = skill_table.get(skill_name, {})
        features.append(float(STATUS_TO_FEATURE.get(str(entry.get("status", "not_learned")), 0.0)))
        features.append(1.0 if entry.get("active", False) else 0.0)
    return features


def describe_stage_skills(stage_id: str) -> list[str]:
    table = build_skill_table(stage_id)
    lines = []
    for name in FULL_SKILL_ORDER:
        entry = table[name]
        lines.append(f"{name}: {entry['status']} (stage {entry['stage']})")
    return lines
