from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PhaseEvalDecision:
    stage_id: str
    phase: str
    phase_index: int
    passed: bool
    score: float
    checks_required: int
    success_rate: float
    validation_score: float
    pass_reasons: list[str]
    fail_reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary_text(self) -> str:
        status = "passed" if self.passed else "blocked"
        detail_parts = [
            f"success {self.success_rate:.2%}/{self.details.get('required_success', 0.0):.2%}",
            f"score {self.validation_score:.3f}/{self.details.get('required_metric', 0.0):.3f}",
        ]
        if "death_rate" in self.details:
            detail_parts.append(
                f"death {self.details['death_rate']:.2%}/{self.details.get('max_death_rate', 0.0):.2%}"
            )
        if "wave_clear_ratio" in self.details:
            detail_parts.append(
                f"waves {self.details['waves_cleared']:.1f}/{self.details.get('enemy_waves', 0.0):.1f}"
            )
        if "reload_quality" in self.details:
            detail_parts.append(
                f"reloadQ {self.details['reload_quality']:.3f}/{self.details.get('min_reload_quality', 0.0):.3f}"
            )
        if "avg_early_reloads" in self.details:
            detail_parts.append(
                f"early {self.details['avg_early_reloads']:.2f}/{self.details.get('max_avg_early_reloads', 0.0):.2f}"
            )
        if "avg_empty_ammo_steps" in self.details:
            detail_parts.append(
                f"empty {self.details['avg_empty_ammo_steps']:.1f}/{self.details.get('max_avg_empty_ammo_steps', 0.0):.1f}"
            )
        if "avg_no_los_shot_rate" in self.details:
            detail_parts.append(
                f"noLOS {self.details['avg_no_los_shot_rate']:.2%}/{self.details.get('max_avg_no_los_shot_rate', 1.0):.2%}"
            )
        if "avg_blind_rotation_steps" in self.details:
            detail_parts.append(
                f"blindRot {self.details['avg_blind_rotation_steps']:.1f}/{self.details.get('max_avg_blind_rotation_steps', 9999.0):.1f}"
            )
        if "avg_too_close_steps" in self.details:
            detail_parts.append(
                f"tooClose {self.details['avg_too_close_steps']:.1f}/{self.details.get('max_avg_too_close_steps', 9999.0):.1f}"
            )
        if "avg_danger_close_steps" in self.details:
            detail_parts.append(
                f"dangerClose {self.details['avg_danger_close_steps']:.1f}/{self.details.get('max_avg_danger_close_steps', 9999.0):.1f}"
            )
        reason = "; ".join(self.fail_reasons if self.fail_reasons else self.pass_reasons) or "no details"
        return f"{self.phase} {status}: " + ", ".join(detail_parts) + f" | {reason}"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_stage8_phase(phase: dict[str, Any], summary: dict[str, Any], *, default_required_checks: int = 3) -> PhaseEvalDecision:
    mean_metrics = dict(summary.get("mean_metrics", {}))
    success_rate = _float(summary.get("success_rate", 0.0))
    validation_score = _float(summary.get("validation_score", 0.0))

    required_success = _float(phase.get("required_success", 1.01))
    required_metric = _float(phase.get("required_metric", 1.01))
    max_death_rate = _float(phase.get("max_death_rate", 1.0))
    min_reload_quality = _float(phase.get("min_reload_quality", 0.0))
    max_avg_early_reloads = _float(phase.get("max_avg_early_reloads", 999.0))
    max_avg_empty_ammo_steps = _float(phase.get("max_avg_empty_ammo_steps", 9999.0))
    max_avg_no_los_shot_rate = _float(phase.get("max_avg_no_los_shot_rate", 1.0))
    max_avg_no_los_shot_attempts = _float(phase.get("max_avg_no_los_shot_attempts", 9999.0))
    max_avg_blind_rotation_steps = _float(phase.get("max_avg_blind_rotation_steps", 9999.0))
    max_avg_too_close_steps = _float(phase.get("max_avg_too_close_steps", 9999.0))
    max_avg_danger_close_steps = _float(phase.get("max_avg_danger_close_steps", 9999.0))
    min_timely_reloads = _float(phase.get("min_timely_reloads", 0.0))
    checks_required = _int(phase.get("required_eval_checks", default_required_checks), default_required_checks)

    death_rate = _float(mean_metrics.get("deaths", 1.0), 1.0)
    reload_quality = _float(mean_metrics.get("stage8_reload_quality", 0.0), 0.0)
    avg_early_reloads = _float(mean_metrics.get("stage8_early_reloads", 999.0), 999.0)
    avg_empty_ammo_steps = _float(mean_metrics.get("stage8_empty_ammo_steps", 9999.0), 9999.0)
    avg_no_los_shot_attempts = _float(mean_metrics.get("stage8_no_los_shot_attempts", 0.0), 0.0)
    avg_shots_fired = _float(mean_metrics.get("shots_fired", 0.0), 0.0)
    # Use aggregate no-LOS shots divided by aggregate shots, not the mean of
    # per-episode rates. A failed episode with one no-LOS shot and almost no
    # shooting should not dominate a bridge-phase promotion decision.
    avg_no_los_shot_rate = avg_no_los_shot_attempts / max(1.0, avg_shots_fired)
    avg_blind_rotation_steps = _float(mean_metrics.get("stage8_blind_rotation_penalty_steps", 0.0), 0.0)
    avg_too_close_steps = _float(mean_metrics.get("stage8_too_close_steps", 0.0), 0.0)
    avg_danger_close_steps = _float(mean_metrics.get("stage8_danger_close_steps", 0.0), 0.0)
    timely_reloads = _float(mean_metrics.get("stage8_timely_reloads", 0.0), 0.0)
    late_rescue_reloads = _float(mean_metrics.get("stage8_late_rescue_reloads", 0.0), 0.0)
    waves_cleared = _float(mean_metrics.get("stage8_waves_cleared", mean_metrics.get("kills", 0.0)), 0.0)
    enemy_waves = max(1.0, _float(mean_metrics.get("stage8_enemy_waves", phase.get("enemy_waves", 5)), 5.0))
    wave_clear_ratio = _float(mean_metrics.get("stage8_wave_clear_ratio", waves_cleared / enemy_waves), 0.0)

    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    def check(condition: bool, ok_text: str, fail_text: str) -> None:
        if condition:
            pass_reasons.append(ok_text)
        else:
            fail_reasons.append(fail_text)

    check(success_rate + 1e-9 >= required_success, "success threshold met", f"success_rate {success_rate:.3f} < {required_success:.3f}")
    check(validation_score + 1e-9 >= required_metric, "validation score met", f"validation_score {validation_score:.3f} < {required_metric:.3f}")
    check(death_rate <= max_death_rate + 1e-9, "death rate ok", f"death_rate {death_rate:.3f} > {max_death_rate:.3f}")
    check(reload_quality + 1e-9 >= min_reload_quality, "reload quality ok", f"reload_quality {reload_quality:.3f} < {min_reload_quality:.3f}")
    check(avg_early_reloads <= max_avg_early_reloads + 1e-9, "early reloads ok", f"avg_early_reloads {avg_early_reloads:.3f} > {max_avg_early_reloads:.3f}")
    check(avg_empty_ammo_steps <= max_avg_empty_ammo_steps + 1e-9, "empty-ammo steps ok", f"avg_empty_ammo_steps {avg_empty_ammo_steps:.3f} > {max_avg_empty_ammo_steps:.3f}")
    if max_avg_no_los_shot_rate < 1.0 or max_avg_no_los_shot_attempts < 9999.0:
        check(
            avg_no_los_shot_rate <= max_avg_no_los_shot_rate + 1e-9,
            "no-LOS shot rate ok",
            f"avg_no_los_shot_rate {avg_no_los_shot_rate:.3f} > {max_avg_no_los_shot_rate:.3f}",
        )
        check(
            avg_no_los_shot_attempts <= max_avg_no_los_shot_attempts + 1e-9,
            "no-LOS shot count ok",
            f"avg_no_los_shot_attempts {avg_no_los_shot_attempts:.3f} > {max_avg_no_los_shot_attempts:.3f}",
        )
    if max_avg_blind_rotation_steps < 9999.0:
        check(
            avg_blind_rotation_steps <= max_avg_blind_rotation_steps + 1e-9,
            "blind rotation ok",
            f"avg_blind_rotation_steps {avg_blind_rotation_steps:.3f} > {max_avg_blind_rotation_steps:.3f}",
        )
    if max_avg_too_close_steps < 9999.0:
        check(
            avg_too_close_steps <= max_avg_too_close_steps + 1e-9,
            "too-close steps ok",
            f"avg_too_close_steps {avg_too_close_steps:.3f} > {max_avg_too_close_steps:.3f}",
        )
    if max_avg_danger_close_steps < 9999.0:
        check(
            avg_danger_close_steps <= max_avg_danger_close_steps + 1e-9,
            "danger-close steps ok",
            f"avg_danger_close_steps {avg_danger_close_steps:.3f} > {max_avg_danger_close_steps:.3f}",
        )
    if min_timely_reloads > 0.0:
        check(timely_reloads + 1e-9 >= min_timely_reloads, "timely reloads ok", f"timely_reloads {timely_reloads:.3f} < {min_timely_reloads:.3f}")

    details = {
        "required_success": required_success,
        "required_metric": required_metric,
        "max_death_rate": max_death_rate,
        "min_reload_quality": min_reload_quality,
        "max_avg_early_reloads": max_avg_early_reloads,
        "max_avg_empty_ammo_steps": max_avg_empty_ammo_steps,
        "max_avg_no_los_shot_rate": max_avg_no_los_shot_rate,
        "max_avg_no_los_shot_attempts": max_avg_no_los_shot_attempts,
        "max_avg_blind_rotation_steps": max_avg_blind_rotation_steps,
        "max_avg_too_close_steps": max_avg_too_close_steps,
        "max_avg_danger_close_steps": max_avg_danger_close_steps,
        "min_timely_reloads": min_timely_reloads,
        "death_rate": death_rate,
        "reload_quality": reload_quality,
        "avg_early_reloads": avg_early_reloads,
        "avg_empty_ammo_steps": avg_empty_ammo_steps,
        "avg_no_los_shot_attempts": avg_no_los_shot_attempts,
        "avg_no_los_shot_rate": avg_no_los_shot_rate,
        "avg_blind_rotation_steps": avg_blind_rotation_steps,
        "avg_too_close_steps": avg_too_close_steps,
        "avg_danger_close_steps": avg_danger_close_steps,
        "timely_reloads": timely_reloads,
        "late_rescue_reloads": late_rescue_reloads,
        "waves_cleared": waves_cleared,
        "enemy_waves": enemy_waves,
        "wave_clear_ratio": wave_clear_ratio,
    }

    return PhaseEvalDecision(
        stage_id="stage8",
        phase=str(phase.get("phase", "8A")),
        phase_index=_int(phase.get("phase_index", 1), 1),
        passed=not fail_reasons,
        score=validation_score,
        checks_required=checks_required,
        success_rate=success_rate,
        validation_score=validation_score,
        pass_reasons=pass_reasons,
        fail_reasons=fail_reasons,
        details=details,
    )
