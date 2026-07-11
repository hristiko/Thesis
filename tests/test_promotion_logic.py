import unittest

from training.promotion import evaluate_stage8_phase

PHASE_8A = {
    "phase": "8A",
    "phase_index": 1,
    "required_success": 0.30,
    "required_metric": 0.35,
    "max_death_rate": 0.95,
    "min_reload_quality": 0.60,
    "max_avg_early_reloads": 0.35,
    "max_avg_empty_ammo_steps": 60.0,
    "required_eval_checks": 3,
}


def make_summary(success, score, death=0.0, reload_q=0.8, early=0.0, empty=0.0):
    return {
        "success_rate": success,
        "validation_score": score,
        "mean_metrics": {
            "deaths": death,
            "stage8_reload_quality": reload_q,
            "stage8_early_reloads": early,
            "stage8_empty_ammo_steps": empty,
        },
    }


class PromotionLogicTests(unittest.TestCase):
    def test_stage8_phase_passes_when_all_thresholds_met(self):
        decision = evaluate_stage8_phase(PHASE_8A, make_summary(0.5, 0.8))
        self.assertTrue(decision.passed)

    def test_stage8_phase_fails_on_low_score(self):
        decision = evaluate_stage8_phase(PHASE_8A, make_summary(0.5, 0.2))
        self.assertFalse(decision.passed)
        self.assertTrue(any("validation_score" in reason for reason in decision.fail_reasons))

    def test_stage8_phase_fails_on_early_reload_excess(self):
        decision = evaluate_stage8_phase(PHASE_8A, make_summary(0.5, 0.8, early=0.5))
        self.assertFalse(decision.passed)
        self.assertTrue(any("avg_early_reloads" in reason for reason in decision.fail_reasons))


if __name__ == "__main__":
    unittest.main()
