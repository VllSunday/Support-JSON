import copy
import json
import tempfile
import unittest
from pathlib import Path
from support_json.catalog import leaf
from support_json.contracts import messages, parse_output, validate_output
from support_json.data import audit_rows, digest, generate, read_jsonl, semantic_input
from support_json.evaluation import score
from support_json.rules import evaluate


class RuleTests(unittest.TestCase):
    def test_refund_boundary_is_inclusive(self):
        rule = leaf("days", "le", 7)
        self.assertEqual(evaluate(rule, {"days": {"value": 7, "origin": "user_report"}}), (True, []))
        self.assertEqual(evaluate(rule, {"days": {"value": 8, "origin": "user_report"}}), (False, []))

    def test_customer_claim_is_not_system_confirmation(self):
        rule = leaf("duplicate", "eq", True, "tool_observation")
        self.assertEqual(evaluate(rule, {"duplicate": {"value": True, "origin": "user_report"}}), (None, ["duplicate"]))

    def test_false_and_unknown_needs_no_question(self):
        rule = {"all": [leaf("term", "eq", "monthly"), leaf("days", "le", 7)]}
        self.assertEqual(evaluate(rule, {"term": {"value": "annual", "origin": "user_report"}}), (False, []))

    def test_exception_can_resolve_missing_date(self):
        rule = {"any": [leaf("days", "le", 7), leaf("trial", "eq", True)]}
        self.assertEqual(evaluate(rule, {"trial": {"value": True, "origin": "user_report"}}), (True, []))

    def test_bool_is_not_a_number(self):
        with self.assertRaises(ValueError):
            evaluate(leaf("days", "le", 7), {"days": {"value": True, "origin": "user_report"}})


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "pilot"
        cls.manifest = generate(cls.folder)
        cls.rows = [row for split in ("train", "validation", "test") for row in read_jsonl(cls.folder / f"{split}.jsonl")]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_generation_is_reproducible(self):
        second = Path(self.tmp.name) / "second"
        generate(second)
        for file in self.folder.glob("*.jsonl"):
            self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())

    def test_cosmetic_duplicates_are_removed(self):
        self.assertGreater(self.manifest["removed_cosmetic_duplicates"], 0)
        self.assertEqual(len(self.rows), len({digest(semantic_input(r["input"])) for r in self.rows}))

    def test_company_leak_is_rejected(self):
        broken = copy.deepcopy(self.rows)
        test_row = next(r for r in broken if r["metadata"]["split"] == "test")
        test_row["metadata"]["company_id"] = broken[0]["metadata"]["company_id"]
        with self.assertRaisesRegex(ValueError, "companies overlap"):
            audit_rows(broken)

    def test_family_leak_is_rejected(self):
        broken = copy.deepcopy(self.rows)
        test_row = next(r for r in broken if r["metadata"]["split"] == "test")
        test_row["metadata"]["scenario_family_id"] = broken[0]["metadata"]["scenario_family_id"]
        with self.assertRaisesRegex(ValueError, "families overlap"):
            audit_rows(broken)

    def test_gold_and_audit_do_not_reach_prompt(self):
        for row in self.rows:
            prompt = messages(row["input"])
            self.assertEqual(set(prompt[0]), {"role", "content"})
            self.assertNotIn("annotation_version", json.dumps(prompt))
            self.assertNotIn("condition_outcome", json.dumps(prompt))
            self.assertNotIn("scenario_family_id", json.dumps(prompt))
            customer = json.loads(prompt[1]["content"])
            self.assertEqual(set(customer), {"message", "history"})

    def test_critical_can_have_neutral_tone(self):
        self.assertTrue(any(r["target"]["priority"] == "critical" and r["target"]["sentiment"] == "neutral" for r in self.rows))
        self.assertTrue(any(r["target"]["priority"] == "low" and r["target"]["sentiment"] == "negative" for r in self.rows))

    def test_escalation_and_rule_reference_invariants(self):
        row = self.rows[0]
        wrong = copy.deepcopy(row["target"])
        wrong["human_escalation"] = True
        self.assertTrue(validate_output(wrong, row["input"]))
        wrong = copy.deepcopy(row["target"])
        wrong["supported_rule_ids"] = ["made_up"]
        self.assertTrue(validate_output(wrong, row["input"]))

    def test_schema_rejects_extra_field(self):
        row = self.rows[0]
        self.assertTrue(validate_output({**row["target"], "confidence": 0.99}, row["input"]))

    def test_duplicate_json_keys_and_markdown_rejected(self):
        with self.assertRaises(ValueError):
            parse_output('{"priority":"low","priority":"critical"}')
        with self.assertRaises(ValueError):
            parse_output('```json\n{}\n```')

    def test_missing_predictions_count_as_failures(self):
        rows = self.rows[:2]
        metrics, _ = score(rows, [{"id": rows[0]["id"], "output": json.dumps(rows[0]["target"])}])
        self.assertEqual(metrics["accuracy"]["category"], 0.5)
        self.assertEqual(metrics["missing_predictions"], 1)

    def test_invalid_json_counts_as_failure(self):
        row = self.rows[0]
        metrics, _ = score([row], [{"id": row["id"], "output": "not json"}])
        self.assertEqual(metrics["json_parse_rate"], 0)
        self.assertEqual(metrics["accuracy"]["category"], 0)

    def test_classification_and_output_contract_are_measured_separately(self):
        row = self.rows[0]
        prediction = dict(row['target'], recommended_action='invented_operation')
        metrics, _ = score([row], [{'id': row['id'], 'output': json.dumps(prediction)}])
        self.assertEqual(metrics['accuracy']['category'], 0)
        self.assertEqual(metrics['field_only_accuracy']['category'], 1)
        self.assertEqual(metrics['field_only_accuracy']['recommended_action'], 0)

    def test_hallucination_is_unmeasured_without_reviews(self):
        row = self.rows[0]
        metrics, _ = score([row], [{"id": row["id"], "output": json.dumps(row["target"])}])
        self.assertIsNone(metrics["human_evaluation"]["hallucination_rate"])
        self.assertEqual(metrics["human_evaluation"]["coverage"], 0)

    def test_human_review_rates_are_measured_separately(self):
        rows = self.rows[:2]
        predictions = [{"id": row["id"], "output": json.dumps(row["target"])} for row in rows]
        metrics, _ = score(rows, predictions, [{"id": rows[0]["id"], "hallucination": True, "response_quality": 2}])
        self.assertEqual(metrics["human_evaluation"]["hallucination_rate"], 1)
        self.assertEqual(metrics["human_evaluation"]["coverage"], 0.5)

    def test_unknown_prediction_id_cannot_inflate_score(self):
        with self.assertRaises(ValueError):
            score([self.rows[0]], [{"id": "unknown", "output": "{}"}])


if __name__ == "__main__":
    unittest.main()
