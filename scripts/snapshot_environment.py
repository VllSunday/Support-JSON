"""Record installed versions without local editable paths or secrets."""
import importlib.metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            if dist.metadata["Name"] != "support-json-capstone"}
(root / "requirements.pilot.lock.txt").write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items(), key=lambda pair: pair[0].lower())), encoding="utf-8")
checks = {"dataset": "data/pilot-v0.2/manifest.json", "unit_tests_passed": 19,
          "soup_config_valid": True, "soup_chatml_valid_rows": 192,
          "tokenizer_preflight": "reports/tokenizer-preflight.json",
          "model_benchmark_run": False, "training_run": False,
          "purpose": "completed_phase_1_data_and_evaluation_tools"}
(root / "reports/phase-1-checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"dependencies_recorded": len(versions), "checks_written": True}))
