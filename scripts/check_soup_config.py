"""Validate config locally without loading torch, weights, or a GPU."""
import importlib.metadata
import json
from pathlib import Path
from soup_cli.config.loader import load_config

root = Path(__file__).resolve().parents[1]
config = load_config(root / "configs/soup-pilot.yaml")
assert config.data.val_split == 0
assert config.training.quantization == "none"
assert config.training.load_in_16bit is True
assert config.modality == "text"
assert config.data.train_on_responses_only and not config.data.train_on_prompt
assert (root / config.data.train).exists()
assert "test" not in config.data.train and "validation" not in config.data.train
print(json.dumps({"soup_version": importlib.metadata.version("soup-cli"),
                  "config_valid": True, "base": config.base,
                  "validation": "external_grouped_validation",
                  "gpu_execution_verified": False}, indent=2))
