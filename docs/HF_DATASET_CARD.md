---
license: cc-by-4.0
language:
- ru
task_categories:
- text-generation
- text-classification
tags:
- synthetic
- support
- policy-following
- structured-output
- lora
size_categories:
- 10K<n<100K
configs:
- config_name: selected
  default: true
  data_files:
  - split: train
    path: data/selected/train.parquet
  - split: validation
    path: data/selected/validation.parquet
  - split: test
    path: data/selected/test.parquet
- config_name: pilot
  data_files:
  - split: train
    path: data/pilot/train.parquet
  - split: validation
    path: data/pilot/validation.parquet
  - split: test
    path: data/pilot/test.parquet
- config_name: expanded_trained
  data_files:
  - split: train
    path: data/expanded_trained/train.parquet
- config_name: expanded_candidates
  data_files:
  - split: train
    path: data/expanded_candidates/train.parquet
---

# Support-JSON-RU

Synthetic Russian SaaS support data for policy-conditioned JSON decisions and draft replies. The task supplies customer text, company policies, sourced facts and available capabilities; the model predicts a nine-field decision rather than memorizing a single company's policy.

**Русский SaaS-support: обращение + правила + факты → категория, приоритет, настроение, действие, черновик ответа и эскалация.**

[Model](https://huggingface.co/A11Sunday/support-json-qwen3.5-4b-lora) · [Dataset files](https://huggingface.co/datasets/A11Sunday/support-json-ru/tree/main) · [License](LICENSE.txt)

## Configurations

| Configuration | Train | Validation | Test | Purpose |
|---|---:|---:|---:|---|
| **selected**, default | **512** | **96** | **108** | Exact rows used by the selected third adapter and its evaluation |
| pilot | 192 | 96 | 108 | Initial policy-conditioned pilot |
| expanded_trained | 1,024 | — | — | Exact training subset used by the second adapter |
| expanded_candidates | 10,000 | — | — | Additional synthetic candidates; the full set was not trained |

Configurations overlap. The pilot train is included in the selected train and candidate set; the evaluation splits are shared. **Do not concatenate these configurations as independent examples or train on validation/test.** The 10,000 candidates comprise approximately 5,685 authored scenario groups, not 10,000 independent customer conversations.

## Load the data

```python
from datasets import load_dataset
import json

ds = load_dataset("A11Sunday/support-json-ru", "selected")
row = ds["train"][0]
context = {
    "message": row["message"],
    "policies": json.loads(row["policies_json"]),
    "facts": json.loads(row["facts_json"]),
    "history": json.loads(row["history_json"]),
    "capabilities": json.loads(row["capabilities_json"]),
}
target = json.loads(row["target_json"])
training_messages = row["messages"]  # system, user, assistant; apply the native Qwen template
```

The `*_json` columns preserve heterogeneous fact values: strings, numbers, booleans and null. A uniform Parquet representation avoids converting unknown or boolean facts into strings. `json.loads` recovers their original types. `messages` contains the complete training conversation, including company context and the gold assistant JSON. For inference, **exclude the assistant gold message**. Use response-only loss masking when training.

`target_json` and the nine individually typed target columns are equivalent. `metadata_json` includes provenance, scenario groups and executable annotation conditions; it is for auditing, **not model input**. Historical metadata may say `source_license: not_assigned`: that records the pre-publication snapshot. The release is now licensed CC BY 4.0, as indicated by `publication_license` and this card.

## Output contract

| Field | Meaning |
|---|---|
| category | billing, access, technical, refund, cancellation, integration, product, other |
| priority | low, medium, high, critical |
| sentiment | positive, neutral, negative |
| action | answer, ask_details, propose_action, escalate |
| recommended_action | Concrete proposed operation or information / clarification / human review |
| reply | Russian draft for an operator to review |
| missing_info | Facts that still determine the decision |
| supported_rule_ids | Identifiers from the provided policies |
| human_escalation | Whether human handling is required |

The model only proposes operations. The data does not provide real refund, cancellation or account-management tools. `user_report` identifies customer claims; `tool_observation` identifies system observations; null means unknown.

## Origin and annotation

Created for A11Sunday's fine-tuning capstone. No scraped customer conversations, external support corpus or private company records were included. Scenarios, policies and text templates were authored for this project with AI assistance. Decisions are computed from formal conditions, including equality, boundaries, membership, AND / OR, unknown values, source requirements and policy conflicts. Templates provide customer wording and gold replies; a large teacher model did not label all records.

Emotional tone is varied while the policy defines priority. Some cases permit a proposed action, deny eligibility, require missing facts or contain conflicting policies. Expanded data adds numeric boundaries and additional rule/fact structures. These are synthetic variations, not proof of natural conversational coverage.

Scenario groups, company identifiers and authored policy families were separated before evaluation. All Parquet exports were checked row by row against the frozen JSONL snapshots, including type-preserving reconstruction and target validity. Train / validation / test scenario groups in the selected and pilot configurations are disjoint. See [the export manifest](dataset-manifest.json).

## Limitations and responsible use

- All labels are programmatic/template-based. `human_reviewed` is false; there is no independently verified annotation quality.
- Russian wording is mostly templated. Training histories are empty. Real slang, multi-turn dialogue, policy ambiguity and unseen integrations need additional testing.
- Authored families can share semantic templates across splits; group separation is not a guarantee of semantic novelty.
- The published test has already been evaluated and analyzed. It is suitable for reproducibility, **not future tuning**. Create a fresh closed test for subsequent models.
- A small synthetic stress suite is not representative company traffic. The selected LoRA still makes factual/policy mistakes, including 5 flagged replies out of 36 AI-assisted reviewed pairs.
- Do not treat dataset gold replies or model output as permission to execute operations, legal advice or verified facts about an actual customer's account.

## Attribution

Author: **A11Sunday**, 2026. License: **CC BY 4.0**. Attribute the dataset and indicate modifications when redistributing adapted data. Code and adapter have separate Apache-2.0 licenses. [Full preparation notes](documentation/DATASET.md) · [Expansion](documentation/DATASET_EXPANSION.md) · [Evaluation definitions](documentation/EVALUATION.md).
