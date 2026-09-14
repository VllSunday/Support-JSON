import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .contracts import messages, parse_output, validate_output
from .data import digest, read_jsonl, write_jsonl

FIELDS = ("category", "priority", "sentiment", "action", "recommended_action", "human_escalation")


def score(rows, predictions, human_reviews=None):
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    row_ids = {row["id"] for row in rows}
    if len(row_ids) != len(rows):
        raise ValueError("Duplicate dataset ids")
    outputs = {}
    for record in predictions:
        if record["id"] not in row_ids or record["id"] in outputs:
            raise ValueError("Unknown or duplicate prediction id")
        if not isinstance(record["output"], str):
            raise ValueError("output must be raw model text")
        outputs[record["id"]] = record["output"]
    correct, field_only, parsed = Counter(), Counter(), {}
    details = []
    for row in rows:
        target, prediction, errors = row["target"], None, []
        raw = outputs.get(row["id"])
        if raw is None:
            errors = ["Missing prediction"]
        else:
            try:
                prediction = parse_output(raw)
                correct["json_parse"] += 1
                errors = validate_output(prediction, row["input"])
            except (ValueError, TypeError) as error:
                errors = [str(error)]
        if isinstance(prediction, dict):
            for field in FIELDS:
                value = prediction.get(field)
                field_only[field] += type(value) is type(target[field]) and value == target[field]
        if not errors:
            correct["schema_and_invariants"] += 1
            parsed[row["id"]] = prediction
            for field in FIELDS:
                correct[field] += prediction[field] == target[field]
            for field in ("missing_info", "supported_rule_ids"):
                correct[field] += set(prediction[field]) == set(target[field])
        details.append({"id": row["id"], "errors": errors,
                        "field_correct": {field: not errors and prediction[field] == target[field] for field in FIELDS}})
    n = len(rows)
    result = {"n": n, "received_predictions": len(outputs), "missing_predictions": n - len(outputs),
              "accuracy": {field: correct[field] / n for field in FIELDS},
              "field_only_accuracy": {field: field_only[field] / n for field in FIELDS},
              "json_parse_rate": correct["json_parse"] / n,
              "schema_and_invariants_rate": correct["schema_and_invariants"] / n,
              "missing_info_set_accuracy": correct["missing_info"] / n,
              "supported_rule_ids_set_accuracy": correct["supported_rule_ids"] / n}
    result["macro_f1"] = {}
    for field in ("category", "priority"):
        labels = sorted({row["target"][field] for row in rows})
        f1s = []
        for label in labels:
            tp = fp = fn = 0
            for row in rows:
                gold = row["target"][field] == label
                predicted = parsed.get(row["id"], {}).get(field) == label
                tp += gold and predicted
                fp += not gold and predicted
                fn += gold and not predicted
            f1s.append(2 * tp / (2 * tp + fp + fn))
        result["macro_f1"][field] = {"value": sum(f1s) / len(f1s), "labels_present_in_gold": labels}
    critical = [row for row in rows if row["target"]["priority"] == "critical"]
    critical_correct = sum(parsed.get(row["id"], {}).get("priority") == "critical" for row in critical)
    result["critical_recall"] = {"value": critical_correct / len(critical) if critical else None,
                                 "correct": critical_correct, "total": len(critical)}
    result["by_state"] = {}
    for state in sorted({row["metadata"]["state"] for row in rows}):
        selected = [row for row in rows if row["metadata"]["state"] == state]
        result["by_state"][state] = {"n": len(selected), "action_accuracy": sum(parsed.get(row["id"], {}).get("action") == row["target"]["action"] for row in selected) / len(selected)}
    reviewed = {}
    for review in human_reviews or []:
        if review["id"] not in row_ids or review["id"] in reviewed:
            raise ValueError("Unknown or duplicate human review id")
        if type(review.get("hallucination")) is not bool or type(review.get("response_quality")) is not int or not 1 <= review["response_quality"] <= 5:
            raise ValueError("Review needs boolean hallucination and integer response_quality 1..5")
        reviewed[review["id"]] = review
    m = len(reviewed)
    result["human_evaluation"] = {"reviewed": m, "coverage": m/n,
        "hallucination_rate": sum(r["hallucination"] for r in reviewed.values()) / m if m else None,
        "mean_response_quality_1_to_5": sum(r["response_quality"] for r in reviewed.values()) / m if m else None}
    result["dataset_fingerprint"] = digest(rows)
    result['scoring_version'] = '0.4'
    result["note"] = "Strict accuracy: invalid or missing outputs fail all task fields. Field-only accuracy: each exact, correctly typed field is scored independently, even when other fields violate the contract. Human rates cover reviewed subset only; null means not measured. Reply exact match is not used."
    return result, details


def run_local(dataset, destination, base_url, model):
    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only local HTTP model endpoints are supported in this pilot")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Prediction output already exists; use a new run filename")
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("Dataset is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        payload = {"model": model, "messages": messages(row["input"]), "temperature": 0,
                   "max_tokens": 700, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        request = Request(base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=120) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Endpoint returned non-text content")
            record = {"id": row["id"], "output": content, "elapsed_seconds": round(time.perf_counter()-started, 3), "usage": body.get("usage")}
        except Exception:
            # Preserve completed work, but do not silently drop failed requests.
            if records:
                write_jsonl(destination, records)
            raise
        records.append(record)
    write_jsonl(destination, records)
    run = {"model": model, "base_url": base_url, "dataset_fingerprint": digest(rows),
           "settings": {"temperature": 0, "max_tokens": 700, "enable_thinking": False, "constrained_json": False},
           "n": len(records), "note": "Backend must be checked to honor non-thinking mode. No schema constraint or output repair applied."}
    destination.with_suffix(".run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return run
