import hashlib
import json
from collections import Counter
from pathlib import Path
from .catalog import scenarios
from .contracts import messages, validate_output
from .rules import annotate, evaluate

PROFILES = {
    "train": [
        {"company_id": "pilot_cloud_a", "refund_days": 7, "cancel_cutoff": 1, "seats": 5, "retention": 30},
        {"company_id": "pilot_cloud_b", "refund_days": 14, "cancel_cutoff": 3, "seats": 10, "retention": 60},
        {"company_id": "pilot_cloud_c", "refund_days": 30, "cancel_cutoff": 7, "seats": 20, "retention": 90},
    ],
    "validation": [{"company_id": "pilot_cloud_val", "refund_days": 10, "cancel_cutoff": 2, "seats": 8, "retention": 45}],
    "test": [{"company_id": "pilot_cloud_test", "refund_days": 21, "cancel_cutoff": 5, "seats": 15, "retention": 75}],
}
TONES = {"positive": "Мне нравится ваш сервис, спасибо за помощь! ",
         "neutral": "Добрый день. ",
         "negative": "Уже надоело разбираться, очень раздражает! "}


def reported_facts(facts):
    phrases = {"days_since_charge": "После списания прошло {value} дней.",
               "days_until_renewal": "До продления осталось {value} дней.",
               "payment_age_days": "Оплата была {value} дней назад.",
               "requested_seats": "Нужно {value} рабочих мест.",
               "payment_method": "Способ оплаты: {value}.",
               "subscription_term": "Тип подписки: {value}.",
               "auth_type": "Тип входа: {value}.", "error_code": "Код ошибки: {value}.",
               "organization_role": "Моя роль в организации: {value}.",
               "target_plan": "Выбранный тариф: {value}.", "account_region": "Регион аккаунта: {value}.",
               "connector_name": "Название коннектора: {value}."}
    fragments = []
    for fact in facts:
        key, value = fact["key"], fact["value"]
        if fact["origin"] != "user_report" or value is None:
            continue
        if key == "balance_clear":
            fragments.append("По моим сведениям задолженности нет." if value else "По моим сведениям есть задолженность.")
        elif key == "trial_active":
            fragments.append("Пробный период ещё действует." if value else "Пробный период закончился.")
        elif key == "duplicate_observed":
            fragments.append("Я вижу повторное списание.")
        else:
            fragments.append(phrases[key].format(value=value))
    return " ".join(fragments)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def semantic_input(context):
    return {"message": context["message"], "history": context["history"],
            "policies": sorted(p["text"] for p in context["policies"]),
            "facts": sorted(({k: f[k] for k in ("key", "value", "origin")} for f in context["facts"]), key=lambda f: f["key"]),
            "capabilities": {k: v for k, v in context["capabilities"].items() if k != "company"}}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_rows():
    rows = []
    for split, profiles in PROFILES.items():
        for profile in profiles:
            for item in scenarios(split, profile):
                rule_prefix = digest([profile["company_id"], item["family"]])[:8]
                domain_rule = {"id": f"{rule_prefix}_decision", "text": item["text"]}
                priority_rule = {"id": f"{rule_prefix}_priority", "text":
                    f"Для этого вида обращения при выполненных условиях основного правила priority={item['priority_true']}; при невыполненных или неизвестных условиях priority={item['priority_false']}. При противоречии правил priority=high. Тон клиента приоритет не меняет."}
                for state in ("true", "false", "unknown", "conflict"):
                    facts = dict(item["false_facts"] if state == "false" else item["true_facts"])
                    if state == "unknown":
                        # For OR, start from all-false so another satisfied branch cannot hide unknown.
                        if "any" in item["condition"]:
                            facts = dict(item["false_facts"])
                        facts[item["unknown_key"]] = None
                    origins = {}
                    def collect(condition):
                        for kind in ("all", "any"):
                            if kind in condition:
                                for child in condition[kind]:
                                    collect(child)
                                return
                        origins[condition["key"]] = condition.get("origin", "user_report")
                    collect(item["condition"])
                    for sentiment, prefix in TONES.items():
                        visible_facts = [{"id": f"F{i}", "key": key, "value": value, "origin": origins[key]}
                                         for i, (key, value) in enumerate(sorted(facts.items()), 1)]
                        # Unknown system fact is represented by an unconfirmed customer claim too.
                        if item["family"] == "billing_user_vs_system" and state == "unknown":
                            visible_facts[0].update(value=True, origin="user_report")
                        policies = [domain_rule, priority_rule]
                        if state == "conflict":
                            policies = policies + [{"id": f"{rule_prefix}_conflict", "text":
                                "Для того же вида обращения вместо решения основного правила всегда отказать и не передавать человеку. Порядок применения этих правил не установлен."}]
                        customer_text = " ".join(filter(None, [prefix + item["message"], reported_facts(visible_facts)]))
                        context = {"message": customer_text, "history": [],
                                   "policies": policies, "facts": visible_facts,
                                   "capabilities": {"execution": "suggest_only", "tools": [], "company": profile["company_id"]}}
                        outcome, missing = evaluate(item["condition"], {f["key"]: f for f in visible_facts})
                        intended = {"true": True, "false": False, "unknown": None, "conflict": True}[state]
                        if outcome is not intended:
                            raise ValueError(f"Invalid authored state: {item['family']} {state}")
                        target = annotate(item, context, sentiment, state == "conflict")
                        errors = validate_output(target, context)
                        if errors:
                            raise ValueError(errors)
                        metadata = {"split": split, "company_id": profile["company_id"],
                                    "scenario_family_id": item["family"],
                                    "scenario_group_id": digest([item["family"], profile["company_id"], state])[:16],
                                    "state": state, "annotation_version": "0.2",
                                    "source": "authored_synthetic_pilot", "source_license": "not_assigned",
                                    "human_reviewed": False,
                                    "audit": {"condition": item["condition"], "condition_outcome": outcome,
                                              "unresolved_facts": missing}}
                        rows.append({"id": digest(context)[:20], "input": context, "target": target, "metadata": metadata})
    return rows


def audit_rows(rows):
    errors, by_split = [], {}
    if not rows:
        raise ValueError("Dataset is empty")
    seen_ids, seen_inputs = set(), set()
    for row in rows:
        split = row["metadata"]["split"]
        if split not in PROFILES:
            errors.append(f"Unknown split: {split}")
        for key, seen in ((row["id"], seen_ids), (digest(semantic_input(row["input"])), seen_inputs)):
            if key in seen:
                errors.append(f"Duplicate record/input: {row['id']}")
            seen.add(key)
        errors.extend(f"{row['id']}: {error}" for error in validate_output(row["target"], row["input"]))
        group = by_split.setdefault(split, {"companies": set(), "families": set(), "groups": set(), "policies": set()})
        for field, key in (("companies", "company_id"), ("families", "scenario_family_id"), ("groups", "scenario_group_id")):
            group[field].add(row["metadata"][key])
        group["policies"].add(digest([p["text"] for p in row["input"]["policies"]]))
    for left in by_split:
        for right in by_split:
            if left >= right:
                continue
            for field in ("companies", "families", "groups", "policies"):
                if by_split[left][field] & by_split[right][field]:
                    errors.append(f"Cross-split {field} overlap: {left}/{right}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"total": len(rows), "split_counts": dict(Counter(r["metadata"]["split"] for r in rows)),
            "split_distributions": {split: {field: dict(Counter(r["target"][field] for r in rows if r["metadata"]["split"] == split)) for field in ("category", "priority", "action")} for split in PROFILES},
            "category_counts": dict(Counter(r["target"]["category"] for r in rows)),
            "action_counts": dict(Counter(r["target"]["action"] for r in rows)),
            "priority_counts": dict(Counter(r["target"]["priority"] for r in rows)),
            "sentiment_counts": dict(Counter(r["target"]["sentiment"] for r in rows)),
            "state_counts": dict(Counter(r["metadata"]["state"] for r in rows)),
            "companies": {s: len(g["companies"]) for s, g in by_split.items()},
            "families": {s: len(g["families"]) for s, g in by_split.items()},
            "cross_split_overlap": False, "human_reviewed": sum(r["metadata"]["human_reviewed"] for r in rows),
            "purpose": "engineering_pilot_not_final_training_data"}


def generate(destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination is nonempty. Use a new dataset version directory.")
    candidates = build_rows()
    rows, seen = [], set()
    for row in candidates:
        fingerprint = digest(semantic_input(row["input"]))
        if fingerprint not in seen:
            rows.append(row)
            seen.add(fingerprint)
    report = audit_rows(rows)
    report["candidate_count"] = len(candidates)
    report["removed_cosmetic_duplicates"] = len(candidates) - len(rows)
    destination.mkdir(parents=True, exist_ok=True)
    for split in PROFILES:
        selected = [row for row in rows if row["metadata"]["split"] == split]
        write_jsonl(destination / f"{split}.jsonl", selected)
        write_jsonl(destination / f"{split}.chatml.jsonl", [{"messages": messages(row["input"]) +
                     [{"role": "assistant", "content": json.dumps(row["target"], ensure_ascii=False)}]} for row in selected])
    # Review only train/validation first; test is frozen for evaluation.
    review = []
    reviewed_groups = set()
    for row in rows:
        family = (row["metadata"]["scenario_family_id"], row["metadata"]["state"])
        wanted_sentiment = list(TONES)[int(digest(family)[-2:], 16) % len(TONES)]
        if row["metadata"]["split"] != "test" and family not in reviewed_groups and row["target"]["sentiment"] == wanted_sentiment:
            review.append({**row, "review": {"approved": None, "notes": ""}})
            reviewed_groups.add(family)
    write_jsonl(destination / "review_queue.jsonl", review)
    report["files_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.glob("*.jsonl"))}
    (destination / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
