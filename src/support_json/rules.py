"""Small three-valued annotation engine, not an LLM or production agent."""


def evaluate(condition, facts):
    """Return (True/False/None, missing keys). Known outcomes short-circuit unknowns."""
    if "all" in condition or "any" in condition:
        kind = "all" if "all" in condition else "any"
        results = [evaluate(child, facts) for child in condition[kind]]
        decisive = False if kind == "all" else True
        if any(value is decisive for value, _ in results):
            return decisive, []
        missing = sorted({key for _, keys in results for key in keys})
        if any(value is None for value, _ in results):
            return None, missing
        return not decisive, []
    key = condition["key"]
    fact = facts.get(key)
    if fact is None or fact["value"] is None:
        return None, [key]
    if condition.get("origin") and fact["origin"] != condition["origin"]:
        return None, [key]
    value, expected = fact["value"], condition["value"]
    op = condition["op"]
    if op == "eq":
        return type(value) is type(expected) and value == expected, []
    if op == "in":
        return any(type(value) is type(item) and value == item for item in expected), []
    if op in {"le", "ge"}:
        if type(value) not in (int, float) or type(expected) not in (int, float):
            raise ValueError(f"Numeric condition requires numbers: {key}")
        return (value <= expected if op == "le" else value >= expected), []
    raise ValueError(f"Unknown operator: {op}")


def annotate(spec, context, sentiment, conflict=False):
    values = {fact["key"]: fact for fact in context["facts"]}
    outcome, missing = evaluate(spec["condition"], values)
    priority = spec["priority_true"] if outcome is True else spec["priority_false"]
    if conflict:
        action, operation, reply, missing = "escalate", "human_review", "В правилах есть противоречие. Для решения этого обращения требуется специалист поддержки.", []
        priority = "high"
    elif outcome is None:
        if spec["unknown_action"] == "ask_details":
            action, operation = "ask_details", "request_details"
            labels = [spec["fact_labels"][key] for key in missing]
            reply = "Уточните, пожалуйста: " + "; ".join(labels) + ". Это нужно для проверки условий компании."
        else:
            action, operation, missing = "escalate", "human_review", []
            reply = "Доступных подтверждённых данных недостаточно. Проверка требует участия специалиста поддержки."
    else:
        branch = spec["true"] if outcome else spec["false"]
        action, operation, reply = branch
        missing = []
    ids = [rule["id"] for rule in context["policies"]]
    return {"category": spec["category"], "priority": priority,
            "sentiment": sentiment, "action": action,
            "recommended_action": operation, "reply": reply,
            "missing_info": missing, "supported_rule_ids": ids,
            "human_escalation": action == "escalate"}
