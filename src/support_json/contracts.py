import json
from importlib.resources import files
from jsonschema import Draft202012Validator

RESOURCES = files("support_json").joinpath("resources")
SCHEMA = json.loads(RESOURCES.joinpath("output.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
PROMPT = RESOURCES.joinpath("system_prompt.txt").read_text(encoding="utf-8").strip()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def parse_output(raw):
    return json.loads(raw, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_output(target, context):
    errors = [error.message for error in VALIDATOR.iter_errors(target)]
    if errors:
        return errors
    if target["human_escalation"] != (target["action"] == "escalate"):
        errors.append("action and human_escalation disagree")
    rule_ids = {rule["id"] for rule in context["policies"]}
    if not set(target["supported_rule_ids"]) <= rule_ids:
        errors.append("Unknown supported_rule_ids")
    if (target["action"] == "ask_details") != bool(target["missing_info"]):
        errors.append("missing_info must be nonempty exactly for ask_details in v0.2")
    expected = {"answer": {"explain_policy"}, "ask_details": {"request_details"},
                "escalate": {"human_review"}}
    allowed = expected.get(target["action"])
    if allowed is not None and target["recommended_action"] not in allowed:
        errors.append("recommended_action disagrees with action")
    if target["action"] == "propose_action" and not target["recommended_action"].startswith("propose_"):
        errors.append("propose_action requires a concrete proposed operation")
    return errors


def messages(context, examples=None):
    trusted = {key: context[key] for key in ("policies", "facts", "capabilities")}
    customer = {key: context[key] for key in ("message", "history")}
    # Metadata, gold targets and executable annotation rules never reach the model.
    system = PROMPT + "\nСхема:\n" + json.dumps(SCHEMA, ensure_ascii=False)
    if examples:
        system += '\nОбразцы других обращений и заполненных результатов (их факты не относятся к текущему клиенту):\n'
        system += json.dumps(examples, ensure_ascii=False)
        system += '\nДля текущего обращения используй только следующий доверенный контекст и сообщение клиента.\n'
    system += "\nДоверенный контекст:\n" + json.dumps(trusted, ensure_ascii=False)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(customer, ensure_ascii=False)}]
