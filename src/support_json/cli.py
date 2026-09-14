import argparse
import json
from pathlib import Path
from .data import PROFILES, audit_rows, generate, read_jsonl, write_jsonl
from .evaluation import run_local, score


def main():
    parser = argparse.ArgumentParser(description="Support-JSON: pilot data and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", default="data/pilot-v0.2")
    audit = sub.add_parser("audit")
    audit.add_argument("--data", default="data/pilot-v0.2")
    check = sub.add_parser("selfcheck")
    check.add_argument("--dataset", default="data/pilot-v0.2/validation.jsonl")
    check.add_argument("--output", default="reports/harness-selfcheck-v0.2.json")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--human-reviews")
    evaluate.add_argument("--output", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--dataset", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    predict.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate(args.output)
        elif args.command == "audit":
            folder = Path(args.data)
            result = audit_rows([row for split in PROFILES for row in read_jsonl(folder / f"{split}.jsonl")])
        elif args.command == "predict":
            result = run_local(args.dataset, args.output, args.base_url, args.model)
        else:
            rows = read_jsonl(args.dataset)
            if args.command == "selfcheck":
                predictions = [{"id": row["id"], "output": json.dumps(row["target"], ensure_ascii=False)} for row in rows]
                reviews = None
            else:
                predictions = read_jsonl(args.predictions)
                reviews = read_jsonl(args.human_reviews) if args.human_reviews else None
            result, details = score(rows, predictions, reviews)
            result["run_type"] = "harness_selfcheck_not_model_benchmark" if args.command == "selfcheck" else "model_evaluation"
            output = Path(args.output)
            if output.exists():
                raise ValueError("Report already exists; use a new run filename")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            write_jsonl(output.with_suffix(".details.jsonl"), details)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
