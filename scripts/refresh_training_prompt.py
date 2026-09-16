"""Re-export training prompts after a validation-only prompt repair."""
import json
from pathlib import Path
from support_json.contracts import messages, PROMPT
from support_json.data import read_jsonl, write_jsonl, digest

rows = read_jsonl('data/pilot-v0.2/train.jsonl')
destination = Path('data/pilot-v0.3')
destination.mkdir(exist_ok=True)
path = destination/'train.chatml.jsonl'
if path.exists():
    raise ValueError('Training export already exists')
write_jsonl(path, [{'messages': messages(row['input']) +
                   [{'role': 'assistant', 'content': json.dumps(row['target'], ensure_ascii=False)}]}
                  for row in rows])
(destination/'manifest.json').write_text(json.dumps({
    'n': len(rows), 'source': 'pilot-v0.2/train.jsonl', 'source_fingerprint': digest(rows),
    'system_prompt_digest': digest(PROMPT), 'changes': 'Explicit filled-output instruction only',
    'test_targets_used': False}, indent=2), encoding='utf-8')
print(path)
