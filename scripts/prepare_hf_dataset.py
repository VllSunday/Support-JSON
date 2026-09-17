"""Build faithful, viewer-friendly Parquet configurations without changing training snapshots."""
import hashlib
import json
import shutil
from pathlib import Path

from datasets import Dataset
from support_json.contracts import messages, validate_output
from support_json.data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'.build/hf-release/dataset'
CONFIGS = {
    'selected': {'train':'data/diverse-v0.5/train.jsonl',
                 'validation':'data/pilot-v0.2/validation.jsonl', 'test':'data/pilot-v0.2/test.jsonl'},
    'pilot': {'train':'data/pilot-v0.2/train.jsonl',
              'validation':'data/pilot-v0.2/validation.jsonl', 'test':'data/pilot-v0.2/test.jsonl'},
    'expanded_trained': {'train':'data/expanded-v0.4/train-1024.jsonl'},
    'expanded_candidates': {'train':'data/expanded-v0.4/train.jsonl'},
}


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def convert(row, split):
    assert not validate_output(row['target'], row['input']), row['id']
    context = row['input']
    target = row['target']
    return {
        'id':row['id'], 'split':split, 'message':context['message'],
        'policies_json':dump(context['policies']), 'facts_json':dump(context['facts']),
        'history_json':dump(context.get('history',[])), 'capabilities_json':dump(context['capabilities']),
        'target_json':dump(target), 'metadata_json':dump(row['metadata']),
        'publication_license':'CC-BY-4.0',
        **target,
        'messages':messages(context)+[{'role':'assistant','content':dump(target)}],
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {'repo_id':'A11Sunday/support-json-ru', 'license':'cc-by-4.0',
                'author':'A11Sunday', 'configs':{}, 'training_snapshots_modified':False}
    for config, splits in CONFIGS.items():
        manifest['configs'][config] = {}
        groups = {}
        for split, relative in splits.items():
            source = ROOT/relative
            raw = read_jsonl(source)
            rows = [convert(row,split) for row in raw]
            assert len({r['id'] for r in rows}) == len(rows)
            folder = DEST/'data'/config
            folder.mkdir(parents=True, exist_ok=True)
            parquet = folder/f'{split}.parquet'
            Dataset.from_list(rows).to_parquet(str(parquet))
            loaded = Dataset.from_parquet(str(parquet))
            assert len(loaded) == len(raw)
            # Check every row preserves types, labels, history and trusted input.
            for original, actual in zip(raw,loaded):
                for key in ('policies','facts','history','capabilities'):
                    assert json.loads(actual[key+'_json']) == original['input'].get(key,[])
                assert json.loads(actual['target_json']) == original['target']
                assert json.loads(actual['metadata_json']) == original['metadata']
                assert actual['messages'] == messages(original['input'])+[{'role':'assistant','content':dump(original['target'])}]
            groups[split] = {row['metadata']['scenario_group_id'] for row in raw}
            manifest['configs'][config][split] = {
                'rows':len(raw), 'source':relative,
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'parquet_sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),
                'all_rows_roundtrip_verified':True,
            }
        if {'train','validation','test'} <= groups.keys():
            assert not groups['train'] & groups['validation']
            assert not groups['train'] & groups['test']
            assert not groups['validation'] & groups['test']
    for name in ('DATASET.md','DATASET_EXPANSION.md','EVALUATION.md','RESPONSE_REVIEW.md'):
        folder=DEST/'documentation'
        folder.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'docs'/name,folder/name)
    shutil.copyfile(ROOT/'LICENSE.dataset.txt', DEST/'LICENSE.txt')
    manifest_path = DEST/'dataset-manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/hf-dataset-export.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({c:{s:v['rows'] for s,v in splits.items()} for c,splits in manifest['configs'].items()}))


if __name__ == '__main__':
    main()
