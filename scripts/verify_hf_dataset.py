"""Load public parquet configurations anonymously and validate restored contexts."""
import json
from pathlib import Path
from datasets import load_dataset
from support_json.contracts import validate_output

root = Path(__file__).resolve().parents[1]
report = {'repo_id':'A11Sunday/support-json-ru', 'anonymous':True, 'configs':{}}
for config, expected in {
    'selected':{'train':512,'validation':96,'test':108},
    'pilot':{'train':192,'validation':96,'test':108},
    'expanded_trained':{'train':1024}, 'expanded_candidates':{'train':10000}
}.items():
    ds = load_dataset(report['repo_id'], config, token=False)
    assert {key:len(value) for key,value in ds.items()} == expected
    for split in ds:
        for row in ds[split]:
            context = {'message':row['message'], **{key:json.loads(row[key+'_json'])
                for key in ('policies','facts','history','capabilities')}}
            target = json.loads(row['target_json'])
            assert not validate_output(target, context), row['id']
            assert len(row['messages']) == 3
            assert row['messages'][-1]['role'] == 'assistant'
            assert json.loads(row['messages'][-1]['content']) == target
    report['configs'][config] = {'splits':expected,'all_restored_contracts_valid':True}
    (root/'reports/hf-dataset-download.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(config+': anonymous load and all contracts verified',flush=True)
