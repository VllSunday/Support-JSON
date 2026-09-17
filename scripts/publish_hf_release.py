"""Publish reviewed staging directories using saved HF auth; never print credentials."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'reports/hf-publication.json'
REPOS = {'model':'A11Sunday/support-json-qwen3.5-4b-lora','dataset':'A11Sunday/support-json-ru'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts-only', action='store_true')
    args = parser.parse_args()
    api = HfApi()
    assert api.whoami()['name'] == 'A11Sunday', 'Unexpected publishing account'
    report = json.loads(DEST.read_text(encoding='utf-8')) if DEST.exists() else {'account':'A11Sunday','repos':{}}
    for kind in (['model'] if args.artifacts_only else ['dataset','model']):
        repo = REPOS[kind]
        if api.repo_exists(repo, repo_type=kind):
            assert report['repos'].get(kind,{}).get('repo_id') == repo, 'Refuse overwriting an unrelated existing repository'
        else:
            api.create_repo(repo, repo_type=kind, private=False, exist_ok=False)
            report['repos'][kind] = {'repo_id':repo,'created_by_this_release':True}
            DEST.write_text(json.dumps(report,indent=2), encoding='utf-8')
        commit = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=ROOT/'.build/hf-release'/kind,
            allow_patterns=['artifacts/**','evaluation/TZ_AUDIT.md'] if args.artifacts_only else None,
            commit_message='Publish Support-JSON capstone materials' if args.artifacts_only else 'Publish documented Support-JSON release')
        info = api.repo_info(repo,repo_type=kind)
        report['repos'][kind].update(commit=commit.oid, public=not info.private,
            url=f'https://huggingface.co/{"datasets/" if kind=="dataset" else ""}{repo}')
        DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{kind}: public upload completed', flush=True)
    if not args.artifacts_only:
        # Anonymous download verifies public availability and exact trained weights.
        path = hf_hub_download(REPOS['model'],'adapter_model.safetensors', token=False,
            local_dir=ROOT/'.build/hf-download-check')
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        manifest = json.loads((ROOT/'output/model/support-json-qwen3.5-4b-lora/delivery-manifest.json').read_text(encoding='utf-8'))
        assert actual == manifest['files_sha256']['adapter_model.safetensors']
        report['anonymous_adapter_download_sha256'] = actual
        report['anonymous_adapter_matches_training'] = True
        DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Anonymous adapter download hash verified',flush=True)

if __name__ == '__main__':
    main()
