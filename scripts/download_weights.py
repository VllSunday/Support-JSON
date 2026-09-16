"""Verified, resumable HTTP ranges for networks that stall on large transfers."""
import concurrent.futures
import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_download

REPO = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DESTINATION = Path('models/qwen3.5-4b')
CHUNK = 8 * 1024 * 1024


def download_file(file):
    destination = DESTINATION / file.rfilename
    expected_hash = file.lfs.sha256
    if destination.exists() and destination.stat().st_size == file.size:
        hasher = hashlib.sha256()
        with destination.open('rb') as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b''):
                hasher.update(chunk)
        if hasher.hexdigest() == expected_hash:
            print(json.dumps({'file': file.rfilename, 'already_verified': True}), flush=True)
            return {'filename': file.rfilename, 'sha256': expected_hash, 'bytes': file.size}
    parts = DESTINATION / '.range-parts' / file.rfilename
    parts.mkdir(parents=True, exist_ok=True)
    size = file.size
    offsets = list(range(0, size, CHUNK))
    def fetch(start):
        end = min(size, start+CHUNK)-1
        path = parts / str(start)
        if path.exists() and path.stat().st_size == end-start+1:
            return end-start+1
        for attempt in range(5):
            try:
                # Distinct query keys prevent intermediaries caching another range.
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{file.rfilename}?part={start}'
                with httpx.Client(follow_redirects=True, timeout=40) as client:
                    response = client.get(url, headers={'Range': f'bytes={start}-{end}'})
                response.raise_for_status()
                if (response.status_code != 206 or len(response.content) != end-start+1
                        or response.headers.get('content-range') != f'bytes {start}-{end}/{size}'):
                    raise ValueError('Server returned an incorrect byte range')
                path.write_bytes(response.content)
                return len(response.content)
            except (httpx.HTTPError, ValueError):
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))
    done = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, offset) for offset in offsets]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            done += future.result()
            if index % 16 == 0 or index == len(offsets):
                print(json.dumps({'file': file.rfilename, 'chunks': index, 'total_chunks': len(offsets),
                                  'downloaded_gb': round(done/1e9, 3),
                                  'seconds': round(time.perf_counter()-started, 1)}), flush=True)
    hasher = hashlib.sha256()
    temporary = destination.with_suffix('.assembling')
    with temporary.open('wb') as handle:
        for offset in offsets:
            data = (parts / str(offset)).read_bytes()
            handle.write(data)
            hasher.update(data)
    if hasher.hexdigest() != expected_hash:
        raise ValueError('Downloaded weight SHA256 does not match pinned Hugging Face metadata')
    temporary.replace(destination)
    # Remove only this script's verified temporary chunks, one file at a time.
    for offset in offsets:
        (parts / str(offset)).unlink()
    print(json.dumps({'file': file.rfilename, 'verified_sha256': expected_hash, 'bytes': size}), flush=True)
    return {'filename': file.rfilename, 'sha256': expected_hash, 'bytes': size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    metadata = []
    allowed = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
               'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
               'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'LICENSE'}
    for file in info.siblings:
        if file.rfilename not in allowed:
            continue
        path = DESTINATION / file.rfilename
        if not path.exists():
            hf_hub_download(REPO, filename=file.rfilename, revision=REVISION, local_dir=DESTINATION)
        data = path.read_bytes()
        if file.lfs:
            valid = hashlib.sha256(data).hexdigest() == file.lfs.sha256
        else:
            valid = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == file.blob_id
        if not valid:
            raise ValueError(f'Pinned metadata checksum mismatch: {file.rfilename}')
        metadata.append({'filename': file.rfilename, 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest(), 'pinned_repository_hash_verified': True})
    Path('reports/model-config-manifest.json').write_text(
        json.dumps({'repo': REPO, 'revision': REVISION, 'files': metadata}, indent=2), encoding='utf-8')
    verified = []
    if not args.metadata_only:
        for file in info.siblings:
            if file.rfilename.endswith('.safetensors'):
                verified.append(download_file(file))
        Path('reports/weights-manifest.json').write_text(
            json.dumps({'repo': REPO, 'revision': REVISION, 'files': verified}, indent=2), encoding='utf-8')
    print(json.dumps({'metadata_files_verified': len(metadata), 'metadata_only': args.metadata_only}), flush=True)
