"""Loopback-only operator demo. Model loading is lazy, with one GPU request."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .contracts import messages, parse_output, validate_output
from .data import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
FONT_FILES = {
    'plex-sans-cyrillic.woff2', 'plex-sans-latin.woff2', 'plex-sans-latin-ext.woff2',
    'plex-mono-400-cyrillic.woff2', 'plex-mono-400-latin.woff2',
    'plex-mono-500-cyrillic.woff2', 'plex-mono-500-latin.woff2',
}


def validate_context(context):
    if not isinstance(context, dict):
        raise ValueError('Контекст должен быть объектом JSON')
    if not isinstance(context.get('message'), str) or not context['message'].strip():
        raise ValueError('Введите текст обращения')
    policies = context.get('policies')
    if not isinstance(policies, list) or not policies:
        raise ValueError('Добавьте хотя бы одно правило')
    if any(not isinstance(p, dict) or not isinstance(p.get('id'), str) or not p['id'].strip()
           or not isinstance(p.get('text'), str) or not p['text'].strip() for p in policies):
        raise ValueError('У каждого правила должны быть непустые текст и id')
    if len({p['id'] for p in policies}) != len(policies):
        raise ValueError('id правил не должны повторяться')
    facts = context.get('facts', [])
    if not isinstance(facts, list):
        raise ValueError('Факты должны быть списком JSON')
    if any(not isinstance(f, dict) or not isinstance(f.get('key'), str) or not f['key'].strip()
           or 'value' not in f or f.get('origin') not in {'user_report', 'tool_observation'} for f in facts):
        raise ValueError('У каждого факта нужны key, value и origin: user_report или tool_observation')
    if len({f['key'] for f in facts}) != len(facts):
        raise ValueError('Ключи фактов не должны повторяться')
    if not isinstance(context.get('history', []), list):
        raise ValueError('История должна быть списком JSON')
    if any(not isinstance(turn, dict) or turn.get('role') not in {'user', 'assistant'}
           or not isinstance(turn.get('content'), str) or not turn['content'].strip()
           for turn in context.get('history', [])):
        raise ValueError('В истории нужны сообщения с role user/assistant и непустым content')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(ROOT/'models/qwen3.5-4b'))
    parser.add_argument('--revision', default=BASE_REVISION)
    parser.add_argument('--adapter')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    state = {'model': None, 'tokenizer': None}
    lock = threading.Lock()
    examples = read_jsonl(ROOT/'data/pilot-v0.2/validation.jsonl')

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/api/examples':
                self.send_json([{'label': f"{r['target']['category']} / {r['metadata']['state']}",
                                 'input': r['input']} for r in examples[::3]])
            elif self.path == '/api/status':
                self.send_json({'adapter_configured': bool(args.adapter),
                                'model_loaded': state['model'] is not None,
                                'mode': 'LoRA' if args.adapter else 'Base'})
            elif self.path in {'/', '/app.js'} or self.path in {
                    f'/assets/fonts/{font}' for font in FONT_FILES}:
                if self.path == '/':
                    file, content_type = ROOT/'web/index.html', 'text/html; charset=utf-8'
                elif self.path == '/app.js':
                    file, content_type = ROOT/'web/app.js', 'text/javascript; charset=utf-8'
                else:
                    file, content_type = ROOT/'web'/self.path.lstrip('/'), 'font/woff2'
                body = file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({'error': 'Страница не найдена'}, 404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.send_json({'error': 'Запрос разрешён только из локальной демки'}, 403)
                return
            if self.path != '/api/analyze':
                self.send_json({'error': 'Маршрут не найден'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
            except ValueError:
                self.send_json({'error': 'Некорректный размер запроса'}, 400)
                return
            if not 0 < length <= 100000:
                self.send_json({'error': 'Обращение и контекст слишком большие'}, 400)
                return
            if not lock.acquire(blocking=False):
                self.send_json({'error': 'Модель обрабатывает другое обращение. Повторите после завершения.'}, 409)
                return
            try:
                context = json.loads(self.rfile.read(length))
                validate_context(context)
                context.setdefault('facts', [])
                context.setdefault('history', [])
                # The demo has no execution capability, regardless of submitted data.
                context['capabilities'] = {'execution': 'suggest_only', 'tools': []}
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                started = time.perf_counter()
                if state['model'] is None:
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA недоступна. Запустите демку в GPU-окружении проекта.')
                    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
                    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                        revision=args.revision, device_map='cuda', attn_implementation='sdpa')
                    if args.adapter:
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, args.adapter)
                    model.eval()
                    state.update(model=model, tokenizer=tokenizer)
                tokenizer, model = state['tokenizer'], state['model']
                encoded = tokenizer.apply_chat_template(messages(context), tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=True,
                    return_tensors='pt').to('cuda')
                if encoded['input_ids'].shape[1] > 2048:
                    raise ValueError('Для пилота сократите правила и обращение до 2048 токенов')
                with torch.inference_mode():
                    configured_eos = model.generation_config.eos_token_id
                    stop_ids = sorted(set([tokenizer.eos_token_id] + (
                        configured_eos if isinstance(configured_eos, list) else [configured_eos])))
                    result = model.generate(**encoded, do_sample=False, max_new_tokens=700,
                                            pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
                raw = tokenizer.decode(result[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
                try:
                    parsed = parse_output(raw)
                    errors = validate_output(parsed, context)
                except (ValueError, TypeError) as error:
                    parsed, errors = None, [str(error)]
                self.send_json({'output': parsed, 'raw': raw, 'errors': errors,
                                'elapsed_seconds': round(time.perf_counter()-started, 2),
                                'mode': 'LoRA' if args.adapter else 'Base'})
            except (ValueError, KeyError, TypeError) as error:
                self.send_json({'error': str(error)}, 400)
            except Exception as error:
                print(f'Inference error: {type(error).__name__}: {error}', flush=True)
                self.send_json({'error': 'Не удалось запустить модель. Проверьте журнал демки и наличие весов.'}, 500)
            finally:
                lock.release()

    print(f'Support-JSON demo: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
