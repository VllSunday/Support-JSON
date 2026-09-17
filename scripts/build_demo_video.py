"""CPU-only animated montage of captured real application screens, not simulated UI."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
CAPTURES = OUT / 'demo-captures'
W, H, FPS, SECONDS = 1600, 900, 24, 35
BG, FG, MUTED, ACCENT = '#0f1216', '#e6eaf0', '#a7b1c0', '#4fb3c4'

def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf', size)

def text(draw, pos, value, size=28, color=FG, bold=False):
    draw.text(pos, value, font=font(size, bold), fill=color)

def ease(x):
    x = max(0, min(1, x))
    return x*x*(3-2*x)

def build():
    OUT.mkdir(exist_ok=True)
    shots = [Image.open(CAPTURES/name).convert('RGB').crop((0, 0, 977, 687))
             for name in ('01-input.png', '02-loading.png', '03-result.png', '04-json.png')]
    stages = [
        (4, 11, 0, '01 / Контекст', 'Правила компании', ['Обращение на русском', 'Условия переданы в контексте', 'Факт подтверждён системой']),
        (11, 16, 1, '02 / Обработка', 'Локальная LoRA', ['Qwen3.5 · 4B', 'Без исполнения операций', 'Ожидание в монтаже сокращено']),
        (16, 25, 2, '03 / Решение', 'Тон ≠ срочность', ['Клиент раздражён', 'Приоритет остаётся low', 'Предложить скачать счёт']),
        (25, 31, 3, '04 / Результат', 'Готовый JSON', ['Девять согласованных полей', 'Ссылки на правила', 'Копирование для оператора']),
    ]
    writer = imageio_ffmpeg.write_frames(str(OUT/'demo.mp4'), (W, H), fps=FPS,
        codec='libx264', pix_fmt_out='yuv420p', quality=8, macro_block_size=2,
        output_params=['-preset', 'fast', '-threads', '2', '-movflags', '+faststart'])
    writer.send(None)
    for frame in range(SECONDS*FPS):
        t = frame/FPS
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        text(d, (60, 34), '{ }  SUPPORT-JSON', 25, ACCENT, True)
        text(d, (1190, 37), 'Локальный AI-помощник', 22, MUTED)
        d.line((60, 86, 1540, 86), fill='#29333f', width=1)
        if t < 4 or t >= 31:
            local = t if t < 4 else t-31
            offset = int(30*(1-ease(local/0.8)))
            title = 'Поддержка по вашим правилам' if t < 4 else 'Решение. Черновик. Проверка.'
            text(d, (80, 205+offset), title, 67, FG, True)
            text(d, (84, 310+offset), 'Обращение + правила + факты → структурированный ответ', 31, MUTED)
            text(d, (84, 363+offset), 'На русском. Локально. С проверкой человеком.', 31, MUTED)
            labels = ['Контекст компании', 'Qwen3.5-4B + LoRA', 'JSON и черновик']
            for n, label in enumerate(labels):
                x = 84+n*490
                d.rounded_rectangle((x, 480+offset, x+430, 605+offset), radius=12, fill='#151a20', outline='#29333f', width=2)
                text(d, (x+24, 507+offset), f'0{n+1}', 20, ACCENT, True)
                text(d, (x+24, 540+offset), label, 27, FG, True)
            note = 'Настоящий прогон приложения · анимированный монтаж' if t < 4 else 'Модель, датасет и исходники доступны на Hugging Face'
            text(d, (84, 665+offset), note, 25, ACCENT)
        else:
            start, end, index, label, title, lines = next(s for s in stages if s[0] <= t < s[1])
            progress = (t-start)/(end-start)
            text(d, (60, 146), label, 22, ACCENT, True)
            text(d, (60, 205), title, 39, FG, True)
            for n, line in enumerate(lines):
                text(d, (62, 295+n*53), line, 23, MUTED)
            # Crop only native screenshot letterboxing; keep the actual UI intact.
            sw = int(1110*(1+0.012*ease(progress)))
            sh = round(sw*687/977)
            screen = shots[index].resize((sw, sh), Image.Resampling.LANCZOS)
            panel = Image.new('RGB', (1110, 780), BG)
            panel.paste(screen, ((1110-sw)//2, (780-sh)//2))
            im.paste(panel, (430, 110))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((427, 107, 1543, 893), radius=12, outline='#29333f', width=2)
            alpha = ease((t-start)/0.5)
            if alpha < 1:
                im = Image.blend(Image.new('RGB', (W, H), BG), im, alpha)
                d = ImageDraw.Draw(im)
        d.rectangle((0, H-5, int(W*t/SECONDS), H), fill=ACCENT)
        writer.send(im.tobytes())
    writer.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-threads', '2', '-i', str(OUT/'demo.mp4'),
        '-vf', 'fps=5,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer',
        '-filter_complex_threads', '1', '-loop', '0', str(OUT/'demo-preview.gif')], check=True, capture_output=True)
    qa = ROOT/'.build/video/qa'
    qa.mkdir(parents=True, exist_ok=True)
    for second in (2, 7, 20, 28, 33):
        subprocess.run([ffmpeg, '-y', '-ss', str(second), '-i', str(OUT/'demo.mp4'),
            '-frames:v', '1', str(qa/f'{second:02d}.png')], check=True, capture_output=True)
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(OUT/'demo.mp4'), '-f', 'null', '-'], capture_output=True, check=True)
    report = {'duration_seconds': SECONDS, 'fps': FPS, 'dimensions': [W,H], 'silent': True,
              'actual_application_screens': True, 'waiting_shortened': True,
              'source_crop': [0,0,977,687], 'full_decode_verified': not decoded.stderr,
              'files': {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in [OUT/'demo.mp4', OUT/'demo-preview.gif', *CAPTURES.glob('*.png')]}}
    (ROOT/'reports/demo-video.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    build()
