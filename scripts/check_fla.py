"""Check the optional Windows Triton fast path against the Torch reference."""
import inspect
import json
import time
from pathlib import Path

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

torch.manual_seed(42)
shape = (1, 128, 16, 128)
q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
g = -torch.rand(shape[:3], device='cuda', dtype=torch.float32)
beta = torch.rand(shape[:3], device='cuda', dtype=torch.bfloat16)
reference = inspect.unwrap(torch_chunk_gated_delta_rule)
with torch.no_grad():
    expected, _ = reference(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
started = time.perf_counter()
actual, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
compile_seconds = time.perf_counter()-started
torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.04)
actual.float().square().mean().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
started = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
report = {'forward_matches_torch_reference': True, 'finite_backward': True,
          'first_compile_seconds': compile_seconds,
          'warm_seconds_per_call': (time.perf_counter()-started)/10,
          'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
          'max_absolute_error': (actual-expected).abs().max().item()}
Path('reports/fla-preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
