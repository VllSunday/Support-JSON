"""Check fused masked CE and hidden-state gradients for a frozen classifier."""
import json
from pathlib import Path
import torch
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

torch.manual_seed(42)
x = torch.randn(64, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(512, 256, device='cuda', dtype=torch.bfloat16)
targets = torch.randint(512, (64,), device='cuda')
targets[:48] = -100
reference = torch.nn.functional.cross_entropy((x @ w.t()).float(), targets)
reference.backward()
gradient = x.grad.clone()
x.grad = None
actual = LigerFusedLinearCrossEntropyLoss()(w, x, targets)
actual.backward()
torch.testing.assert_close(actual, reference, atol=.05, rtol=.01)
torch.testing.assert_close(x.grad, gradient, atol=.04, rtol=.08)
assert torch.isfinite(x.grad).all()
report = {'frozen_classifier': True, 'masked_prompt': True,
          'loss_matches_reference': True, 'gradient_matches_reference': True,
          'reference_loss': reference.item(), 'fused_loss': actual.item(),
          'max_gradient_absolute_error': (x.grad-gradient).abs().max().item()}
Path('reports/liger-preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
