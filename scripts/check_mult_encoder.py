"""Is our TransformerEncoder numerically identical to MMSA's?

We substituted nn.MultiheadAttention for MMSA's hand-rolled one and documented
it as "the maths is the same". This tests that claim instead of asserting it.
"""
import importlib.util
import sys
from pathlib import Path

import torch

MMSA = Path("/workspace/MMSA/src/MMSA/models/subNets/transformers_encoder")
sys.path.insert(0, str(MMSA.parent.parent.parent))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# MMSA's transformer.py imports from .position_embedding and .multihead_attention
pkg = "mmsa_enc"
sys.modules[pkg] = type(sys)(pkg)
sys.modules[pkg].__path__ = [str(MMSA)]
load(f"{pkg}.position_embedding", MMSA / "position_embedding.py")
load(f"{pkg}.multihead_attention", MMSA / "multihead_attention.py")
their_mod = load(f"{pkg}.transformer", MMSA / "transformer.py")

sys.path.insert(0, "/workspace/msa/src")
from msa.models.transformers import TransformerEncoder as Ours  # noqa: E402

torch.manual_seed(0)
D, H, L = 50, 10, 4
ours = Ours(D, H, L, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
            embed_dropout=0.0, attn_mask=True).eval()
theirs = their_mod.TransformerEncoder(
    embed_dim=D, num_heads=H, layers=L, attn_dropout=0.0, relu_dropout=0.0,
    res_dropout=0.0, embed_dropout=0.0, attn_mask=True).eval()

# Copy ours -> theirs so both hold identical weights.
for a, b in zip(ours.layers, theirs.layers):
    b.self_attn.in_proj_weight.data.copy_(a.attention.in_proj_weight.data)
    b.self_attn.in_proj_bias.data.copy_(a.attention.in_proj_bias.data)
    b.self_attn.out_proj.weight.data.copy_(a.attention.out_proj.weight.data)
    b.self_attn.out_proj.bias.data.copy_(a.attention.out_proj.bias.data)
    b.fc1.weight.data.copy_(a.fc1.weight.data)
    b.fc1.bias.data.copy_(a.fc1.bias.data)
    b.fc2.weight.data.copy_(a.fc2.weight.data)
    b.fc2.bias.data.copy_(a.fc2.bias.data)
    for i in range(2):
        b.layer_norms[i].weight.data.copy_(a.norms[i].weight.data)
        b.layer_norms[i].bias.data.copy_(a.norms[i].bias.data)
theirs.layer_norm.weight.data.copy_(ours.layer_norm.weight.data)
theirs.layer_norm.bias.data.copy_(ours.layer_norm.bias.data)

print("normalize flag on MMSA encoder:", getattr(theirs, "normalize", "<absent>"))
print("MMSA embed_positions:", theirs.embed_positions)

with torch.no_grad():
    # self-attention
    x = torch.randn(20, 4, D)
    d_self = (ours(x) - theirs(x)).abs().max().item()
    # cross-modal attention, different query/key lengths
    q, kv = torch.randn(20, 4, D), torch.randn(37, 4, D)
    d_cross = (ours(q, kv, kv) - theirs(q, kv, kv)).abs().max().item()

print(f"\nself-attention   max |diff| = {d_self:.3e}")
print(f"cross-modal      max |diff| = {d_cross:.3e}")
print("VERDICT:", "equivalent" if max(d_self, d_cross) < 1e-5 else "*** DIFFERENT ***")
