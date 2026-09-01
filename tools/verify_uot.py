"""Check the UOT integration from first principles. No GPU, no checkpoints, ~5 seconds.

Each test states what it asserts and why that property has to hold, so a failure
points at the thing that broke rather than just at a line number. Run it after
touching models/uot.py or the fusion hook in Generate_Model.

  python tools/verify_uot.py
"""

import math
import os
import re
import sys
import textwrap

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.uot import UOTFusion, unbalanced_sinkhorn_log

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print("  [{}] {:<52} {}".format("PASS" if ok else "FAIL", name, detail))


print("=" * 78)
print("A. SOLVER -- balanced OT phai la truong hop rieng cua UOT")
print("=" * 78)

C = torch.rand(1, 16, 32)
C[:, :, :10] += 1.5                       # 10 cot co tinh lam nhieu, xa moi hang

pi_bal = unbalanced_sinkhorn_log(C, eps=0.05, tau=1e12, n_iters=400)
check("tau->inf: tong khoi luong = 1",
      abs(float(pi_bal.sum()) - 1.0) < 1e-4, "= {:.6f}".format(float(pi_bal.sum())))
check("tau->inf: bien hang deu = 1/16",
      float((pi_bal.sum(2) - 1 / 16).abs().max()) < 1e-4,
      "lech max {:.2e}".format(float((pi_bal.sum(2) - 1 / 16).abs().max())))

pi_uot = unbalanced_sinkhorn_log(C, eps=0.05, tau=0.1, n_iters=400)
m_bal = float(pi_bal[:, :, :10].sum())
m_uot = float(pi_uot[:, :, :10].sum())
check("UOT bo duoc cot khong khop, balanced thi khong",
      m_uot < m_bal * 0.1, "balanced {:.4f} -> UOT {:.4f}".format(m_bal, m_uot))

pi_h = unbalanced_sinkhorn_log(C.half(), eps=0.05, tau=1.0, n_iters=10)
check("fp16 khong sinh NaN/Inf", bool(torch.isfinite(pi_h).all()))

print()
print("=" * 78)
print("B. UOTFusion -- khop dung call-site trong Generate_Model")
print("=" * 78)

torch.manual_seed(0)
n, t, n_image, D = 2, 16, 196, 128
img = torch.randn(n * t, n_image + 7, D)          # 1 CLS + 196 patch + 6 prompt
aud = torch.randn(n, 32 * 8 + 7, D)               # 1 CLS + 256 patch + 6 prompt
m = UOTFusion(dim=D, n_frames=t, n_image=n_image, n_audio_t=32, n_audio_f=8)

a2v, v2a = m(img, aud, n, t)
check("shape a2v/v2a", a2v.shape == (n * t, 1, D) and v2a.shape == (n, 1, D),
      "{} / {}".format(tuple(a2v.shape), tuple(v2a.shape)))
check("cong vao 2 luong khong doi shape",
      (img + a2v).shape == img.shape and (aud + v2a).shape == aud.shape)
check("gate zero-init -> output bang DUNG 0",
      float(a2v.abs().max()) == 0.0 and float(v2a.abs().max()) == 0.0)

v = m._video_tokens(img, n, t)
order_ok = all(torch.equal(v[s, f], img[s * t + f, 0]) for s in range(n) for f in range(t))
check("thu tu batch sample*t+frame duoc giu", order_ok)

print()
print("=" * 78)
print("C. state_dict -- bat UOT khong duoc doi kien truc baseline")
print("=" * 78)


class Mock(nn.Module):
    def __init__(self, use_uot):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.our_classifier = nn.Linear(8, 7)
        self.use_uot = use_uot
        if use_uot:
            self.uot_fusion = nn.ModuleList([UOTFusion(dim=D) for _ in range(2)])


base_sd, uot_sd = Mock(False).state_dict(), Mock(True).state_dict()
extra = set(uot_sd) - set(base_sd)
check("'use_uot' khong lot vao state_dict", 'use_uot' not in base_sd and 'use_uot' not in uot_sd)
check("khong mat key nao khi bat UOT", not (set(base_sd) - set(uot_sd)))
check("key them vao deu la uot_fusion.*",
      bool(extra) and all(k.startswith('uot_fusion') for k in extra),
      "{} key".format(len(extra)))

msg = Mock(True).load_state_dict(base_sd, strict=False)
check("nap checkpoint baseline vao model UOT: unexpected = 0",
      len(msg.unexpected_keys) == 0)
check("... va missing chi gom uot_fusion.*",
      all('uot_fusion' in k for k in msg.missing_keys), "{} key".format(len(msg.missing_keys)))

print()
print("=" * 78)
print("D. Gradient -- UOT phai song sau khi all_gate lech khoi 0")
print("=" * 78)

torch.manual_seed(0)
uot = UOTFusion(dim=D, n_frames=t, n_image=n_image)
post, head = nn.Linear(D, 768), nn.Linear(768, 7)
all_gate = nn.Parameter(torch.zeros(1))          # gate zero-init cua CHINH MMA-DFER
opt = torch.optim.AdamW(list(uot.parameters()) + list(post.parameters())
                        + [all_gate] + list(head.parameters()), lr=1e-4)
y = torch.randint(0, 7, (n * t,))
grads = []
for _ in range(4):
    a2v_, _ = uot(img, aud, n, t)
    out = head(torch.tanh(all_gate) * post(img + a2v_)).mean(1)
    opt.zero_grad()
    nn.functional.cross_entropy(out, y).backward()
    grads.append(float(uot.gate_a2v.grad.abs()))
    opt.step()

check("step 0: gradient = 0 (bi all_gate=0 nhan bay -- DUNG nhu thiet ke)",
      grads[0] == 0.0)
check("step >=1: gate UOT nhan gradient", max(grads[1:]) > 0,
      "max {:.2e}".format(max(grads[1:])))

print()
print("=" * 78)
print("E. Tich hop -- day noi trong repo")
print("=" * 78)

gm = open(os.path.join(REPO, 'models', 'Generate_Model.py')).read()
mn = open(os.path.join(REPO, 'main.py')).read()

check("Generate_Model goi uot_fusion trong vong lap block",
      'self.uot_fusion[ii](image_lowdim_norm, audio_lowdim, n, t)' in gm)
check("fusion mean-pool GOC van con (UOT chi cong them)",
      'audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t,0)' in gm)
check("uot_fusion chi tao khi bat co", 'if self.use_uot:' in gm and 'ModuleList' in gm)
check("main.py mo requires_grad cho uot_fusion",
      'if "uot_fusion" in name:' in mn)
check("uot_fusion KHONG nam trong image_encoder (se bi dong bang)",
      'self.image_encoder.uot_fusion' not in gm)

print()
print("=" * 78)
n_fail = sum(1 for _, ok, _ in results if not ok)
print("{}/{} PASS".format(len(results) - n_fail, len(results)))
if n_fail:
    print("\nFAIL:")
    for name, ok, _ in results:
        if not ok:
            print("  -", name)
print("""
Nhung dieu script nay KHONG kiem tra duoc:
  - Cost matrix co Y NGHIA khong. No chi kiem tra so hoc chay dung. Hai phep chieu
    temporal_pre/audio_proj_pre hoc doc lap nen 1-cosine giua chung co the la nhieu;
    nhom da do duoc 6.5%% so voi 6.0%% doan mo. Xem UOT_ARCHITECTURE.md muc 7.
  - UOT co giup tang accuracy khong. Chi co train A/B moi tra loi duoc.
  - Duong ong du lieu. Dung tools/check_data.py va tools/smoke_test.py (can GPU).
""")
sys.exit(1 if n_fail else 0)
