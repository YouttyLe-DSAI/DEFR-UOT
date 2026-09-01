"""Check the attention control arm before spending GPU hours on it.

The four arms of the ablation are meant to differ in exactly one thing: how the
cosine cost matrix becomes routing weights.

    baseline      no fusion module at all
    --uot-mode attn                     row-wise softmax
    --uot-mode uot  --uot-tau 1e6       Sinkhorn scaling, marginals enforced
    --uot-mode uot  --uot-tau 1.0       Sinkhorn scaling, marginals relaxed

If anything else differs -- parameter count, gate initialisation, output shape --
the comparison is worthless and no amount of training will fix it. This script
checks that before the fact.

Run:  python3 tools/verify_attn.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.uot import UOTFusion, attention_plan, unbalanced_sinkhorn_log

N_FRAMES, N_AUDIO_T, DIM = 16, 32, 128
N_IMAGE = 100          # 160x160 -> (160/16)^2
ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"  {'DAT ' if passed else 'HONG'}  {name}{('  ' + detail) if detail else ''}")


def make(mode, tau=1.0):
    return UOTFusion(dim=DIM, n_frames=N_FRAMES, n_image=N_IMAGE,
                     n_audio_t=N_AUDIO_T, n_audio_f=8,
                     eps=0.05, tau=tau, n_iters=10, mode=mode)


torch.manual_seed(0)
n, t = 4, N_FRAMES
img = torch.randn(n * t, 1 + N_IMAGE + 6, DIM)
aud = torch.randn(n, 1 + N_AUDIO_T * 8 + 6, DIM)

print("=" * 70)
print("1. SO THAM SO — bon nhanh phai GIONG HET, khong phai 'tuong duong'")
print("=" * 70)
p_uot = sum(p.numel() for p in make("uot").parameters())
p_attn = sum(p.numel() for p in make("attn").parameters())
check("attn va uot cung so tham so", p_uot == p_attn, f"{p_uot:,} vs {p_attn:,}")
print(f"       (x12 block = {12 * p_uot:,} tham so them vao)")

print()
print("=" * 70)
print("2. GATE KHOI TAO 0 — moi nhanh phai tra ve dung 0 o buoc dau")
print("=" * 70)
# Neu khong dung 0 thi model co UOT KHONG con trung khit baseline o step 0,
# va toan bo lap luan "bat/tat mot co, khong co bien an" sup do.
for mode, tau in [("attn", 1.0), ("uot", 1e6), ("uot", 1.0)]:
    m = make(mode, tau)
    with torch.no_grad():
        a2v, v2a = m(img, aud, n, t)
    label = f"{mode}" + (f" tau={tau:g}" if mode == "uot" else "")
    check(f"{label:<16} du luong = 0", a2v.abs().max() == 0 and v2a.abs().max() == 0)

print()
print("=" * 70)
print("3. KHOI LUONG HANG — chi UOT moi co tin hieu khoi luong")
print("=" * 70)
# Day la co che ma 'unbalanced' ban. Neu no khong xuat hien trong so lieu thi
# chu 'unbalanced' trong ten de tai khong co can cu.
C = torch.rand(n, N_FRAMES, N_AUDIO_T)
C[:, :, 24:] += 2.0        # gia lap audio khong co doi ung hinh anh

plans = {
    "attn":        attention_plan(C, temp=0.05),
    "uot tau=1e6": unbalanced_sinkhorn_log(C, eps=0.05, tau=1e6, n_iters=10),
    "uot tau=1.0": unbalanced_sinkhorn_log(C, eps=0.05, tau=1.0, n_iters=10),
}
# PHAI kiem CA HAI truc. Ban dau script nay chi kiem row_mass va da bo lot mot
# loi that: attention_plan chuan hoa theo HANG, nen row_mass dung bang 1/N, nhung
# col_mass hoan toan tu do va trai rong 0.0000 - 4.8134 -- manh hon ca nhanh UOT
# no phai doi chung. Kiem mot nua co che thi cong kiem tra chi cho cam giac an toan.
print(f'  {"nhanh":<14}{"tong":>8}{"he so nhan HANG":>22}{"he so nhan COT":>22}')
spread_r, spread_c = {}, {}
for name, pi in plans.items():
    rm, cm = pi.sum(2) * N_FRAMES, pi.sum(1) * N_AUDIO_T
    spread_r[name] = (rm.max() - rm.min()).item()
    spread_c[name] = (cm.max() - cm.min()).item()
    print(f"  {name:<14}{pi.sum(dim=(1,2)).mean():>8.4f}"
          f"{rm.min():>10.4f} - {rm.max():<9.4f}{cm.min():>10.4f} - {cm.max():<9.4f}")
print()
# Ghi chu: bang tren la cua RIENG ke hoach van chuyen. Trong UOTFusion.forward,
# nhanh attn duoc BO QUA hai phep nhan khoi luong, nen bien do cot cua no khong
# di vao model. Cac check duoi kiem dung dieu do.
check("attn: he so nhan HANG vo hieu", spread_r["attn"] < 1e-6)
check("uot tau=1e6: he so nhan HANG gan vo hieu", spread_r["uot tau=1e6"] < 0.1)
check("uot tau=1.0: he so nhan HANG co bien thien", spread_r["uot tau=1.0"] > 0.1)
check("uot tau=1e6: he so nhan COT gan vo hieu", spread_c["uot tau=1e6"] < 0.1)
check("uot tau=1.0: he so nhan COT co bien thien", spread_c["uot tau=1.0"] > 0.1)

print()
print("  --- nhanh attn phai BO QUA ca hai phep nhan khoi luong ---")
# Dung ke hoach Y HET (cung module, cung trong so), dung lai duong di ben trong
# NHUNG khong nhan khoi luong, roi doi chieu. Trung khit = module da bo qua that.
# KHONG duoc kiem bang cach doi self.mode sang 'uot': lam vay doi ca THUAT TOAN
# lan phep nhan, chenh lech thu duoc khong tach duoc hai nguyen nhan.
m_attn = make("attn")
with torch.no_grad():
    torch.nn.init.constant_(m_attn.gate_a2v, 1.0)
    torch.nn.init.constant_(m_attn.gate_v2a, 1.0)
    a2v_mod, v2a_mod = m_attn(img, aud, n, t)

    v = m_attn._video_tokens(img, n, t)
    a = m_attn._audio_tokens(aud)
    v_n = F.normalize(m_attn.norm_v(v), dim=-1)
    a_n = F.normalize(m_attn.norm_a(a), dim=-1)
    pi_ = attention_plan(1.0 - torch.bmm(v_n, a_n.transpose(1, 2)), temp=m_attn.eps)
    rm_ = pi_.sum(dim=2, keepdim=True)
    cm_ = pi_.sum(dim=1, keepdim=True).transpose(1, 2)
    a2v_ref = torch.bmm(pi_ / rm_.clamp_min(1e-8), a)          # khong nhan rm_
    v2a_ref = torch.bmm(pi_.transpose(1, 2) / cm_.clamp_min(1e-8), v)   # khong nhan cm_
    a2v_ref = (torch.tanh(m_attn.gate_a2v) * m_attn.proj_a2v(a2v_ref)).reshape(n * t, 1, DIM)
    v2a_ref = (torch.tanh(m_attn.gate_v2a) * m_attn.proj_v2a(v2a_ref)).mean(dim=1, keepdim=True)

d_a = (a2v_mod - a2v_ref).abs().max().item()
d_v = (v2a_mod - v2a_ref).abs().max().item()
check("attn a2v trung khit ban khong nhan khoi luong", d_a < 1e-6, f"lech {d_a:.2e}")
check("attn v2a trung khit ban khong nhan khoi luong", d_v < 1e-6, f"lech {d_v:.2e}")

print()
print("=" * 70)
print("4. HINH DANG DAU RA — phai giong nhau de diem chen khong doi")
print("=" * 70)
shapes = {}
for mode, tau in [("attn", 1.0), ("uot", 1.0)]:
    m = make(mode, tau)
    with torch.no_grad():
        torch.nn.init.constant_(m.gate_a2v, 1.0)     # mo gate de nhin hinh dang that
        torch.nn.init.constant_(m.gate_v2a, 1.0)
        a2v, v2a = m(img, aud, n, t)
    shapes[mode] = (tuple(a2v.shape), tuple(v2a.shape))
    print(f"  {mode:<6} a2v {tuple(a2v.shape)}   v2a {tuple(v2a.shape)}")
check("hai nhanh cung hinh dang dau ra", shapes["attn"] == shapes["uot"])
check("a2v dung [n*t, 1, D]", shapes["uot"][0] == (n * t, 1, DIM))
check("v2a dung [n, 1, D]", shapes["uot"][1] == (n, 1, DIM))

print()
print("=" * 70)
print("DAT — bon nhanh chi khac nhau o cach bien chi phi thanh trong so."
      if ok else "HONG — dung train cho toi khi sua xong.")
print("=" * 70)
sys.exit(0 if ok else 1)
