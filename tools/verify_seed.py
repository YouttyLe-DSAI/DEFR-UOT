"""Chung minh bon nhanh nhin thay CUNG mot luong du lieu.

Loi duoc va: `seed = 1` chi dat mot lan luc import main.py. Dung `--use-uot` thi
model dung them 12 UOTFusion x 2 nn.Linear(128,128) = 24 lan rut RNG toan cuc ma
baseline khong he rut. DataLoader(shuffle=True) sau do lay seed cua RandomSampler
VA seed goc cua worker tu chinh RNG toan cuc da bi lech -> thu tu batch, frame
duoc chon va augmentation deu khac nhau giua baseline va ba nhanh UOT, suot
25 epoch x 5 fold. Log van ghi seed=1. Khong mot canh bao nao.

Ba nhanh UOT khop nhau (dung module y het) nen chi dung NHANH LAM MOC bi lech --
tuc dung cai duy nhat khong duoc phep lech.

Ban va: set_seed(seed, fold, epoch) + generator rieng cho train_loader. Sau do
luong du lieu la ham thuan cua (seed, fold, epoch), doc lap voi so module da dung.

Script nay gia lap dung tinh huong do, KHONG can GPU hay timm.

Chay:  python3 tools/verify_seed.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_MAU = 500
BATCH = 8
ok = True


def check(ten, dat, chi_tiet=""):
    global ok
    ok &= bool(dat)
    print(f"  {'DAT ' if dat else 'HONG'}  {ten}{('  ' + chi_tiet) if chi_tiet else ''}")


def set_seed(s):
    """Ban sao cua main.set_seed. Giu dong bo neu ben kia doi."""
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def thu_tu_batch(n_rut_truoc, epoch_seed, dung_generator):
    """Tra ve thu tu mau cua mot epoch.

    n_rut_truoc gia lap so lan dung model rut tu RNG toan cuc:
      0  = nhanh baseline (khong dung UOTFusion)
      24 = ba nhanh UOT   (12 block x 2 Linear)
    """
    set_seed(epoch_seed)
    # dung model: rut RNG toan cuc n_rut_truoc lan
    for _ in range(n_rut_truoc):
        torch.nn.init.kaiming_uniform_(torch.empty(128, 128), a=5 ** 0.5)

    ds = list(range(N_MAU))
    if dung_generator:
        g = torch.Generator()
        g.manual_seed(epoch_seed)          # <- ban va: doc lap voi RNG toan cuc
        sampler = torch.utils.data.RandomSampler(ds, generator=g)
    else:
        sampler = torch.utils.data.RandomSampler(ds)   # <- ban cu: RNG toan cuc
    return list(sampler)[:BATCH * 4]


print("=" * 72)
print("1. BAN CU — khong co generator rieng (tai hien loi)")
print("=" * 72)
cu_base = thu_tu_batch(0, 12345, dung_generator=False)
cu_uot = thu_tu_batch(24, 12345, dung_generator=False)
print(f"  baseline : {cu_base[:8]}")
print(f"  UOT      : {cu_uot[:8]}")
check("ban cu THAT SU lech (neu day bao DAT nghia la loi co that)",
      cu_base != cu_uot,
      "hai nhanh nhin thay batch khac nhau")

print()
print("=" * 72)
print("2. BAN VA — generator rieng theo (seed, fold, epoch)")
print("=" * 72)
va_base = thu_tu_batch(0, 12345, dung_generator=True)
va_uot = thu_tu_batch(24, 12345, dung_generator=True)
print(f"  baseline : {va_base[:8]}")
print(f"  UOT      : {va_uot[:8]}")
check("bon nhanh nhin thay CUNG thu tu batch", va_base == va_uot)

print()
print("  --- doc lap voi so lan rut truoc do ---")
moi = {n: thu_tu_batch(n, 12345, dung_generator=True) for n in (0, 1, 24, 100, 1000)}
check("thu tu batch khong doi du rut truoc 0/1/24/100/1000 lan",
      all(v == moi[0] for v in moi.values()))

print()
print("=" * 72)
print("3. EPOCH VA FOLD PHAI CHO THU TU KHAC NHAU")
print("=" * 72)
# Neu moi epoch cho cung thu tu thi shuffle mat tac dung -- tai lap qua tay.
e0 = thu_tu_batch(24, 1 * 100000 + 1 * 1000 + 0, dung_generator=True)
e1 = thu_tu_batch(24, 1 * 100000 + 1 * 1000 + 1, dung_generator=True)
f2 = thu_tu_batch(24, 1 * 100000 + 2 * 1000 + 0, dung_generator=True)
check("epoch 0 khac epoch 1", e0 != e1)
check("fold 1 khac fold 2", e0 != f2)

print()
print("=" * 72)
print("4. RESUME TAI LAP DUOC — epoch 14 chay tiep phai giong chay lien mach")
print("=" * 72)
# Chay lien mach: dung mot tien trinh, di qua epoch 0..13 roi toi 14.
set_seed(1 * 100 + 1)
for ep in range(14):
    thu_tu_batch(24, 1 * 100000 + 1 * 1000 + ep, dung_generator=True)
lien_mach = thu_tu_batch(24, 1 * 100000 + 1 * 1000 + 14, dung_generator=True)
# Tien trinh MOI sau khi resume: chua tung chay epoch nao truoc do.
set_seed(1 * 100 + 1)
sau_resume = thu_tu_batch(24, 1 * 100000 + 1 * 1000 + 14, dung_generator=True)
check("epoch 14 sau resume trung khit chay lien mach", lien_mach == sau_resume,
      "-> khong can luu trang thai RNG vao checkpoint")

print()
print("=" * 72)
print("DAT — luong du lieu la ham thuan cua (seed, fold, epoch)."
      if ok else "HONG — ban va RNG chua dung. Dung train.")
print("=" * 72)
sys.exit(0 if ok else 1)
