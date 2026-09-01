"""Theo doi chay: wandb va/hoac TensorBoard. Ca hai deu TUY CHON va CHIU LOI.

Nguyen tac: `log.txt` van la nguon su that duy nhat. `computer_uar_war` o cuoi
main.py ghi UAR/WAR vao do, va do la con so di vao bai. Hai cong cu duoi chi la
CUA SO NHIN VAO -- de theo doi hai may tu xa, khong phai de luu ket qua.

Vi vay moi loi o day deu bi nuot: mat mang, het quota, chua dang nhap, chua cai
goi -- khong duoc phep lam do mot run da chay 8 tieng. Doi lai, khong bao gio
duoc dung chung lam noi luu ket qua duy nhat.

Dung:
    tr = Tracker(args, fold=3, log_dir='log/DFEW-...-set3-log',
                 sha='a1b2c3d', gpu='NVIDIA_GeForce_RTX_4090')
    tr.epoch(0, train_acc=..., train_loss=..., val_acc=..., val_loss=..., lr=...)
    tr.final(uar=..., war=..., recall={...})
    tr.close()

Bat bang co dong lenh:
    --wandb --wandb-project defr-uot        # theo doi tu xa, gom ca hai may
    --tb                                    # TensorBoard cuc bo
"""
import os
import re


def _arm_of(args):
    """Ten nhanh ablation, suy tu co UOT. Dung de gom nhom tren dashboard."""
    if not getattr(args, 'use_uot', False):
        return 'baseline'
    if getattr(args, 'uot_mode', 'uot') == 'attn':
        return 'attn'
    tau = float(getattr(args, 'uot_tau', 1.0))
    return 'balanced' if tau >= 1e3 else 'uot'


class Tracker:
    def __init__(self, args, fold, log_dir, sha=None, gpu=None):
        self.args = args
        self.fold = fold
        self.arm = _arm_of(args)
        self.wb = None
        self.tb = None

        cfg = dict(vars(args))
        cfg.update(fold=fold, arm=self.arm, sha=sha, gpu=gpu)

        if getattr(args, 'wandb', False):
            try:
                import wandb
                self.wb = wandb.init(
                    project=getattr(args, 'wandb_project', 'defr-uot'),
                    name=f'{self.arm}_f{fold}',
                    # group theo FOLD, khong theo nhanh: bon nhanh cua mot fold la
                    # mot phep so ghep cap, dashboard nen bay chung canh nhau.
                    group=f'fold{fold}',
                    job_type=self.arm,
                    tags=[self.arm, f'fold{fold}', args.dataset,
                          f'img{args.img_size}'] + ([gpu] if gpu else []),
                    config=cfg,
                    reinit=True,
                )
                print(f'wandb: {self.wb.url}')
            except Exception as e:
                print(f'wandb tat (khong sao, log.txt van ghi du): {e}')
                self.wb = None

        if getattr(args, 'tb', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, 'tb'))
                self.tb.add_text('config', '\n'.join(f'{k}={v}' for k, v in cfg.items()))
                print(f'tensorboard: tensorboard --logdir {log_dir}/tb')
            except Exception as e:
                print(f'tensorboard tat: {e}')
                self.tb = None

    def epoch(self, ep, **kv):
        kv = {k: v for k, v in kv.items() if v is not None}
        if self.wb is not None:
            try:
                self.wb.log(kv, step=ep)
            except Exception:
                pass
        if self.tb is not None:
            try:
                for k, v in kv.items():
                    self.tb.add_scalar(k, float(v), ep)
            except Exception:
                pass

    def gates(self, ep, values):
        """Gate cua 12 khoi UOT, kem gradient dang nam tren chung.

        Gate khoi tao 0. Van ~0 sau 25 epoch nghia la model da hoc cach BO QUA
        nhanh UOT -- run do trung voi baseline. Hai chi so tach bach hai nguyen
        nhan, va chung doi hoi phan ung nguoc nhau:

          gate/abs_mean ~ 0  VA  gate/grad_abs_mean ~ 0
              -> dau ra nhanh UOT khong giup gi cho loss. Optimizer tu choi dung
                 no, va do la KET QUA, khong phai loi.

          gate/abs_mean ~ 0  NHUNG  gate/grad_abs_mean > 0
              -> co tin hieu ma tham so khong di theo. Van de toi uu; luc do
                 tang lr rieng cho gate hoac khoi tao khac 0 la dang thu, vi ket
                 qua am tinh se la nguy tao.
        """
        if not values:
            return
        gia_tri = {k: v for k, v in values.items() if not k.startswith('grad_')}
        grad = {k[5:]: v for k, v in values.items() if k.startswith('grad_')}

        flat = {f'gate/{k}': v for k, v in gia_tri.items()}
        flat.update({f'gate_grad/{k}': v for k, v in grad.items()})
        if gia_tri:
            flat['gate/abs_mean'] = sum(abs(v) for v in gia_tri.values()) / len(gia_tri)
            flat['gate/abs_max'] = max(abs(v) for v in gia_tri.values())
        if grad:
            flat['gate/grad_abs_mean'] = sum(abs(v) for v in grad.values()) / len(grad)
        self.epoch(ep, **flat)

    def final(self, uar, war, recall=None):
        kv = {'final/uar': uar, 'final/war': war}
        if recall:
            kv.update({f'final/recall_{k}': v for k, v in recall.items()})
        if self.wb is not None:
            try:
                self.wb.summary.update(kv)
                self.wb.log(kv)
            except Exception:
                pass
        if self.tb is not None:
            try:
                for k, v in kv.items():
                    self.tb.add_scalar(k, float(v), self.args.epochs)
            except Exception:
                pass

    def close(self):
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception:
                pass
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass


def git_sha():
    """SHA ngan, de moi run truy nguoc duoc ve dung ban ma. Tra None neu that bai."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def gpu_name():
    try:
        import torch
        return re.sub(r'\s+', '_', torch.cuda.get_device_name(0))
    except Exception:
        return None
