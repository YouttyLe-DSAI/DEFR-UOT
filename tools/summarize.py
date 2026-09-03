"""Doc toan bo log cua mot loat ablation va in ra bang ket qua.

compare_runs.py so DUNG HAI run. Loat nay co 4 nhanh x 5 fold = 20 run, nen can
mot cai khac: gom theo nhanh, ghep cap theo fold, va kiem dinh.

Chay TREN SERVER roi dan ket qua ve -- nhanh hon nhieu so voi rsync ca thu muc:

    python3 tools/summarize.py                    # tu do tim trong ./log
    python3 tools/summarize.py --dataset MAFW
    python3 tools/summarize.py --log-root /duong/dan/khac

Ba thu can nhin, theo thu tu:

  1. NHANH BASELINE co gan con so paper khong. O 160 paper bao 66.61/77.15 cho
     DFEW va 44.19/57.90 cho MAFW. Lech nhieu la duong ong sai o dau do, va moi
     so con lai deu khong dang tin.

  2. GATE co roi khoi 0 khong. Van ~0 nghia la model da hoc cach bo qua nhanh
     UOT, va run do trung voi baseline. Cot |g| cho biet.

  3. CHENH LECH giua cac nhanh, ghep cap theo fold. Do lech giua cac fold toi
     12 diem UAR nen so trung binh tho khong noi len gi; phai ghep cap.
"""
import argparse
import glob
import os
import re
import sys

# Nguong t hai phia, df = n-1. Chi den n=5 vi giao thuc la 5 fold.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PAPER = {('DFEW', 160): (66.61, 77.15), ('DFEW', 224): (67.01, 77.51),
         ('MAFW', 160): (44.19, 57.90), ('MAFW', 224): (44.11, 58.52)}
NHANH = ['baseline', 'attn', 'balanced', 'uot']


def doc(path):
    txt = open(path, errors='replace').read()

    def m1(pat, ep=str):
        m = re.search(pat, txt, re.M)
        return ep(m.group(1)) if m else None

    return dict(
        uar=m1(r'^UAR: ([\d.]+)', float),
        war=m1(r'^WAR: ([\d.]+)', float),
        dataset=m1(r'^dataset=(\w+)'),
        img=m1(r'^img_size=(\d+)', int),
        use_uot=m1(r'^use_uot=(\w+)') == 'True',
        uot_mode=m1(r'^uot_mode=(\w+)'),
        uot_tau=m1(r'^uot_tau=([\deE.+-]+)', float),
        epochs=m1(r'^epochs=(\d+)', int),
        seed_ok=m1(r'^seed=(\d+)', int),
        gate=m1(r'gates\s+\|g\| mean ([\d.]+)', float),
        canh_bao=bool(re.search(r'CANH BAO epoch 5', txt)),
        n_epoch_chay=len(re.findall(r'An epoch time:', txt)),
    )


def ten_nhanh(r):
    if not r['use_uot']:
        return 'baseline'
    if r['uot_mode'] == 'attn':
        return 'attn'
    return 'balanced' if (r['uot_tau'] or 1.0) >= 1e3 else 'uot'


def tt(a, b):
    """t ghep cap. Tra (chenh lech trung binh, t, co y nghia)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (d[0] if d else 0.0), None, False
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1)
    if var <= 0:
        return mu, None, False
    t = mu / ((var / n) ** 0.5)
    return mu, t, abs(t) > T_CRIT.get(n - 1, 1.96)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-root', default='log')
    p.add_argument('--dataset', default=None, help='DFEW hoac MAFW; mac dinh: ca hai')
    a = p.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(a.log_root, '*-set*-log'))):
        f = os.path.join(d, 'log.txt')
        if not os.path.isfile(f):
            continue
        r = doc(f)
        if r['uar'] is None or (a.dataset and r['dataset'] != a.dataset):
            continue
        fold = int(re.search(r'-set(\d+)-log', d).group(1))
        runs.setdefault((r['dataset'], r['img']), {}).setdefault(ten_nhanh(r), {})[fold] = r

    if not runs:
        sys.exit('Khong tim thay run nao da xong trong ' + a.log_root)

    for (ds, img), theo_nhanh in sorted(runs.items()):
        print('=' * 76)
        print('{}  {}x{}'.format(ds, img, img))
        print('=' * 76)

        pap = PAPER.get((ds, img))
        print('{:<10}{:>26}{:>26}{:>9}'.format('nhanh', 'UAR (tb +- do lech)',
                                               'WAR', '|gate|'))
        for nh in NHANH:
            fo = theo_nhanh.get(nh)
            if not fo:
                continue
            u = [fo[k]['uar'] for k in sorted(fo)]
            w = [fo[k]['war'] for k in sorted(fo)]
            g = [fo[k]['gate'] for k in sorted(fo) if fo[k]['gate'] is not None]
            n = len(u)
            mu = sum(u) / n
            sd = (sum((x - mu) ** 2 for x in u) / (n - 1)) ** 0.5 if n > 1 else 0.0
            mw = sum(w) / n
            sw = (sum((x - mw) ** 2 for x in w) / (n - 1)) ** 0.5 if n > 1 else 0.0
            gs = '{:.4f}'.format(sum(g) / len(g)) if g else '-'
            print('{:<10}{:>13.2f} +-{:<5.2f} (n={}){:>13.2f} +-{:<5.2f}{:>13}'
                  .format(nh, mu, sd, n, mw, sw, gs))
            if nh == 'baseline' and pap:
                print('{:<10}{:>13.2f}      paper 160{:>13.2f}{:>19}'
                      .format('  lech', mu - pap[0], mw - pap[1], ''))

        # canh bao gate
        for nh in NHANH[1:]:
            fo = theo_nhanh.get(nh) or {}
            chet = [k for k in sorted(fo)
                    if fo[k]['gate'] is not None and fo[k]['gate'] < 1e-3]
            if chet:
                print('\n  !! {}: gate ~0 o fold {} -- nhanh UOT khong dong gop gi,'
                      ' run do trung baseline'.format(nh, chet))

        # run chua xong 25 epoch
        thieu = [(nh, k, fo[k]['n_epoch_chay'])
                 for nh, fo in theo_nhanh.items() for k in sorted(fo)
                 if fo[k]['n_epoch_chay'] and fo[k]['n_epoch_chay'] < (fo[k]['epochs'] or 25)]
        if thieu:
            print('\n  !! chua du epoch: ' + ', '.join(
                '{} fold{} ({} epoch)'.format(*x) for x in thieu))

        # so ghep cap
        base = theo_nhanh.get('baseline')
        if base:
            print('\n  {:<22}{:>10}{:>8}{:>8}   {}'
                  .format('so ghep cap theo fold', 'dUAR', 't', 'n', 'ket luan'))
            for goc, moi in [('baseline', 'attn'), ('attn', 'balanced'),
                             ('balanced', 'uot'), ('baseline', 'uot')]:
                A, B = theo_nhanh.get(goc), theo_nhanh.get(moi)
                if not A or not B:
                    continue
                chung = sorted(set(A) & set(B))
                if not chung:
                    continue
                d, t, co = tt([B[k]['uar'] for k in chung], [A[k]['uar'] for k in chung])
                print('  {:<22}{:>+10.2f}{:>8}{:>8}   {}'.format(
                    '{} - {}'.format(moi, goc), d,
                    '{:.2f}'.format(t) if t is not None else '-', len(chung),
                    'CO Y NGHIA' if co else '-'))
            print('\n  nguong hai phia df=4: 2.776   |   df=1: 12.706')
        print()

    print('Cau hoi theo thu tu: (1) baseline co gan paper khong,'
          ' (2) gate co roi 0 khong, (3) chenh lech co y nghia khong.')


if __name__ == '__main__':
    main()
