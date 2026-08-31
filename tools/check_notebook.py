"""Dry-run a Kaggle notebook: catch ordering and naming errors before a GPU session.

Shell escapes and %cd become no-ops while the Python runs for real, in one shared
namespace, cell by cell. That is what surfaces a variable used before the cell that
defines it, or a name left behind by a rename -- the failures that otherwise appear
only after uploading, and cost a session each.

Paths under /kaggle are remapped to a scratch tree so the cells can touch the
filesystem safely.

  python tools/check_notebook.py kaggle/AB_TRAINING.ipynb mafw7
"""
import json, re, sys

def to_python(src):
    out, buf = [], []
    for line in src.split("\n"):
        s = line.strip()
        if buf:                                  # dang noi tiep lenh shell
            buf.append(s.rstrip("\\").strip())
            if not s.endswith("\\"):
                out.append("_sh(f%r)" % " ".join(buf)); buf = []
            continue
        if s.startswith("!"):
            body = s[1:].strip()
            if body.endswith("\\"):
                buf = [body.rstrip("\\").strip()]
            else:
                out.append(" " * (len(line) - len(line.lstrip())) + "_sh(f%r)" % body)
        elif s.startswith("%cd"):
            out.append("_cd(f%r)" % s[3:].strip())
        elif s.startswith("%"):
            out.append("pass")
        else:
            out.append(line)
    return "\n".join(out)

import os
ROOT = os.environ.get("FAKE_KAGGLE", "/tmp/fake-kaggle")
for d in ("working", "temp", "input"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
nb = json.load(open(sys.argv[1]))
SETUP_OVERRIDE = sys.argv[2] if len(sys.argv) > 2 else None
cmds = []
ns = {"_sh": lambda c: cmds.append(c), "_cd": lambda c: cmds.append("cd " + c)}

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = to_python("\n".join(c["source"])).replace("/kaggle/", ROOT + "/")
    if SETUP_OVERRIDE and "SETUP = " in src:
        src = re.sub(r'SETUP\s*=\s*"[a-z0-9]+"', 'SETUP = "%s"' % SETUP_OVERRIDE, src, count=1)
    try:
        exec(compile(src, "cell%d" % i, "exec"), ns)
    except Exception as e:
        print("  CELL %-3d %s: %s" % (i, type(e).__name__, e))
        print("     " + src.replace("\n", "\n     ")[:300])
        sys.exit(1)

print("  tat ca cell chay qua. Lenh shell sinh ra:")
for c in cmds:
    print("   $", c)
