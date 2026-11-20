"""README 22: neuron statistics mixed from the chat and the general ones (as calib_pass.py's mixes), for
build_shape.py --from-stats.  python mix_stats.py --chat stats.npz --general calib_stats_general.npz --w 0.0625 --out x.npz"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--chat", required=True)
ap.add_argument("--general", required=True)
ap.add_argument("--w", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
c, g = np.load(a.chat), np.load(a.general)
mc, ic = c["mass"].astype(np.float64), c["imp"].astype(np.float64)
mg, ig = g["mass"].astype(np.float64), g["imp"].astype(np.float64)
m = (1 - a.w) * mc / mc.sum(1, keepdims=True) + a.w * mg / mg.sum(1, keepdims=True)
im = (1 - a.w) * ic / ic.sum((1, 2), keepdims=True) + a.w * ig / ig.sum((1, 2), keepdims=True)
np.savez(a.out, mass=m.astype(np.float32), imp=im.astype(np.float32))
print(f"wrote {a.out} (general weight {a.w})")
