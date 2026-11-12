"""Readable depth: the verdict of one raw device run against exact answers, and the readable depth of a family.

Definitions (as used in the declared tests of TN-Holdings-LLC/psf-os-lab-notebook):
  READS   the mean squared error of the measured values (one per qubit, or one per circuit) against their exact
          values is at most half the mean square of the exact values: MSE <= mean(exact ** 2) / 2.
          A device that returned zeros everywhere would have MSE = mean(exact ** 2); READS asks for at least half of
          the signal's squared size to survive, with no mitigation applied to the device values.
  AGREES  counted per value: |device - exact| <= tol (0.05 for per-qubit <Z>).
  readable depth of a family (one problem, one device, one day; depths declared before the run): the largest
          declared depth that READS. The first declared depth that does not read is reported beside it, and a family
          where some depth reads beyond one that does not is marked "not monotone".

usage:
  python -m readable_depth.score card CARD.json          # {"z": [...], "exact": [...]} -> verdict
  python -m readable_depth.score depth ROWS.json         # [{"family": .., "depth": .., "verdict": ..}, ...]
"""
import json
import sys

import numpy as np

AGREE_TOL = 0.05


def verdict(z, exact, tol=AGREE_TOL):
    """Verdict of one unit: device values z against exact values (same order)."""
    z, e = np.asarray(z, float), np.asarray(exact, float)
    if z.shape != e.shape or z.ndim != 1 or len(z) == 0:
        raise ValueError("z and exact must be equal-length, non-empty 1-D lists")
    mse, msq = float(np.mean((z - e) ** 2)), float(np.mean(e ** 2))
    return dict(mse=mse, mean_square=msq, common=mse / msq if msq > 0 else None,
                reads="READS" if mse <= msq / 2 else "DOES NOT READ",
                agrees=int(np.sum(np.abs(z - e) <= tol)), values=len(z))


def readable_depth(rows):
    """Per family: the largest depth that READS, the first depth that does not, and whether the order is monotone.
    rows: iterable of dicts with family, depth (number) and verdict ("READS" / "DOES NOT READ")."""
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append((float(r["depth"]), r["verdict"] == "READS"))
    out = {}
    for f, pts in fam.items():
        pts.sort()
        reads = [d for d, ok in pts if ok]
        fails = [d for d, ok in pts if not ok]
        out[f] = dict(readable_depth=max(reads) if reads else None,
                      first_not_reading=min(fails) if fails else None,
                      monotone=not (reads and fails and max(reads) > min(fails)),
                      depths=[d for d, _ in pts])
    return out


def main(argv):
    if len(argv) != 3 or argv[1] not in ("card", "depth"):
        raise SystemExit(__doc__)
    data = json.load(open(argv[2]))
    res = verdict(data["z"], data["exact"]) if argv[1] == "card" else readable_depth(data)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
