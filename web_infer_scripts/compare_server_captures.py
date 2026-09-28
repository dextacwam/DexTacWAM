#!/usr/bin/env python3
"""Diff two capture_server_responses.py recordings, byte-for-byte.

The other half of the bimanual byte-parity regression. Server denoising was
confirmed deterministic across runs and connections (identical accuracy digits
on repeat F3 runs), so there is no reason to accept a tolerance here: the
refactored server must return the SAME BYTES as the pre-refactor one for the
same checkpoint and the same observations. Comparison is on .tobytes(), which
unlike np.array_equal also catches NaN-vs-NaN, -0.0 vs 0.0 and dtype drift.

Two failure classes are kept apart on purpose:
  * OBSERVATION MISMATCH -- the two runs were not fed the same input, so the
    comparison proves nothing either way. This is a hard error, not a diff.
  * RESPONSE MISMATCH    -- the real regression signal.

Ping is compared leniently: the refactored server intentionally ADDS layout keys
(arms, raw_obs_layout, contract sha...). Added keys are reported; a key present
in both whose value CHANGED is a failure, since that is the client-visible
contract.

Usage:
    python web_infer_scripts/compare_server_captures.py /tmp/cap_pre.pkl /tmp/cap_new.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _same_bytes(a, b) -> tuple[bool, str]:
    """Strict equality for whatever a server puts in a response dict."""
    if isinstance(a, np.ndarray) != isinstance(b, np.ndarray):
        return False, f"type {type(a).__name__} vs {type(b).__name__}"
    if not isinstance(a, np.ndarray):
        return (a == b), ("equal" if a == b else f"{a!r} vs {b!r}")
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    aa, bb = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if aa.tobytes() == bb.tobytes():
        return True, f"identical {a.shape} {a.dtype}"
    with np.errstate(invalid="ignore"):
        d = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
        n = int(np.count_nonzero(aa != bb))
        return False, (f"{n}/{a.size} elements differ, max |delta| = "
                       f"{np.nanmax(d):.3e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("before")
    p.add_argument("after")
    args = p.parse_args()

    A = pickle.loads(Path(args.before).read_bytes())
    B = pickle.loads(Path(args.after).read_bytes())
    print(f"[cmp] {A['label']}  vs  {B['label']}")
    print(f"[cmp] config  : {A['config']}")
    if A["config"] != B["config"]:
        print(f"[cmp]           {B['config']}   <-- DIFFERENT CONFIG")
    print(f"[cmp] episode : {A['episode']} / {A['n_chunks']} chunks, "
          f"layout={A['arm_layout']}, cams={A['cameras']}")

    # ---- 1. did both servers actually see the same input? -----------------
    if A["obs_sha256"] != B["obs_sha256"]:
        print("[cmp] FATAL: the two runs were not fed identical observations; "
              "a parity result would be meaningless.")
        for i, (x, y) in enumerate(zip(A["obs_sha256"], B["obs_sha256"])):
            if x != y:
                print(f"    chunk {i}: {x[:16]} vs {y[:16]}")
        sys.exit(2)
    print(f"[cmp] observations identical ({len(A['obs_sha256'])} chunks, "
          f"first sha {A['obs_sha256'][0][:16]})")

    if A["n_chunks"] != B["n_chunks"]:
        print(f"[cmp] FATAL: chunk count {A['n_chunks']} vs {B['n_chunks']}")
        sys.exit(2)

    ok = True

    # ---- 2. ping contract: additions fine, changes are not ----------------
    pa, pb = A["ping"], B["ping"]
    added = sorted(set(pb) - set(pa))
    removed = sorted(set(pa) - set(pb))
    if added:
        print(f"[cmp] ping ADDED (expected from the refactor): {added}")
    if removed:
        print(f"[cmp] ping REMOVED: {removed}   <-- breaks existing clients")
        ok = False
    for k in sorted(set(pa) & set(pb)):
        same, why = _same_bytes(pa[k], pb[k])
        if not same:
            print(f"[cmp] ping CHANGED {k}: {why}")
            ok = False

    # ---- 3. the actual responses ------------------------------------------
    for i, (ca, cb) in enumerate(zip(A["chunks"], B["chunks"])):
        ka, kb = set(ca), set(cb)
        # Same rule as ping: a new key is an extension of the contract (Phase 4b
        # added "status" to every response), a missing one breaks clients.
        if kb - ka:
            print(f"[cmp] chunk {i}: response ADDED {sorted(kb - ka)}")
        if ka - kb:
            print(f"[cmp] chunk {i}: response REMOVED {sorted(ka - kb)}   <-- "
                  f"breaks existing clients")
            ok = False
        for k in sorted(ka & kb):
            same, why = _same_bytes(ca[k], cb[k])
            flag = "OK  " if same else "FAIL"
            print(f"[cmp] chunk {i} {k:16s} {flag}  {why}")
            ok = ok and same

    print(f"\n[cmp] BYTE PARITY: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
