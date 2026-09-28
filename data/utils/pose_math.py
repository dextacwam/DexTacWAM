"""Shared, vectorized SE(3) + rot6d pose-math helpers for the DexVTAM
``relative_eef_rot6d`` action mode.

Single source of truth so that the dataset (``dex_vtam_dataset.get_batch``),
the offline stats pass (``scripts/get_statistics.py``) and future deployment
all use *identical* math and rot6d conventions.

VERIFIED conventions (2026-07, pinned to the DexVTAM corpus + VTAM reference):
  * ``mat16`` flatten is ROW-MAJOR (numpy C-order): the arm pose stored in the
    150-D action / 90-D state is ``mat44.reshape(16)`` of a 4x4 homogeneous
    transform. Confirmed from ``vtam_data_scripts/convert_to_vtam.py``
    (``pose.reshape(n, 4*4)``) AND empirically from the erase stats block
    (bottom row ``[0,0,0,1]`` lands at dims 12-15 of each 16-block, std==0).
    Hence: ``mat44 = mat16.reshape(4, 4)``, ``t = mat44[:3, 3]``,
    ``R = mat44[:3, :3]``.
  * rot6d = the FIRST TWO COLUMNS of the rotation matrix, laid out as
    ``[col0(3), col1(3)]`` (i.e. ``swapaxes(R[..., :3, :2], -1, -2).reshape(6)``).
  * ``rot6d_to_mat`` rebuilds a proper rotation via Gram-Schmidt:
    ``b1 = norm(a1); b2 = norm(a2 - <b1,a2> b1); b3 = b1 x b2``,
    ``R = stack([b1, b2, b3], axis=-1)`` (columns).
  * relative transform direction: ``T_rel = inv(T_anchor) @ T_target``;
    deploy/compose inverse: ``T_abs = T_anchor @ T_rel``.

The rot6d and relativize/compose math is copied verbatim from
``VTAM/data/utils/pose_math.py``; only the pose container differs (DexVTAM
stores a flattened 4x4 ``mat16`` where VTAM stored ``pose6 = [xyz, rotvec]``).

All functions are batched over arbitrary leading dims unless noted.
"""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


# --------------------------------------------------------------------------- #
# rot6d <-> rotation matrix (first-two-columns + Gram-Schmidt)
# --------------------------------------------------------------------------- #
def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3, 3) proper rotation matrix (columns)."""
    d6 = np.asarray(d6, dtype=np.float32)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 6) rot6d = [col0(3), col1(3)]."""
    mat = np.asarray(mat, dtype=np.float32)
    return np.swapaxes(mat[..., :3, :2], -1, -2).reshape(mat.shape[:-2] + (6,))


# --------------------------------------------------------------------------- #
# mat16 (row-major flattened 4x4) <-> 4x4 homogeneous matrix  (batched)
# --------------------------------------------------------------------------- #
def mat16_to_mat44(mat16: np.ndarray) -> np.ndarray:
    """(..., 16) row-major flattened 4x4 -> (..., 4, 4) homogeneous transform."""
    mat16 = np.asarray(mat16, dtype=np.float32)
    return mat16.reshape(mat16.shape[:-1] + (4, 4))


def mat44_to_mat16(mat44: np.ndarray) -> np.ndarray:
    """(..., 4, 4) homogeneous transform -> (..., 16) row-major flattened 4x4."""
    mat44 = np.asarray(mat44, dtype=np.float32)
    return mat44.reshape(mat44.shape[:-2] + (16,))


def project_to_se3_mat16(mat16: np.ndarray) -> np.ndarray:
    """Project a (possibly noisy) mat16 onto the closest valid SE(3) mat16.

    Rotation is re-orthonormalized by round-tripping through rot6d (first two
    columns + Gram-Schmidt); translation is kept; bottom row forced to
    ``[0, 0, 0, 1]``. Used only by tests / diagnostics, never on training data.
    """
    T = mat16_to_mat44(mat16).copy()
    R = rot6d_to_mat(mat_to_rot6d(T[..., :3, :3]))
    T[..., :3, :3] = R
    T[..., 3, :3] = 0.0
    T[..., 3, 3] = 1.0
    return mat44_to_mat16(T)


# --------------------------------------------------------------------------- #
# relative action (SE(3) body-frame) <-> absolute pose, rot6d representation
# --------------------------------------------------------------------------- #
def relativize_mat16_to_rot6d(anchor16: np.ndarray, targets16: np.ndarray) -> np.ndarray:
    """Body-frame relative action, rot6d.

    anchor16:  (16,) or (T, 16) current EEF pose (row-major flattened 4x4).
    targets16: (T, 16) future absolute EEF poses (row-major flattened 4x4).
    returns:   (T, 9) = [rel_xyz(3), rel_rot6d(6)] with T_rel = inv(T_anchor) @ T_target.
    """
    targets16 = np.asarray(targets16, dtype=np.float32)
    anchor16 = np.asarray(anchor16, dtype=np.float32)
    if anchor16.ndim == 1:
        anchor16 = np.broadcast_to(anchor16, targets16.shape)
    T_anchor = mat16_to_mat44(anchor16)     # (T, 4, 4)
    T_targ = mat16_to_mat44(targets16)      # (T, 4, 4)
    T_rel = np.linalg.inv(T_anchor) @ T_targ
    rel_xyz = T_rel[..., :3, 3]                     # (T, 3)
    rel_rot6d = mat_to_rot6d(T_rel[..., :3, :3])    # (T, 6)
    return np.concatenate([rel_xyz, rel_rot6d], axis=-1).astype(np.float32)


def compose_mat16_and_relative_rot6d(anchor16: np.ndarray, rel9: np.ndarray) -> np.ndarray:
    """Deploy inverse: absolute EEF mat16 from anchor pose + relative rot6d action.

    anchor16: (16,) or (T, 16) current EEF pose (row-major flattened 4x4).
    rel9:     (T, 9) = [rel_xyz(3), rel_rot6d(6)].
    returns:  (T, 16) absolute EEF mat16, with T_abs = T_anchor @ T_rel.
    """
    rel9 = np.asarray(rel9, dtype=np.float32)
    anchor16 = np.asarray(anchor16, dtype=np.float32)
    lead = rel9.shape[:-1]
    if anchor16.ndim == 1:
        anchor16 = np.broadcast_to(anchor16, lead + (16,))
    T_rel = np.broadcast_to(np.eye(4, dtype=np.float32), lead + (4, 4)).copy()
    T_rel[..., :3, :3] = rot6d_to_mat(rel9[..., 3:9])
    T_rel[..., :3, 3] = rel9[..., :3]
    T_abs = mat16_to_mat44(anchor16) @ T_rel
    return mat44_to_mat16(T_abs)


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _random_se3_mat16(n: int, rng: np.random.Generator) -> np.ndarray:
    """n random valid SE(3) as row-major mat16, rotations via QR-orthonormalization."""
    A = rng.standard_normal((n, 3, 3))
    Q, Rr = np.linalg.qr(A)
    # make it a proper rotation (det=+1) and fix the QR sign ambiguity
    Q = Q * np.sign(np.diagonal(Rr, axis1=-2, axis2=-1))[..., None, :]
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1.0
    T = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
    T[:, :3, :3] = Q
    T[:, :3, 3] = rng.standard_normal((n, 3))
    return T.reshape(n, 16).astype(np.float32)


def _run_tests() -> None:
    rng = np.random.default_rng(0)
    atol = 1e-4

    # (a) rotation round-trip is exact for valid SO(3)
    T = _random_se3_mat16(16, rng)
    R = mat16_to_mat44(T)[:, :3, :3]
    R2 = rot6d_to_mat(mat_to_rot6d(R))
    assert np.allclose(R2, R, atol=atol), "rot6d_to_mat(mat_to_rot6d(R)) != R"

    # (b) output is a proper rotation for arbitrary 6D input
    x = rng.standard_normal((16, 6)).astype(np.float32)
    Rc = rot6d_to_mat(x)
    I = np.broadcast_to(np.eye(3, dtype=np.float32), Rc.shape)
    assert np.allclose(np.swapaxes(Rc, -1, -2) @ Rc, I, atol=atol), "R^T R != I"
    assert np.allclose(np.linalg.det(Rc), 1.0, atol=atol), "det(R) != 1"

    # (c) canonicalization idempotence (NOT mat_to_rot6d(rot6d_to_mat(x)) == x)
    x_canon = mat_to_rot6d(rot6d_to_mat(x))
    assert np.allclose(rot6d_to_mat(x_canon), rot6d_to_mat(x), atol=atol), \
        "rot6d not idempotent after canonicalization"

    # (d) mat16 <-> mat44 round-trip preserves layout exactly
    assert np.allclose(mat44_to_mat16(mat16_to_mat44(T)), T, atol=0.0), "mat16 round-trip failed"

    # (e) IDENTITY relative: anchor == target -> rel = [0,0,0, identity_rot6d]
    anchor = _random_se3_mat16(8, rng)
    rel_id = relativize_mat16_to_rot6d(anchor, anchor)
    assert np.allclose(rel_id[:, :3], 0.0, atol=atol), "identity rel translation != 0"
    id_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    assert np.allclose(rel_id[:, 3:9], id_rot6d, atol=atol), "identity rel rot6d != [1,0,0,0,1,0]"

    # (f) relativize -> compose recovers the absolute target (compare on the matrix)
    anchor1 = _random_se3_mat16(1, rng)
    targets = _random_se3_mat16(8, rng)
    rel9 = relativize_mat16_to_rot6d(anchor1[0], targets)
    rec16 = compose_mat16_and_relative_rot6d(anchor1[0], rel9)
    T_tgt = mat16_to_mat44(targets)
    T_rec = mat16_to_mat44(rec16)
    assert np.allclose(T_rec[:, :3, 3], T_tgt[:, :3, 3], atol=atol), "compose translation mismatch"
    assert np.allclose(T_rec[:, :3, :3], T_tgt[:, :3, :3], atol=atol), "compose rotation mismatch"
    assert np.allclose(T_rec[:, 3, :], np.array([0, 0, 0, 1], np.float32), atol=atol), \
        "compose bottom row != [0,0,0,1]"

    print("pose_math: synthetic self-tests passed (identity, random SE(3), round-trip)")


def _run_real_sample_test(stats_path: str) -> None:
    """Real-sample sanity using per-dim means from a corpus stats JSON.

    Not a substitute for a raw-row round-trip (means are not valid SE(3)), but a
    cheap machine-agnostic confirmation of the flatten offsets + homogeneous row.
    For the true raw-row round-trip run on a host where the raw capture share is mounted.
    """
    import json
    d = json.load(open(stats_path))
    key = next(k for k in d if k.endswith("_joint") and not k.endswith("delta_joint")
               and not k.endswith("state_joint"))
    mean = np.array(d[key]["mean"], dtype=np.float32)
    std = np.array(d[key]["std"], dtype=np.float32)
    for name, s in (("L", 118), ("R", 134)):
        m44 = mat16_to_mat44(mean[s:s + 16])
        sd44 = mat16_to_mat44(std[s:s + 16])
        assert np.allclose(m44[3], [0, 0, 0, 1], atol=1e-4), f"{name} bottom row != [0,0,0,1]"
        assert np.allclose(sd44[3], 0.0, atol=1e-4), f"{name} bottom row std != 0"
        print(f"  {name}_pose flatten OK: t~={np.round(m44[:3, 3], 3)}")
    print(f"pose_math: real-sample flatten check passed on {stats_path}")


if __name__ == "__main__":
    import sys
    _run_tests()
    if len(sys.argv) > 1:
        _run_real_sample_test(sys.argv[1])
