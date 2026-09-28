#!/usr/bin/env python3
"""The server's tactile-health backstop: what it accepts, and what it refuses.

The client decides health and holds locally, so in a healthy rollout none of
these refusals ever fire. They exist for the case that matters: the two ends
disagreeing about the contract. A server that quietly accepted an unfiltered
client, or a stale copy of the filter, would give back exactly the train/serve
mismatch the whole design removes -- and it would look like a clean run.

Exercises the REAL ``_verify_client_health`` (bound to a stand-in engine, since
the method touches nothing but layout fields), with the heavy model imports
stubbed so this runs anywhere:

    python web_infer_scripts/test_server_health_contract.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def undefined_globals(module) -> set:
    """Global names the module's code loads but never defines.

    Deleting an import leaves its uses behind, and Python only notices on the
    line that runs -- ``STATUS_OK`` survived removal of the health filter and
    reached remote, because no unit test calls the response builder that used it.
    Walking every code object catches that at import time instead.
    """
    import builtins
    import dis

    defined = set(vars(module)) | set(dir(builtins))
    own_file = module.__file__
    missing, seen = set(), set()

    def walk(code):
        # vars() also holds imported functions and classes; their globals live in
        # THEIR module, so only this file's code objects are ours to judge.
        if id(code) in seen or code.co_filename != own_file:
            return
        seen.add(id(code))
        for ins in dis.get_instructions(code):
            if ins.opname == "LOAD_GLOBAL":
                name = ins.argval.lstrip("+ ")   # 3.11 tags NULL-pushing loads
                if name not in defined:
                    missing.add(name)
        for const in code.co_consts:
            if hasattr(const, "co_code"):
                walk(const)

    for obj in vars(module).values():
        code = getattr(obj, "__code__", None)
        if code is not None:
            walk(code)
        elif isinstance(obj, type):
            for attr in vars(obj).values():
                fn = getattr(attr, "__func__", attr)
                if hasattr(fn, "__code__"):
                    walk(fn.__code__)
    return missing


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    """Only the model stack is stubbed; layouts and tactile_health stay real."""
    torch = _stub("torch", no_grad=lambda: (lambda f: f), device=object,
                  uint8=np.uint8, float32=np.float32, Tensor=object)
    _stub("torch.nn")
    _stub("torch.nn.functional", interpolate=None)
    torch.nn = sys.modules["torch.nn"]
    _stub("torchvision")
    _stub("torchvision.transforms", Resize=object, Normalize=object)
    _stub("einops", rearrange=None)
    _stub("runner")
    _stub("runner.tactile_inferencer", TactileInferencer=object)
    _stub("utils")
    _stub("utils.data_utils", get_text_conditions=None,
          gen_noise_from_condition_frame_latent=None, randn_tensor=None,
          _normalize_latents=None)
    _stub("models")
    _stub("models.pipeline")
    _stub("models.pipeline.custom_pipeline", calculate_shift=None,
          retrieve_timesteps=None)


_install_stubs()

import web_infer_scripts.tactile_server_sharpa_dexmate as srv  # noqa: E402
from data.utils.tactile_health import (  # noqa: E402
    CONTRACT_VERSION,
    STATUS_DEGRADED_SAFE,
    STATUS_NOT_READY,
    STATUS_OK,
    STATUS_SENSOR_UNAVAILABLE,
    module_sha256,
)

SHA = module_sha256()


def engine(mode="client_30hz", arms=("right",)):
    """A stand-in carrying only the fields _verify_client_health reads."""
    return types.SimpleNamespace(
        tactile_health_mode=mode,
        arms=list(arms),
        task_id="bowl",
        _health_sha=SHA,
    )


def verify(eng, health, tac):
    return srv.TactileBimanualInference._verify_client_health(eng, health, tac)


def good_tactile(n_arms=1):
    return np.full((n_arms, 5, 4, 4), 42, np.uint8)


def block(**over):
    base = {"status": STATUS_OK, "contract_version": CONTRACT_VERSION,
            "sha256": SHA, "fill_age": [[0] * 5], "valid_streak": 3}
    base.update(over)
    return base


def expect_refusal(label, health, tac, needle=""):
    try:
        verify(engine(), health, tac)
    except srv.TactileHealthContractError as exc:
        hit = needle.lower() in str(exc).lower()
        print(f"  {'PASS' if hit else 'FAIL'}  {label}: {str(exc).splitlines()[0][:88]}")
        return hit
    print(f"  FAIL  {label}: NOT REFUSED")
    return False


def expect_accept(label, health, tac, mode="client_30hz"):
    try:
        blank = verify(engine(mode=mode), health, tac)
    except Exception as exc:
        print(f"  FAIL  {label}: refused with {type(exc).__name__}: {exc}")
        return False
    print(f"  PASS  {label} (blank mask {blank.sum()} fingers)")
    return True


def main() -> int:
    ok = True
    print("accepted:")
    ok &= expect_accept("healthy ok", block(), good_tactile())
    ok &= expect_accept("in-window fill (degraded_safe still executes)",
                        block(status=STATUS_DEGRADED_SAFE, fill_age=[[3, 0, 0, 0, 0]]),
                        good_tactile())
    ok &= expect_accept("--task none: no health block required, blanks tolerated",
                        None, np.zeros((1, 5, 4, 4), np.uint8), mode="disabled")

    print("refused:")
    ok &= expect_refusal("no health block at all (unfiltered client)",
                         None, good_tactile(), "no 'tactile_health'")
    ok &= expect_refusal("contract version skew",
                         block(contract_version=CONTRACT_VERSION - 1),
                         good_tactile(), "contract mismatch")
    ok &= expect_refusal("vendored copy edited (sha differs)",
                         block(sha256="0" * 64), good_tactile(), "contract mismatch")
    ok &= expect_refusal("missing sha", block(sha256=None), good_tactile(),
                         "contract mismatch")
    for status in (STATUS_NOT_READY, STATUS_SENSOR_UNAVAILABLE, "fault_latched",
                   "ok_probably", ""):
        ok &= expect_refusal(f"client sent an unusable status {status!r}",
                             block(status=status), good_tactile(),
                             "had already judged")
    blank_one = good_tactile()
    blank_one[0, 2] = 0
    ok &= expect_refusal("claims ok but a finger arrived all-zero",
                         block(), blank_one, "arrived all-zero")
    ok &= expect_refusal("claims a fill but sent the blank frame anyway",
                         block(status=STATUS_DEGRADED_SAFE), blank_one,
                         "arrived all-zero")

    # The named finger must be the one that is actually blank: a right-only
    # server reporting "left2" would send someone hunting the wrong hand.
    print("diagnostics:")
    try:
        verify(engine(), block(), blank_one)
    except srv.TactileHealthContractError as exc:
        named = "right2" in str(exc)
        print(f"  {'PASS' if named else 'FAIL'}  blames the finger that is "
              f"actually blank (right2)")
        ok &= named

    # A bimanual server must name the correct hand for the same finger index.
    tac2 = good_tactile(2)
    tac2[0, 1] = 0                                  # LEFT index
    try:
        verify(engine(arms=("left", "right")), block(fill_age=[[0] * 5] * 2), tac2)
        print("  FAIL  bimanual blank not refused")
        ok = False
    except srv.TactileHealthContractError as exc:
        named = "left1" in str(exc) and "right" not in str(exc).split("fingers")[1][:20]
        print(f"  {'PASS' if named else 'FAIL'}  bimanual: names left1, not a "
              f"right finger")
        ok &= named

    print("statelessness:")
    src = Path(srv.__file__).read_text()
    body = src[src.index("class TactileBimanualInference"):]
    instantiates = "OnlineTactileHealthFilter(" in body
    print(f"  {'PASS' if not instantiates else 'FAIL'}  the server never "
          f"constructs a filter")
    ok &= not instantiates

    undefined = undefined_globals(srv)
    print(f"  {'PASS' if not undefined else 'FAIL'}  every global the server "
          f"references is defined"
          + (f" -- missing {sorted(undefined)}" if undefined else ""))
    ok &= not undefined

    print(f"\nSERVER HEALTH CONTRACT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
