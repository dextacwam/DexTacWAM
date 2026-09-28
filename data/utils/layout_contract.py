"""Cross-repo validation of the arm/hand index layouts.

Three codebases slice the same flat action/state vectors: DexVTAM (training),
DexVTAM_deploy (policy server) and vtam_data_scripts (dataset conversion). They
cannot import each other -- deploy is a separate checkout on the robot host, and
vtam_data_scripts is a different repository -- so a layout edit in one place
would otherwise sit undetected until a rollout mis-slices an observation.

``layout_contracts/arm_layouts.json`` is the single declaration; every consumer
asserts its own layout objects against it and records ``contract_version``.
Resolution order for the file:

  1. ``$DEXVTAM_LAYOUT_CONTRACT`` -- point an isolated checkout at the canonical
     file (or a pinned copy whose hash is recorded on both sides).
  2. the copy shipped next to this module.

The contract deliberately mirrors EVERY field of ``RelativeArmLayout`` rather
than just the three widths: matching widths with a shifted internal block is
exactly the failure that a width-only check would wave through.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from typing import Any, Dict, Optional

CONTRACT_ENV_VAR = "DEXVTAM_LAYOUT_CONTRACT"
_DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "layout_contracts", "arm_layouts.json"
)


def resolve_contract_path(path: Optional[str] = None) -> str:
    """Explicit arg > env var > in-repo copy. Missing file is a hard error."""
    resolved = path or os.environ.get(CONTRACT_ENV_VAR) or _DEFAULT_CONTRACT_PATH
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"layout contract not found at {resolved!r}. Set ${CONTRACT_ENV_VAR} to the "
            f"canonical layout_contracts/arm_layouts.json (or a pinned copy of it)."
        )
    return resolved


def load_contract(path: Optional[str] = None) -> Dict[str, Any]:
    resolved = resolve_contract_path(path)
    with open(resolved, "rb") as fh:
        raw = fh.read()
    contract = json.loads(raw)
    for key in ("contract_version", "layouts"):
        if key not in contract:
            raise ValueError(f"{resolved}: layout contract is missing {key!r}")
    # The hash lets an isolated deploy checkout record WHICH copy it validated
    # against, so a silently edited local copy is still attributable after the fact.
    contract["_path"] = resolved
    contract["_sha256"] = hashlib.sha256(raw).hexdigest()
    return contract


def _normalize(value: Any) -> Any:
    """Tuples/lists compare structurally; JSON has no tuple type."""
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def assert_layout_matches_contract(layout: Any, contract: Optional[Dict] = None) -> int:
    """Assert one layout object equals its contract entry field-for-field.

    Returns the ``contract_version`` so callers can record it in provenance.
    """
    contract = contract or load_contract()
    name = getattr(layout, "name", None)
    entry = contract["layouts"].get(name)
    if entry is None:
        raise AssertionError(
            f"layout {name!r} is absent from the contract at {contract.get('_path')} "
            f"(has {sorted(contract['layouts'])}). Add it there first -- a layout that "
            f"exists in only one repo is exactly the drift this guards against."
        )

    mismatches = []
    for f in fields(layout):
        if f.name not in entry:
            mismatches.append(f"  {f.name}: absent from contract")
            continue
        got = _normalize(getattr(layout, f.name))
        want = _normalize(entry[f.name])
        if got != want:
            mismatches.append(f"  {f.name}: local={got!r} contract={want!r}")
    extra = sorted(set(entry) - {f.name for f in fields(layout)})
    if extra:
        mismatches.append(f"  contract has fields this code does not know: {extra}")

    if mismatches:
        raise AssertionError(
            f"arm_layout {name!r} disagrees with the shared contract "
            f"({contract.get('_path')}, version {contract['contract_version']}):\n"
            + "\n".join(mismatches)
            + "\nOne of the repos changed a layout without updating the contract."
        )
    return int(contract["contract_version"])


def assert_all_layouts_match_contract(
    layouts: Dict[str, Any], contract: Optional[Dict] = None
) -> int:
    """Validate a whole registry and require the contract to declare no others.

    A contract-only layout means this checkout is behind, which for the deploy
    server is worth failing on rather than discovering mid-rollout.
    """
    contract = contract or load_contract()
    version = int(contract["contract_version"])
    for layout in layouts.values():
        assert_layout_matches_contract(layout, contract)
    missing_locally = sorted(set(contract["layouts"]) - set(layouts))
    if missing_locally:
        raise AssertionError(
            f"contract {contract.get('_path')} declares layouts this checkout lacks: "
            f"{missing_locally}. Update the local registry before relying on it."
        )
    return version
