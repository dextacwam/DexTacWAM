# `relative_eef_rot6d` deployment gates

Serving a relative-EEF policy through a client that always sends a full bimanual
observation. This file pins down exactly *what was run* for each gate, so that a
failure on the robot can be traced to a version rather than argued about.
Everything below is measured, not expected.

Two layouts are in scope. **Right-only** (`tong`, `bowl`) is the case the gates
were originally written for: the server trims the client's bimanual observation
down to the right arm. **Bimanual** (`unscrew_v2`) is the newer case, and it is
not simply "the old path": erase, the only bimanual checkpoint ever gated here,
runs 1 camera, while `unscrew_v2` runs 3.

Gate order, and what each one is allowed to conclude:

| Gate | Where | Proves | Status |
|---|---|---|---|
| F1 | remote | server boots on the real checkpoint, 4 guards pass | PASS (bowl + tong + unscrew_v2) |
| F2 | anywhere, no GPU | config/layout/stats/gather/compose are self-consistent | PASS (bowl + tong + unscrew_v2) |
| F3 | remote | live server returns correct absolute targets for real episodes | PASS (bowl + tong + unscrew_v2) |
| P  | remote | the layout refactor did not change **bimanual** serving | PASS (erase 1-camera at `ee2ef77`, stale); **PASS re-taken on unscrew_v2, 3 cameras, at `aa00202`** |
| F4 | robot box | client dry-run against a live server, incl. IK | pending |
| F4b| client 30 Hz + remote | tactile blank-fill, hold/recovery, prompt provenance | prompts, parity and wrapper replay PASS; control-loop injection outstanding |
| F5 | robot box | contact-rich rollout | blocked on F4b |

For bowl and tong, F1–F3 have all been re-run against the current code and pass.

`unscrew_v2` passes F1, F2, F3 and P on `step_30000`, and its registry prompt is
verified byte-for-byte against the corpus. Gate P was re-taken on unscrew itself
rather than inherited: the erase capture predates the registry, the health
verification and the removal of the response `status` key, *and* it was 1 camera,
so the 3-camera bimanual path had never been compared against anything.

Every gate that can be run away from the robot has now passed for all three
tasks. What remains for `unscrew_v2` is F4/F4b/F5, which need the hardware.

## Versions under test (2026-07-30)

| | |
|---|---|
| server repo | `DexVTAM` branch `eef_9d_relative_deploy` |
| server code | `de30873` (server itself unchanged since `d612151`) |
| client tooling | `3d8311d` (offline replay client only) |
| hardware repo | `dexvtam_hardware_infra` branch `feat/relative-eef-deploy` @ `ed61afc` |
| layout contract | v1, sha256 `c7837020…` |

Checkpoint (bowl B3 action, **not** the B2 world model it warmstarted from):

```
outputs/
  stage3_action_full_0729_bowl_right_only_eef_relative_wm_bypass_shared_rmsnorm_long50000/
  2026_07_29_22_53_51/step_10000
      config.json                          764 B
      diffusion_pytorch_model.safetensors  4225092906 B
      projector.pt                         137127 B
```

Config `configs/bowl_unstack/stage3_action_expert.yaml`,
stats `…/20260725_pinch_from_bowl_with_fingers_right_only_relative_stats.json`,
`action_chunk=54`, `n_previous=4`, cameras `['head_img', 'right_wrist_img']`,
`arm_layout=right_only`, `disable_projector=True`, `use_pose_injection=True`.

Data: `data/datasets_lerobot/20260725_pinch_from_bowl_with_fingers_right_only`,
episode 90 — held out (train 0–89, val 90–99), 237 frames.

## F1 — server boot

Re-run 2026-07-30 at `bfade41` for both tasks: bowl `step_10000` and tong
`step_20000` (`.../2026_07_30_18_44_52/`), GPU 3. The launcher now picks the
config variant whose pretrained directory exists on the host, so remote resolves
`.yaml` on its own.

`web_infer_scripts/run_server_tactile_relative.sh` with `TASK=bowl STEP=10000`.
All four guards passed before the model loaded: complete checkpoint directory,
checkpoint `config.json` vs served YAML (`action_in_channels`), v0d adapter
sha256 `031d5e05…` matching the snapshot training used, and the F2 dry-run. The
port only opens after the model is fully loaded, so the client's successful
connect is itself part of this gate.

`STEP` has no default on purpose. The step numbers appearing in the YAML belong
to the B2 world model that B3 warmstarted from; a stale default here would
silently serve the wrong weights, so the launcher instead lists the available
`step_*` directories and exits.

### `unscrew_v2` — 2026-08-01 at `aa00202`, PASS

`TASK=unscrew_v2 STEP=30000`, `.../2026_08_01_01_40_35/step_30000`. All four
guards passed and the port opened. The manifest is the gate's real output, since
this is the first bimanual checkpoint served through the layout contract:

```
[tactile_server] task          = unscrew_v2  prompt_sha256=843a6f8e19c0
[tactile_server] arm_layout    = bimanual arms=['left', 'right'] (contract v1 sha256=c7837020)
[tactile_server] raw_obs_layout= bimanual state 90 -> model 90  (identity)
[tactile_server] action key    = ..._relative_eef  (dim 136)
[tactile_server] state  key    = ..._state_eef  (dim 90)
[tactile_server] cameras       = ['head_img', 'left_wrist_img', 'right_wrist_img']
[tactile_server] n_view total  = 5 (visual=3, tactile=2)
[tactile_server] tactile hands = raw [0, 1] -> ['left', 'right']; blank windows all 5
[tactile_server] tactile health= client_30hz (contract v2 sha 71a91b79)
```

The blank-window line is the per-task policy fix arriving live: all ten fingers
at 5, which is what unscrew's corpus was converted with. Under the old module
constant seven of them would have read 10 here.

## F2 — offline dry-run

`web_infer_scripts/dryrun_relative_server.py`, numpy only, no GPU, no model.
Passed on four configs, chosen to cover both gather paths and both camera counts:

- bimanual erase (1 camera): identity gather, leakage check n/a, per-arm anchors
  at `[58:74]` and `[74:90]`, response `(54,2,4,4)` / `(54,44)`
- tong and bowl right-only (2 cameras): state gather 90→45, left-poison
  invariance, compose round-trip `2.4e-07`
- **bimanual `unscrew_v2` (3 cameras)**: identity gather 90→90, `n_view_total=5`
  against `max_view=5` and `projector.num_views=2`, stats `136 + 90 = 226`,
  compose round-trip `2.38e-07` on **both** arms, response `(54,2,4,4)` /
  `(54,44)`, tactile keeps both hands `(2,5,H,W)`

```bash
python3 web_infer_scripts/dryrun_relative_server.py \
  -c configs/bottle_cap/stage3_action_expert.yaml \
  --task unscrew_v2
```

Pass `--task`. Without it the run falls back to the legacy blank-window table
and labels itself as such; see the note below.

### Blank-window policy is per task, not a constant

Adding `unscrew_v2` surfaced a train/serve mismatch that F2 now covers. The
online blank-fill window has to be the window each finger's *training* data was
built with, but it was a module constant — `OPERATING_FINGERS_GLOBAL = (5,6,7)`,
strict 5 / lenient 10 — which is tong and bowl's `run_convert.sh` policy
(`--operating-fingers 5 6 7 --nonoperating-short-run-max 10`). `unscrew_v2`
passes **no** finger flags, so the converter's defaults applied and all ten
fingers were converted at 5.

Served under the constant, 7 of unscrew's 10 fingers would have been given a
window of 10: the error is in the lenient direction, so instead of holding the
loop the server would have carried a dropout forward for twice as many frames as
any training frame contained. The policy now lives on `TaskEntry` next to the
prompt, in the converter's own three variables so the two can be diffed:

| task | operating fingers | strict | non-operating | resulting windows (global 0–9) |
|---|---|---|---|---|
| tong, bowl | 5, 6, 7 | 5 | 10 | `10 10 10 10 10 5 5 5 10 10` (unchanged) |
| unscrew_v2 | all ten | 5 | 5 | `5 5 5 5 5 5 5 5 5 5` |
| `--task none` | legacy constant | 5 | 10 | as tong/bowl, so byte-parity captures stay comparable |

## F3 — live replay, bowl episode 90

`web_infer_scripts/offline_client_relative.py`. The right-only dataset has no
left arm in it at all, so the client re-embeds each row into the raw bimanual
wire format (state 45→90, tactile (1,5)→(2,5)) and **fills the left with NaN**.
The server gathers only the right blocks, so NaN cannot reach the model — and if
it ever did, every output would be NaN rather than subtly wrong.

```
chunk 0 right: compose vs arm_target_pose 0.00e+00 | rel xyz 1.19e-07 | rel rot6d 5.96e-08   [non-orthonormality 0.120]
chunk 1 right: compose vs arm_target_pose 0.00e+00 | rel xyz 8.20e-08 | rel rot6d 1.19e-07   [non-orthonormality 0.060]
chunk 2 right: compose vs arm_target_pose 0.00e+00 | rel xyz 8.94e-08 | rel rot6d 1.19e-07   [non-orthonormality 0.057]
GATE F3: PASS (shapes, no-leak, compose round-trip, SE(3) validity, hand passthrough)
```

Tong `step_20000`, episode 90 (829 frames), also PASS, with the same exact
compose and identical `hand_target` passthrough. Both runs now also print
`prompt byte-identical to the dataset`, so the registry path is confirmed
against a live server, not just offline.

The informational accuracy block is **not** part of the gate, but tong's is worth
recording because it is not just "early checkpoint" noise:

| | bowl `step_10000` | tong `step_20000` |
|---|---|---|
| EEF translation, mean / max | 28.7 / 93.5 mm | 46.7 / 171.3 mm |
| EEF rotation, mean / max | 7.0 / 18.2 deg | 5.6 / **112.9** deg |
| rot6d non-orthonormality, chunk 0 | 0.120 | **0.929** |
| open-loop `mse_state` (B3 eval) | 0.030 | **5.07** |

Non-orthonormality is how far the head's raw 6 numbers sit from a valid rotation
before `rot6d_to_mat` fixes them up. Bowl stays near 0.06–0.12 throughout; tong's
first chunk at 0.929 means that chunk's rotation output is essentially
unconstrained and only becomes a rotation because the server normalizes it. That
lines up with the 112.9 deg rotation outlier and with `mse_state` being 170x
bowl's while `mse_action` matches it — an action head that is fine next to a
state echo that is not. The asymmetry points at tong's state normalization
rather than at undertraining, which is the `std < 1e-6` failure class already hit
once when generating bowl's stats. F1/F2/F3 cannot see it: they check that the
server faithfully transports and composes whatever the model emits, not whether
what it emits is sane.

### `unscrew_v2` — 2026-08-01, PASS, and the first two-arm F3

`step_30000`, episode 90 (693 frames), 3 chunks. This is the first F3 where the
per-arm anchors are both exercised: bowl and tong only ever had a right arm, so
`[58:74]` vs `[74:90]` and the `action[118:127]` / `action[127:136]` split were
until now asserted by unit tests rather than demonstrated against a live server.

All six checks (3 chunks × 2 arms) compose to **exactly** `0.00e+00`, with `rel
xyz` and Gram-Schmidt-modulo `rot6d` residuals at `6e-08`–`1.2e-07`. The client
also confirmed `arm_layout=bimanual arms=['left','right']`, images
`(3,192,256,3)`, tactile `(2,5,192,256)` and the prompt byte-identical to the
dataset.

Accuracy (informational, and the reason it is worth recording is that unscrew
shows **none** of tong's pathology):

| | bowl `step_10000` | tong `step_20000` | unscrew_v2 `step_30000` |
|---|---|---|---|
| EEF translation, mean / max | 28.7 / 93.5 mm | 46.7 / 171.3 mm | 52.0 / 132.9 mm |
| EEF rotation, mean / max | 7.0 / 18.2 deg | 5.6 / **112.9** deg | 8.8 / 30.2 deg |
| rot6d non-orthonormality | 0.120 | **0.929** | 0.015–0.018 (L), 0.025–0.079 (R) |
| hand joint MAE | — | — | 0.0222 rad |

Unscrew's non-orthonormality is the lowest of the three and its rotation error
stays bounded, so the rotation head is emitting near-valid rotations rather than
relying on the server to normalize noise into one. Translation is the largest of
the three, but it is a different corpus and a harder task, and F3 is not an
accuracy gate. Nothing here resembles the state-normalization signature that
still blocks tong.

The compose identity is exactly zero, not merely within tolerance: the server
composed with the anchor from the observation we sent, read from the model-space
pose slice, held fixed across all 54 rows of the chunk.

Two results worth carrying forward:

**Rotation is only recoverable up to Gram-Schmidt.** The model emits six
unconstrained numbers per arm and `rot6d_to_mat` orthonormalizes them, so
relativizing the composed pose returns the *normalized* rot6d and can never
equal the raw predicted vector. An assertion demanding raw equality fails on
every checkpoint — it did here, at 0.06–0.12, before the check was corrected.
That residual is now reported as a checkpoint diagnostic; it should shrink as
B3 trains, and growth is the thing to watch.

**Denoising is deterministic across runs and connections.** Two separate runs
produced identical accuracy digits. This is what makes the byte-parity gate
below possible without tolerances.

Accuracy against ground-truth absolute targets, informational only — 10k steps:
28.67 mm mean / 15.10 mm median / 93.47 mm max, 6.99° mean / 5.82° median, hand
joint MAE 0.0303 rad. Far too coherent for a mis-read anchor, which would be off
by tens of centimetres.

## P — bimanual byte-parity

Taken twice. The erase run below is the original and is now stale — it predates
the registry, the health verification and the removal of the response `status`
key. It is kept because it is the only evidence for the 1-camera path. The
authoritative current result is the
[unscrew_v2 re-run](#p-re-run-for-unscrew_v2--pass), which also covers 3 cameras.

F1–F3 only exercise the right-only path. The refactor replaced *every* hardcoded
bimanual slice in the server with layout lookups, so erase/chip had to be shown
to serve identically to before. Denoising is deterministic, so this is an exact
comparison rather than a tolerance argument.

Held constant across the two runs: checkpoint `stage3_action_full_0718_erase_whiteboard_no_wrist_eef_relative_erasewm_bypass_shared_rmsnorm_long50000/2026_07_18_16_46_18/step_20000`,
GPU 3, port 5009, `--denoise-steps 10`, erase episode 90 (1361 frames), 3 chunks.
The `stat_file` in that config is a *relative* path, so each server resolves it
against its own repo root; both worktrees were checked to hold byte-identical
copies (config `58199f73…`, stats `d8dbfa6b…`) before serving.

| | |
|---|---|
| pre-refactor server | `7ec602d` in a detached worktree `DexVTAM_deploy_pre` |
| refactored server | `ee2ef77` |
| observations | identical, sha `d924fc3784f7` / `43350791fff9` / `2fcb386fda25` |
| responses | all 12 arrays byte-identical: `action (54,136)`, `arm_target_pose (54,2,4,4)`, `hand_target (54,44)` × 3 chunks |
| ping | additions only (`arms`, `raw_obs_layout`, contract sha, blank windows, …); nothing removed or changed |

The pre-refactor ping reports no `arm_layout` at all, which is how the capture
confirms it really reached the old server rather than a stale process on the
port.

```bash
# reproduce: same checkpoint, same GPU, same port, one server at a time
python web_infer_scripts/capture_server_responses.py -c <cfg> --data-root <ds> \
    --episode 90 --n-chunks 3 --port 5009 --label pre-refactor \
    --out /tmp/cap_erase_pre.pkl --shutdown-server
python web_infer_scripts/capture_server_responses.py -c <cfg> --data-root <ds> \
    --episode 90 --n-chunks 3 --port 5009 --label refactored \
    --out /tmp/cap_erase_new.pkl --shutdown-server
python web_infer_scripts/compare_server_captures.py /tmp/cap_erase_pre.pkl /tmp/cap_erase_new.pkl
```

Both servers must run on the **same GPU**: different GPU models can produce
different floating-point results for identical code, which would break parity
for reasons unrelated to the refactor. The capture records a sha256 of every
observation sent, so the comparator rejects a vacuous pass where the two runs
were not actually fed the same input. Ping keys added by the refactor are
reported, not failed; a key present in both whose value changed is a failure,
since that is the client-visible contract.

Not covered by that run: chip (`0718_pick_chip_with_wrist_eef_relative`) and
handover both have bimanual relative B3 checkpoints on the training host, but their configs
are not in this repo. Erase exercises the 1-camera bimanual path only.

### P re-run for `unscrew_v2` — PASS

Run 2026-08-01. `unscrew_v2` is the first 3-camera bimanual checkpoint we intend
to serve, and the erase PASS did not speak for it on two counts: it was taken at
`ee2ef77`, before the task registry, the health verification and the removal of
the response `status` key; and it was 1 camera, so the 3-camera gather and the
`n_view_total=5` projector path had never been compared against a reference.

| | |
|---|---|
| pre-refactor server | `7ec602d`, worktree `DexVTAM_deploy_pre` |
| refactored server | `aa00202`, worktree `DexVTAM_deploy`, `--task none` |
| checkpoint | `.../2026_08_01_01_40_35/step_30000`, same GPU, port 5009, `--denoise-steps 10`, `threshold 54` both sides |
| config / stats | sha256 `13fabd97…` / `ead163ed…`, byte-identical in both worktrees |
| episode | 90, 3 chunks, `layout=bimanual`, cams `['head_img','left_wrist_img','right_wrist_img']` |
| observations | identical, first sha `d93cd43c2872` |
| responses | all 12 arrays byte-identical: `action (54,136)`, `arm_target_pose (54,2,4,4)`, `hand_target (54,44)` × 3 chunks, plus `action_mode` equal |
| ping | 23 keys ADDED, none removed or changed |

The added ping keys are the whole refactor surface — `arm_layout`, `arms`,
`raw_obs_layout`, `camera_names`, the contract sha, the blank windows, the
tactile-health block and the task/prompt triple. That the pre-refactor server
reports none of them is also what proves the capture reached the old process
rather than a stale one on the port.

So the layout lookups that replaced every hardcoded bimanual slice are
transparent at 3 cameras as well as 1. The procedure that produced this:

Same shape as the erase run — two servers, one at a time, same GPU, same port,
same episode — with `7ec602d` as the pre-refactor reference. Note that `7ec602d`
predates the registry, so the reference server takes the prompt verbatim from
the observation; run the current server with `--task none` so both are serving
the same string and the comparison is about the layout refactor rather than the
prompt. `--task none` also selects the legacy blank-window table, which is what
the reference has.

`capture_server_responses.py` is a *client*: it connects to a server that is
already listening. So `--task none` goes on the **server** command, and the
parity runs bypass `run_server_tactile_relative.sh` (whose `TASK` must be a
registry id) in favour of invoking the server directly.

```bash
CFG=configs/bottle_cap/stage3_action_expert.yaml
W=outputs/stage3_action_full_0730_unscrew_bottle_cap_v2_with_wrist_eef_relative_wm_bypass_shared_rmsnorm_long50000/<TS>/step_<N>
DS=data/datasets_lerobot/20260724_unscrew_bottle_cap_v2

# 0. The reference worktree must hold the config AND the stats, because 7ec602d
#    predates both and `stat_file` is repo-relative -- each server resolves it
#    against its own root, so a missing copy is a boot failure and a DIFFERENT
#    copy is a silently wrong de-normalisation. Take them from the same commit
#    the new worktree is on rather than copying, and check the blobs match.
#    Do NOT move the reference worktree onto the branch: the point of it is the
#    pre-refactor CODE, and pulling would make the comparison vacuous.
D=configs/bottle_cap
git -C "$PRE" checkout "$(git -C "$NEW" rev-parse HEAD)" -- "$D"
git -C "$PRE" hash-object "$D"/*.yaml "$D"/*.json
git -C "$NEW" hash-object "$D"/*.yaml "$D"/*.json

# 1. pre-refactor reference, from a detached worktree at 7ec602d, GPU 3 port 5009
#    (that commit has no --task flag at all)
CUDA_VISIBLE_DEVICES=3 python web_infer_scripts/tactile_server_sharpa_dexmate.py \
    -c "$CFG" -w "$W" --domain-name 20260724_unscrew_bottle_cap_v2 \
    --denoise-steps 10 --device cuda:0 --port 5009 &
python web_infer_scripts/capture_server_responses.py -c "$CFG" --data-root "$DS" \
    --episode <n> --n-chunks 3 --port 5009 --label pre-refactor \
    --out /tmp/cap_unscrew_pre.pkl --shutdown-server

# 2. current server, SAME GPU and port, prompt taken from the observation
CUDA_VISIBLE_DEVICES=3 python web_infer_scripts/tactile_server_sharpa_dexmate.py \
    -c "$CFG" -w "$W" --domain-name 20260724_unscrew_bottle_cap_v2 --task none \
    --denoise-steps 10 --device cuda:0 --port 5009 &
python web_infer_scripts/capture_server_responses.py -c "$CFG" --data-root "$DS" \
    --episode <n> --n-chunks 3 --port 5009 --label refactored \
    --out /tmp/cap_unscrew_new.pkl --shutdown-server

python web_infer_scripts/compare_server_captures.py \
    /tmp/cap_unscrew_pre.pkl /tmp/cap_unscrew_new.pkl
```

Expect `action (54,136)`, `arm_target_pose (54,2,4,4)` and `hand_target (54,44)`
byte-identical across all 3 chunks, and ping differences to be additions only.

A pass here says the refactor is transparent for 3-camera bimanual. It says
nothing about whether `unscrew_v2`'s own weights are any good — that is F1, F3
and then F5. And because parity is run with `--task none`, the served prompt and
the per-task window table are **not** exercised by it; those are F1's registry
guard and F2's window table respectively.

## F4b — tactile health, now client-side at 30 Hz

Training data went through blank fill, so a policy fed live black frames has a
train/serve mismatch exactly when the sensor is failing — i.e. during contact.
The filter that closes that gap is `data/utils/tactile_health.py`
(`OnlineTactileHealthFilter`), and it reproduces the converter's decision
causally, per frame. Semantics are taken from `data_fixes.py`, not inferred:

- blank = `np.all(raw == 0)`, the same test as `detect_dropout`
- an **internal** run of length ≤ W is carry-forward filled; > W is a length the
  converter would have discarded the episode over (`classify_runs`:
  `short_fill if length <= srm else long_discard`)
- a **leading** run — before a finger's first good frame — is a different policy:
  offline it is *trimmed* up to `BOUNDARY_TRIM_MAX = 15`, so online it maps to
  `not_ready`, never to the 5/10 window

### Why the filter moved to the client

The previous revision ran this inside the server's `step()`. That was wrong by a
factor of 54. The client executes `num_action_execute = 54` rows at
`COMMAND_HZ = 30` before it sends another observation, so `process()` was called
**once per chunk, about every 1.8 s**, while every window in the module is a
converter frame count at 30 Hz (W=5 is 167 ms). At the observation cadence those
integers mean 9 s and 18 s, and a "carry-forward" inserted a frame one whole
chunk old — then committed it to the keyframe buffer.

The filter now runs in the client's existing 30 Hz tactile fetch thread, where
those numbers mean what they say and a fill is 33 ms old. Two consequences shape
the rest of the design:

**The client decides, and simply does not send.** It reads the health snapshot at
the chunk boundary and, if the tactile is not something the model was trained on,
builds no observation and makes no RPC. The server's refusal path is a backstop
for a contract bug, not a control path — routing an unsafe request through the
server just to be refused would make the failure path part of normal control.

**The server is stateless about health.** It never constructs a filter. It
verifies the client's claim (contract sha, status, and that no finger actually
arrived all-zero) and refuses on disagreement; `web_infer_scripts/test_server_health_contract.py`
asserts, by reading the source, that no filter is instantiated.

### Status taxonomy

| status | usable | meaning |
|---|---|---|
| `ok` | yes | every finger live |
| `degraded_safe` | **yes** | in-window carry-forward fill (`age ≤ W`) |
| `not_ready` | no | no `last_good` yet; offline these frames were trimmed away |
| `sensor_unavailable` | no | `age > W`: longer than training ever filled |
| `fault_latched` | no | declared for wrapper-level faults; never emitted here |

`degraded_safe` executes, and that is deliberate. The converter kept short
internal gaps in the corpus as previous-good carry-forward and the demonstrator's
actions trained against them, so an in-window fill is input the model has already
seen; holding on it would be *less* faithful to training. Stopping is reserved for
the two cases where that argument fails.

Nothing latches permanently. `age > W` says the gap outran what training filled,
not that the sensor is broken, so both holding states clear on their own — but
only after `recovery_valid_streak = 3` consecutive **strictly live** frames (a
fill never counts). That is what stops a flickering sensor from starting and
stopping the arm, and it also guarantees every `last_good` was refreshed after the
gap, so nothing is ever filled from before it. `fault_latched` exists for
persistent faults that are not visible in a tactile frame (a wedged fetch thread,
a device disconnect); wiring its triggers is deferred.

A chunk already executing is **not** aborted. It was computed from tactile that
was healthy at the time, and stopping mid-chunk would produce a deceleration
profile nothing was validated against. The hold takes effect at the next chunk
boundary, so the worst case is a 1.8 s tail — accepted, and the reason
`num_action_execute` stays at 54 (lowering it would make keyframe spacing
irregular, which is its own train/serve mismatch).

### Client structure

The 30 Hz fetch thread publishes an immutable `TactilePolicySnapshot` through a
one-lock `TactileHealthChannel`, so the control loop can never read a status from
one tick next to frames from another. The snapshot carries an **epoch** (bumped
per episode) and a capture timestamp; the control loop refuses a snapshot from a
previous episode, one older than 3 fetch periods (a stalled thread must read as a
hold, never as a stale `ok`), or one the filter marked unusable.

Recording and policy paths are isolated by construction: the shared buffers keep
the **pristine** frames — a dataset of our own fills would be unusable for
training — and the filled copy exists only inside the snapshot.

The client's vendored copy of the module lives at `teleop/vendor/tactile_health.py`
and must be byte-identical; both ends publish `sha256` of their own file and the
server refuses a mismatch. Re-vendor with `teleop/vendor/sync_from_deploy.sh`.

### Prompt provenance (same rollout, same class of bug)

`eval_genie.py` hardcoded the erase instruction and the server took
`obs.get("prompt", "")` verbatim, so serving bowl or tong conditioned a right-only
pick policy on whiteboard erasing — silently, with plausible-looking motion.
`data/utils/task_registry.py` now owns the served prompt: `--task` is required,
the entry is cross-checked against the config's own domain and arm layout at
startup, ping publishes `task_id` / `task_prompt` / `task_prompt_sha256`, the
client takes the prompt from ping and verifies the hash, and the server refuses a
step whose prompt echo disagrees. Prompts are copied byte-exact from the
`run_convert.sh` branch that produced each dataset and are verifiable against the
dataset itself:

```bash
python -m data.utils.task_registry --task bowl \
    --verify-dataset-root data/datasets_lerobot/20260725_pinch_from_bowl_with_fingers_right_only
```

`--task none` keeps the pre-registry behaviour for offline tooling (parity replay,
byte-parity capture): prompt verbatim, health verification off, `task_id: null` in
ping — which the hardware client refuses to connect to.

### Validation

| what | result |
|---|---|
| `data/utils/tactile_health.py` | 23 self-tests: W and W+1 for both windows, hold/auto-recovery, partial streak, fill-does-not-count-as-live, leading limit, worst-finger-wins, reset, no input mutation, no output aliasing |
| `web_infer_scripts/test_server_health_contract.py` | 19 checks: accepts `ok`/`degraded_safe`, refuses a missing block, version skew, edited vendored copy, every unusable status, and any all-zero finger under a usable claim; names the correct hand/finger; asserts the server constructs no filter |
| `teleop/tests/test_tactile_health_client.py` | 39 checks through the **real** `_tactile_fetch_loop`: hand mapping (a left dropout does not hold a right-only policy but does hold a bimanual one), recording buffers stay pristine, no aliasing either way, dtype/shape, W/W+1/3-frame recovery end to end, leading gap, epoch invalidation, staleness, and a live threaded publish/read |
| real episodes (bowl) | 100 kept / 0 discarded, 36712 frames, **3001 blank finger-frames filled byte-identically**; blank detection also matches `seg.black_original` |
| real episodes (tong) | 100 kept / 0 discarded, 92282 frames, **6529 blank finger-frames filled byte-identically**, zero divergence |
| accepted divergence | 6 frames, all bowl, 2/200 segments (0.016% of bowl frames, 0 for tong) |
| registry prompts | bowl `cb09decd…` and tong `5a8bc0fb…` byte-identical to each dataset's `meta/tasks.jsonl`; unscrew_v2 `843a6f8e…` likewise, checked on the training host 2026-08-01 |

Re-measured on the training host at `bbff175` under the new status names; both PASS. The
per-episode fill is byte-for-byte what it was, as expected — the status rework
changed naming and the hold policy, not the fill arithmetic. Bowl's accepted
divergence grew from 2 frames to 6 because the recovery streak now counts as
part of the leading withheld run: each of the two affected segments withholds
its leading blank plus the two live frames the 3-frame streak needs before the
first usable one. Neither corpus produced a gap-discard, so
`0 gap-discards reproduced as sensor_unavailable` is a statement about the data
rather than evidence about that path.

The accepted divergence is a segment that BEGINS with a blank required finger,
which offline is backfilled from a later frame. No causal filter can reproduce
that, and it survives the leading trim because `classify_runs` only calls a run
"leading" when it starts at absolute frame 0 — a trim boundary landing on the
start of an internal run leaves that run at the head of the segment. Online those
frames are withheld, so the arm holds a moment longer at episode start.

Not covered by real data: neither corpus produced a gap-discard, so
`sensor_unavailable` rests on synthetic and wrapper tests alone. Both datasets
kept 100/100 episodes once the operating-finger windows were tuned, so the first
real over-long dropout will be the first live exercise of that path. The failure
direction is a hold.

Health logging writes one JSONL line per step to `$DEXVTAM_HEALTH_LOG_DIR`
(default `/tmp/dexvtam_health_logs`), one file per episode, plus a summary on
roll. It now records what the client *claimed* next to what actually arrived,
which is the pair worth having when the two disagree. A logging failure can never
take the policy down.

`web_infer_scripts/tactile_blank_stats.py` measures the dropout structure that
sizes these thresholds (tick-level degraded rate, simultaneous-blank histogram,
per-finger run lengths against each finger's own window, blank-vs-contact phase,
carry-forward delta as a multiple of the ordinary frame-to-frame change).

### Gate 11 — the real client wrapper, on real episodes

`teleop/tests/gate11_wrapper_replay.py` (hardware repo). Everything else in this
area proves something narrower than it sounds: the self-tests prove the state
machine, `parity_tactile_fill` proves it agrees with the converter but calls
`process()` directly, the earlier wrapper tests drive the real loop on scripted
frames, and the vendored sha proves both repos hold the same algorithm while
saying nothing about where it was installed. Gate 11 closes the remaining gap,
which is the wiring.

A converter-kept segment's PRE-fill frames are served through the real
`_tactile_fetch_loop` — as the hardware produces a dropout, with the SDK
returning nothing and the client's own pre-zeroed row making the frame blank —
and the published snapshots are compared to the converter's filled output. The
same frames also go straight through the filter, which separates the two failure
modes: wrapper vs direct isolates wiring, direct vs offline isolates the
algorithm.

| | bowl `episode_0001` seg 0 | tong `episode_0000` seg 0 |
|---|---|---|
| frames / blank finger-frames | 454 / 47 | 1006 / 91 |
| wrapper == direct (wiring) | PASS | PASS |
| byte-identical to the converter | PASS | PASS |
| one snapshot per frame, `seq` increasing | PASS | PASS |
| uint8, (2,5,240,320) | PASS | PASS |
| recording buffer holds the PRE-fill frame | PASS | PASS |
| no aliasing between snapshot and buffer | PASS | PASS |
| left rows untouched by a right-only policy | PASS | PASS |

`--self-test` runs the same replay path on fabricated frames with neither the
converter nor a dataset, so a failure on the training host is a finding rather than a bug in
the harness. Negative controls confirm the gate can fail: it catches a
transposed channel map, a policy fed the wrong hand's rows, and a snapshot that
aliases the recording buffer.

### The control-loop decision

The pre-RPC gate is now `decide_next_request` in `eval_genie.py` rather than
inline in `main()`, so the decision the whole health design exists to make is
verified off-robot instead of on one. `teleop/tests/test_request_gate.py`, 29
checks: `ok` and `degraded_safe` send; `not_ready`, `sensor_unavailable`, a
missing snapshot, a snapshot from a previous episode, a stale one and a `seq`
that has not advanced all hold; the more specific reason wins when several apply.

The timing half is driven by snapshots the **real** fetch loop produced from a
scripted dropout, with a counter standing in for the server:

| | |
|---|---|
| `age == W` | every frame sent, nothing withheld |
| `age == W+1` | the over-window frame is not sent, and nothing after it is |
| recovery | the first two live frames still hold; the third sends exactly one RPC |
| partial streak | no RPC at all |
| clean stream | one RPC per chunk boundary, not per tick |
| healthy but 10x stale | refused |

Refusing a `seq` that has not advanced is new behaviour rather than moved
behaviour: it makes "never send the same frame twice" true by construction
instead of as a consequence of timing.

**Outstanding before F4b passes:** one integration confirmation on the robot —
inject blank tactile in software and check that the arm stops at the next chunk
boundary rather than mid-chunk, that no stale rows are applied, and that
recovery takes three live frames. Every layer under that is now covered offline,
so a failure there points at the robot integration rather than at the logic.

Until F4b passes, the robot session is limited to non-contact bring-up: homing,
camera and tactile stream checks, the ping contract, a non-contact dry-run,
left-arm constant-hold verification, free-space IK motion and controller
tracking tolerance. No bowl/tong contact-rich rollout.
