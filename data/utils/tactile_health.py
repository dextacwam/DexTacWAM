"""Online tactile dropout fill + health state machine.

Runs in the CLIENT's 30 Hz tactile fetch thread, one call per fetched frame.
That placement is the whole point: the windows below are converter *frames*
(W=5 is 167 ms at 30 Hz), so a filter called once per action chunk -- every
1.8 s -- would stretch W=5 into 9 seconds and carry a frame a whole chunk old
into the model's memory. The arithmetic here is only correct at the rate the
training corpus was recorded at.

MIRRORED FROM THE CONVERTER (``vtam_data_scripts/data_fixes.py``; do not
"improve" these independently):

  * blank  := a finger's raw frame is EXACTLY all-zero, ``np.all(raw == 0)``.
    Not a mean/threshold test -- ``detect_dropout``.
  * INTERNAL run (the finger has produced at least one good frame) of length
    ``<= W`` -> carry-forward fill from the most recent good frame. W is per
    finger: strict ``SHORT_RUN_MAX`` for the task's operating fingers, lenient
    ``nonoperating_short_run_max`` for the rest (``internal_window``).
  * a run ``> W`` is one the converter would have DISCARDED the episode over,
    so no such frame exists in training. Online that is ``sensor_unavailable``.
  * LEADING run (no good frame yet) is a different policy offline: the converter
    TRIMS it, up to ``BOUNDARY_TRIM_MAX``, and discards beyond that. Trimmed
    frames are not in the dataset at all, so online the honest equivalent is to
    withhold the policy -- ``not_ready`` -- rather than invent a fill.

That last rule is what makes offline/online parity attainable: the fill inside a
kept segment is causal, and this module reproduces it frame for frame -- verified
byte-exact on real episodes by ``web_infer_scripts/parity_tactile_fill.py``.

Why a fill still EXECUTES
-------------------------
``degraded_safe`` is usable, not merely tolerable. The converter kept short
internal gaps in the corpus as previous-good carry-forward and the demonstrator's
actions trained against them, so a filled frame inside the window is input the
model has already seen. Holding on it would be *less* faithful to training, not
more. Stopping is reserved for the cases where that argument fails: no history to
fill from, or a gap longer than the corpus ever contained.

Everything here is RECOVERABLE
------------------------------
``fill_age > W`` says this gap outran what training ever filled -- it is not
evidence of a broken sensor. If every required finger returns real data the
observation is fully valid again, so the hold clears on its own. What it must
not do is clear on a single lucky frame: a flickering sensor would start and
stop the arm repeatedly. Resuming therefore requires ``recovery_valid_streak``
consecutive frames in which every finger is STRICTLY LIVE (not filled), which
also guarantees every ``last_good`` was refreshed after the gap, so nothing is
ever filled from before it.

``STATUS_FAULT_LATCHED`` is declared for the persistent faults that belong to the
wrapper rather than to this arithmetic -- a wedged fetch thread, a snapshot that
stopped advancing, a device disconnect, a hold that outlasts its time bound. None
of those are detectable from a tactile frame, so this module never emits it. A
latch with no clearly defined trigger makes logs and tests unexplainable.

One bounded, measured divergence: a kept segment can BEGIN with a blank required
finger, which ``carry_forward_fill`` backfills from a LATER frame. No causal
filter can match that. On the bowl corpus it was 1 frame in 8418 across 20
episodes. Online those frames are withheld, so the arm holds a moment longer at
episode start -- the safe direction. The converter's other divergence is
structural: it can discard an episode retroactively, and we cannot un-execute.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERS = 5

# Bumped whenever the wire-visible semantics change, so a version skew reports
# WHICH side is older instead of just "different". v1 was the server-side
# variant with an unconditional sensor_lost latch.
CONTRACT_VERSION = 2

# vtam_data_scripts/data_fixes.py: BOUNDARY_TRIM_MAX. Leading dropout is trimmed
# offline up to this length and discards the episode beyond it.
LEADING_TRIM_MAX = 15

# Frames of strictly-live tactile required to clear a hold (~100 ms at 30 Hz).
DEFAULT_RECOVERY_VALID_STREAK = 3

STATUS_OK = "ok"
STATUS_DEGRADED_SAFE = "degraded_safe"
STATUS_NOT_READY = "not_ready"
STATUS_SENSOR_UNAVAILABLE = "sensor_unavailable"
# Declared for the wrapper's persistent faults; never emitted by this module.
STATUS_FAULT_LATCHED = "fault_latched"

USABLE_STATUSES = (STATUS_OK, STATUS_DEGRADED_SAFE)
ALL_STATUSES = (
    STATUS_OK,
    STATUS_DEGRADED_SAFE,
    STATUS_NOT_READY,
    STATUS_SENSOR_UNAVAILABLE,
    STATUS_FAULT_LATCHED,
)

# Worst (most restrictive) wins when fingers disagree within a tick.
_STATUS_RANK = {
    STATUS_OK: 0,
    STATUS_DEGRADED_SAFE: 1,
    STATUS_NOT_READY: 2,
    STATUS_SENSOR_UNAVAILABLE: 3,
    STATUS_FAULT_LATCHED: 4,
}


def module_sha256() -> str:
    """Hash of this file's bytes.

    The client vendors a copy of this module and both ends publish the hash, so
    a divergent copy is a refused connection rather than two subtly different
    fill policies agreeing on a status name.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class TactileHealthResult:
    """One tick's verdict. ``tactile`` is only meaningful for a usable status."""

    tactile: np.ndarray            # (V_hand, F, H, W) uint8, carry-forward filled
    status: str
    blank_mask: np.ndarray         # (V_hand, F) bool -- all-zero on the wire this tick
    filled_mask: np.ndarray        # (V_hand, F) bool -- actually replaced from history
    fill_age: np.ndarray           # (V_hand, F) int  -- length of the current blank run
    valid_streak: int              # consecutive strictly-live frames
    holding: bool                  # withholding until the recovery streak is met
    fault_latched: bool = False    # always False here; the wrapper owns real faults
    reasons: tuple[str, ...] = ()  # per-finger explanations, for the event log

    @property
    def usable(self) -> bool:
        """Whether this observation may be sent to the policy."""
        return self.status in USABLE_STATUSES

    @property
    def memory_safe(self) -> bool:
        """Whether it may enter the server's keyframe buffer.

        Identical to ``usable`` by construction now that the client decides
        before the RPC: a frame that is not usable is never sent, so it cannot
        reach the buffer. Kept as a separate name because the two answer
        different questions and only coincide under this architecture.
        """
        return self.usable


class OnlineTactileHealthFilter:
    """Per-(hand, finger) causal dropout fill with recoverable holds.

    Constructed from the SAME window table the server publishes in ping, so the
    thresholds a client sees and the thresholds actually enforced cannot drift.
    """

    def __init__(
        self,
        hands: tuple[str, ...],
        windows: dict[tuple[str, int], int],
        leading_trim_max: int = LEADING_TRIM_MAX,
        recovery_valid_streak: int = DEFAULT_RECOVERY_VALID_STREAK,
    ) -> None:
        self.hands = tuple(hands)
        self.leading_trim_max = int(leading_trim_max)
        self.recovery_valid_streak = int(recovery_valid_streak)
        if self.recovery_valid_streak < 1:
            raise ValueError("recovery_valid_streak must be >= 1")
        missing = [(h, f) for h in self.hands for f in range(FINGERS)
                   if (h, f) not in windows]
        if missing:
            raise KeyError(f"no blank window for {missing}; windows={sorted(windows)}")
        self.windows = {(h, f): int(windows[(h, f)])
                        for h in self.hands for f in range(FINGERS)}
        self.reset()

    # ---- per-episode state -------------------------------------------------
    def reset(self) -> None:
        n = len(self.hands)
        self._last_good: list[list[np.ndarray | None]] = [
            [None] * FINGERS for _ in range(n)]
        self._blank_age = np.zeros((n, FINGERS), dtype=np.int64)
        self._seen_good = np.zeros((n, FINGERS), dtype=bool)
        self._valid_streak = 0
        # A hold, unlike v1's latch, names the status it is waiting out and
        # clears itself once the sensor proves it is back.
        self._hold_status: str | None = None
        self.step_index = -1

    @property
    def holding(self) -> bool:
        return self._hold_status is not None

    @property
    def hold_status(self) -> str | None:
        return self._hold_status

    # ---- one tick ----------------------------------------------------------
    def process(self, tactile: np.ndarray) -> TactileHealthResult:
        """tactile: (V_hand, F, H, W) uint8, ALREADY trimmed to the model's hands."""
        tac = np.asarray(tactile)
        n = len(self.hands)
        if tac.ndim != 4 or tac.shape[0] != n or tac.shape[1] != FINGERS:
            raise ValueError(
                f"tactile must be ({n}, {FINGERS}, H, W) for hands={self.hands}; "
                f"got {tac.shape}")
        if tac.dtype != np.uint8:
            raise ValueError(f"tactile must be uint8 (raw), got {tac.dtype}")

        self.step_index += 1
        out = tac.copy()
        blank_mask = np.zeros((n, FINGERS), dtype=bool)
        filled_mask = np.zeros((n, FINGERS), dtype=bool)
        reasons: list[str] = []
        status = STATUS_OK

        for hi, hand in enumerate(self.hands):
            for f in range(FINGERS):
                frame = tac[hi, f]
                # Exactly the converter's detect_dropout test.
                is_blank = not frame.any()
                blank_mask[hi, f] = is_blank
                if not is_blank:
                    self._last_good[hi][f] = frame.copy()
                    self._seen_good[hi, f] = True
                    self._blank_age[hi, f] = 0
                    continue

                self._blank_age[hi, f] += 1
                age = int(self._blank_age[hi, f])
                if not self._seen_good[hi, f]:
                    # Leading gap: offline this was trimmed, not filled. Past the
                    # trim limit the episode would have been discarded, which we
                    # report distinctly but still treat as recoverable -- nothing
                    # has been executed yet, so there is nothing to protect.
                    if age > self.leading_trim_max:
                        status = _worse(status, STATUS_SENSOR_UNAVAILABLE)
                        reasons.append(
                            f"{hand}{f}: no first frame after {age} blanks "
                            f"(leading limit {self.leading_trim_max})")
                    else:
                        status = _worse(status, STATUS_NOT_READY)
                        reasons.append(f"{hand}{f}: awaiting first good frame ({age})")
                    continue

                w = self.windows[(hand, f)]
                if age > w:
                    status = _worse(status, STATUS_SENSOR_UNAVAILABLE)
                    reasons.append(f"{hand}{f}: blank {age} > window {w}")
                else:
                    out[hi, f] = self._last_good[hi][f]
                    filled_mask[hi, f] = True
                    status = _worse(status, STATUS_DEGRADED_SAFE)
                    reasons.append(f"{hand}{f}: filled, blank {age}/{w}")

        # Strictly LIVE, not merely usable: a filled frame must not count toward
        # recovery, or a finger blanking every other frame would resume the arm.
        live = not bool(blank_mask.any())
        self._valid_streak = self._valid_streak + 1 if live else 0

        if status in (STATUS_NOT_READY, STATUS_SENSOR_UNAVAILABLE):
            self._hold_status = status
        elif self._hold_status is not None:
            if live and self._valid_streak >= self.recovery_valid_streak:
                reasons.append(
                    f"recovered from {self._hold_status} after "
                    f"{self._valid_streak} live frames")
                self._hold_status = None
            else:
                # Still waiting out the streak. Keep reporting the hold's own
                # status so a log reads as one continuous event, and so callers
                # counting withheld frames see the whole run.
                status = self._hold_status
                reasons.append(
                    f"holding ({self._hold_status}): live streak "
                    f"{self._valid_streak}/{self.recovery_valid_streak}")

        return TactileHealthResult(
            tactile=out,
            status=status,
            blank_mask=blank_mask,
            filled_mask=filled_mask,
            fill_age=self._blank_age.copy(),
            valid_streak=self._valid_streak,
            holding=self.holding,
            fault_latched=False,
            reasons=tuple(reasons),
        )


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


# --------------------------------------------------------------------------- #
# self-tests: the edge cases a real-episode replay is unlikely to contain
# --------------------------------------------------------------------------- #
def _mk(vals) -> np.ndarray:
    """(F,) per-finger constant frames -> (1, F, 2, 2) uint8; 0 means blank."""
    return np.stack([np.full((2, 2), v, np.uint8) for v in vals])[None]


def _run_tests() -> None:  # noqa: C901
    W = {("right", 0): 5, ("right", 1): 5, ("right", 2): 5,
         ("right", 3): 10, ("right", 4): 10}
    good = [10, 20, 30, 40, 50]
    blank0 = [0, 20, 30, 40, 50]
    R = DEFAULT_RECOVERY_VALID_STREAK

    def fresh(**kw):
        return OnlineTactileHealthFilter(("right",), W, **kw)

    def live_frames(fl, k):
        r = None
        for _ in range(k):
            r = fl.process(_mk(good))
        return r

    # 1. all good -> ok, nothing filled, streak counts up
    fl = fresh()
    for i in range(3):
        r = fl.process(_mk(good))
        assert r.status == STATUS_OK, r.status
        assert not r.filled_mask.any() and r.valid_streak == i + 1
        assert np.array_equal(r.tactile, _mk(good))
        assert r.usable and not r.holding

    # 2. blank at the very first frame -> not_ready, NOT a fill: there is no
    #    history to carry forward, and offline this frame was trimmed away.
    fl = fresh()
    r = fl.process(_mk(blank0))
    assert r.status == STATUS_NOT_READY, r.status
    assert not r.filled_mask.any()
    assert not r.memory_safe and not r.usable

    # 3. carry-forward reproduces the last good frame exactly, AND executes
    fl = fresh()
    fl.process(_mk(good))
    r = fl.process(_mk(blank0))
    assert r.status == STATUS_DEGRADED_SAFE and r.filled_mask[0, 0]
    assert np.array_equal(r.tactile, _mk(good)), "fill must be byte-exact"
    assert r.usable and r.memory_safe, "an in-window fill must not stop the robot"

    # 4. off-by-one at the window: age == W still fills, age == W+1 holds.
    for finger, w in ((0, 5), (3, 10)):
        fl = fresh()
        fl.process(_mk(good))
        blank = list(good)
        blank[finger] = 0
        for age in range(1, w + 1):
            r = fl.process(_mk(blank))
            assert r.status == STATUS_DEGRADED_SAFE, (finger, age, r.status)
            assert r.usable and int(r.fill_age[0, finger]) == age
        r = fl.process(_mk(blank))
        assert r.status == STATUS_SENSOR_UNAVAILABLE, (finger, w + 1, r.status)
        assert not r.usable and r.holding

    # 5. per-finger windows are independent: finger 3 (W=10) survives a run that
    #    would kill finger 0 (W=5)
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(6):
        r = fl.process(_mk([10, 20, 30, 0, 50]))
    assert r.status == STATUS_DEGRADED_SAFE and r.usable, r.status

    # 6. recovery resets the run, so two short runs never add up
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(5):
        fl.process(_mk(blank0))
    r = fl.process(_mk(good))
    assert r.status == STATUS_OK and int(r.fill_age[0, 0]) == 0
    for _ in range(5):
        r = fl.process(_mk(blank0))
    assert r.status == STATUS_DEGRADED_SAFE, r.status

    # 7. over-window HOLDS, then auto-recovers -- but only after a full streak
    #    of strictly live frames. This replaces v1's permanent latch.
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(6):
        fl.process(_mk(blank0))
    assert fl.holding
    for i in range(1, R):
        r = fl.process(_mk(good))
        assert r.status == STATUS_SENSOR_UNAVAILABLE, (i, r.status)
        assert not r.usable, "must not resume before the streak is met"
    r = fl.process(_mk(good))
    assert r.status == STATUS_OK and r.usable and not r.holding
    assert r.valid_streak == R

    # 8. flicker must produce ONE hold, not a restart per lucky frame
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(6):
        fl.process(_mk(blank0))          # -> sensor_unavailable
    r = fl.process(_mk(good))            # single good frame
    assert not r.usable, "one good frame must not resume"
    for _ in range(6):
        r = fl.process(_mk(blank0))
    assert r.status == STATUS_SENSOR_UNAVAILABLE and r.holding

    # 9. the recovery frame becomes last_good: nothing is ever filled from
    #    before the gap
    fl = fresh()
    fl.process(_mk(good))                        # pre-gap value 10
    for _ in range(6):
        fl.process(_mk(blank0))
    recovered = [11, 20, 30, 40, 50]
    for _ in range(R):
        fl.process(_mk(recovered))
    r = fl.process(_mk([0, 20, 30, 40, 50]))
    assert r.status == STATUS_DEGRADED_SAFE and r.filled_mask[0, 0]
    assert np.array_equal(r.tactile, _mk(recovered)), "filled from a pre-gap frame"

    # 10. reset() clears the hold AND the history, so the next episode cannot
    #     carry forward a frame from the previous one
    fl.reset()
    assert not fl.holding
    r = fl.process(_mk(blank0))
    assert r.status == STATUS_NOT_READY, r.status

    # 11. worst finger wins: one finger over-window while another is merely filled
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(6):
        r = fl.process(_mk([0, 0, 30, 40, 50]))
    assert r.status == STATUS_SENSOR_UNAVAILABLE, r.status

    # 12. one finger recovers while another stays blank
    fl = fresh()
    fl.process(_mk(good))
    fl.process(_mk([0, 0, 30, 40, 50]))
    r = fl.process(_mk([10, 0, 30, 40, 50]))
    assert r.status == STATUS_DEGRADED_SAFE
    assert not r.filled_mask[0, 0] and r.filled_mask[0, 1]
    assert int(r.fill_age[0, 0]) == 0 and int(r.fill_age[0, 1]) == 2

    # 13. leading gap: not_ready up to the trim limit, sensor_unavailable beyond,
    #     and either way it recovers rather than latching
    fl = fresh()
    for age in range(1, LEADING_TRIM_MAX + 1):
        r = fl.process(_mk(blank0))
        assert r.status == STATUS_NOT_READY, (age, r.status)
    r = fl.process(_mk(blank0))
    assert r.status == STATUS_SENSOR_UNAVAILABLE, r.status
    r = live_frames(fl, R)
    assert r.status == STATUS_OK and r.usable

    # 14. a leading hold is not cleared by a partial streak
    fl = fresh()
    fl.process(_mk(blank0))
    for i in range(1, R):
        r = fl.process(_mk(good))
        assert r.status == STATUS_NOT_READY, (i, r.status)
    r = fl.process(_mk(good))
    assert r.status == STATUS_OK

    # 15. a filled frame does NOT count toward recovery (strictly live only)
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(6):
        fl.process(_mk(blank0))              # over window -> hold
    for _ in range(20):
        r = fl.process(_mk([10, 0, 30, 40, 50]))   # finger 1 fillable, not live
        assert not r.usable, "a fill must never satisfy the recovery streak"
    r = live_frames(fl, R)
    assert r.status == STATUS_OK

    # 16. a finger that arrives late is fillable from ITS first frame onward
    fl = fresh()
    fl.process(_mk(blank0))
    live_frames(fl, R)
    r = fl.process(_mk(blank0))
    assert r.status == STATUS_DEGRADED_SAFE and r.filled_mask[0, 0]
    assert np.array_equal(r.tactile, _mk(good))

    # 17. not_ready outranks a fill: never run the policy on a half-filled
    #     observation just because some other finger was fillable
    fl = fresh()
    fl.process(_mk([10, 0, 30, 40, 50]))       # finger 1 never seen
    r = fl.process(_mk([0, 0, 30, 40, 50]))    # finger 0 fillable, finger 1 not
    assert r.status == STATUS_NOT_READY, r.status

    # 18. a nonzero frame is never "blank", however dark
    fl = fresh()
    r = fl.process(_mk([1, 20, 30, 40, 50]))
    assert r.status == STATUS_OK and not r.blank_mask.any()

    # 19. fault_latched is declared but never emitted by this module
    fl = fresh()
    fl.process(_mk(good))
    for _ in range(40):
        r = fl.process(_mk(blank0))
        assert r.status != STATUS_FAULT_LATCHED and not r.fault_latched

    # 20. shape / dtype guards
    fl = fresh()
    for bad in (np.zeros((1, 4, 2, 2), np.uint8), np.zeros((2, 5, 2, 2), np.uint8)):
        try:
            fl.process(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad.shape}")
    try:
        fl.process(np.zeros((1, 5, 2, 2), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted float tactile")

    # 21. the input array is never mutated
    fl = fresh()
    fl.process(_mk(good))
    src = _mk(blank0)
    before = src.copy()
    fl.process(src)
    assert np.array_equal(src, before), "process() must not write to its input"

    # 22. the returned array never aliases the input or the history
    fl = fresh()
    fl.process(_mk(good))
    src = _mk(blank0)
    r = fl.process(src)
    r.tactile[0, 0, 0, 0] = 99
    assert src[0, 0, 0, 0] == 0
    r2 = fl.process(_mk(blank0))
    assert r2.tactile[0, 0, 0, 0] == 10, "history was corrupted by a caller write"

    # 23. recovery_valid_streak=1 resumes on the first live frame (config, not code)
    fl = fresh(recovery_valid_streak=1)
    fl.process(_mk(good))
    for _ in range(6):
        fl.process(_mk(blank0))
    r = fl.process(_mk(good))
    assert r.status == STATUS_OK and not r.holding

    print(f"tactile_health: all tests passed "
          f"(contract v{CONTRACT_VERSION}, sha {module_sha256()[:12]})")


if __name__ == "__main__":
    _run_tests()
