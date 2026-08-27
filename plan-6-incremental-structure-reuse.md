# Plan 6 — Stop rebuilding all structure on every incremental add

Agent plan for the COLMAP fork at `W:\Kyle_Shepherd\photogrammetry_app\colmap`,
branch `incremental-glomap`. Follows plan-5, whose fix
(`skip_global_positioning = true` on the incremental path) is implemented and
verified on the §4 reproducer for all 12 seeds.

Goal: make an incremental add cost ~O(window) instead of ~O(N), by reusing the
prior reconstruction's 3D points and tracks instead of deleting and
re-triangulating the entire model every time; and remove the dead
track-establishment + bundle-adjustment work that plan-5's fix exposed.

**Revision note (2026-08-27).** This document was revised after a code audit
that changed three of its conclusions. `full_solve_interval` is deleted rather
than decided about (§7). The "lossy by design" windowed import turns out to be
lossless on a healthy prior, so it becomes a tripwire rather than a tolerance
(§6). And the audit found a structural defect the original plan did not name:
**a windowed solve can never refine camera intrinsics**, which is the actual
mechanism behind the `7756a83c` collapse (§5). The original §1–§2 diagnosis
stands and is preserved below.

---

## 1. The important thing to know before planning anything

**The mechanism already exists in this fork.** `GlobalMapper::SolveIncrementalWindowed`
(`src/colmap/sfm/global_mapper.cc`) already does exactly what this plan is
nominally about:

- imports the prior reconstruction's 3D points, preserving their `point3D_t`
  ids and re-linking each track element's 2D-3D association;
- drops track elements whose image/keypoint is missing from the database cache,
  and skips tracks left with < 2 views;
- triangulates **only the new image(s)** against that existing structure via
  `IncrementalMapper::TriangulateImage`;
- runs `IterativeLocalRefinement` over a covisibility window of
  `optimize_window_size` images, with everything outside the window entering as
  constant-pose residuals, so the output stays in the prior frame by
  construction (no Sim3 realignment needed);
- never calls `EstablishTracks`, `GlobalPositioning`, or
  `DeleteAllPoints2DAndPoints3D`.

It is CLI-exposed as top-level `--optimize_window_size N` (registered in
`src/colmap/exe/sfm.cc`, *not* in `option_manager.cc` under the
`GlobalMapper.` prefix — that is why a grep of `option_manager.cc` misses it).

So this is **not** a "write a new feature" plan. It is a *validate, default, and
clean up* plan. The single highest-value action is measurement, not code.

Why it is off: `optimize_window_size` defaults to `0`
(`src/colmap/sfm/global_mapper.h`), and the server invokes the binary with
no tuning flags at all (`app/pipeline/colmap_runner.py` passes only
`--database_path`, `--image_path`, `--prior_reconstruction_path`,
`--output_path`). **Production therefore runs the full non-windowed path on
every add.**

---

## 2. Where the waste actually is

Measured on `sessions/camsnap`, prior `after_img_0011...` (12 images), adding
`img_0012`, patched binary, defaults:

| stage | wall time | what it produces | fate |
|---|---|---|---|
| track establishment | 0.010 s | 4066 tracks → 1436 kept | **deleted unused** |
| iterative bundle adjustment | 0.433 s | refines poses against those tracks | tracks **deleted** |
| iterative retriangulation + refinement | 0.975 s | the actual output structure | kept |
| **total** | **1.419 s** | | |

Two distinct problems:

**(a) The full model is rebuilt every add.** `IterativeRetriangulateAndRefine`
opens with:

```cpp
// Delete all existing 3D points and re-establish 2D-3D correspondences.
reconstruction_->DeleteAllPoints2DAndPoints3D();
```

then re-triangulates **every** registered image. Cost grows with N on every
add, so a drip-fed session is ~O(N²) overall. Prior points are never reused:
`LoadPriorPoses` imports poses and intrinsics only, keeping prior 3D points
solely as a ≤1000-point `prior_points_sample_` for the post-bootstrap
cheirality check. Confirmed by counts: prior 1986 points → output 1831 (a
rebuild, not an extension).

**(b) The wart.** `EstablishTracks` creates tracks with default-constructed
`Point3D`, i.e. `xyz = Eigen::Vector3d::Zero()` (`src/colmap/scene/point3d.h`).
With global positioning now skipped, nothing ever assigns those points a
position before bundle adjustment sees them. So BA spends 0.433 s doing what
amounts to triangulation-by-gradient-descent from a degenerate all-at-origin
configuration — and then retriangulation deletes every one of those points and
does the job properly. That is ~31% of the add spent on structure that is
discarded unused.

Note BA is not *purely* wasted: it also refines poses and intrinsics, and those
**do** persist into retriangulation. So "just skip it" is a hypothesis to
measure, not a free win (see P6).

### What I could not measure

I attempted a scaling curve across the `sessions/camsnap/priors/after_*`
snapshots and the result was **invalid**: the database holds all 13 images, so
a prior with N registered images bulk-bootstraps all 13−N remaining ones
(8 new images for `after_img_0004`, 4 for `after_img_0008`). Those runs are not
comparable single-add measurements, and several short-circuited early. Any
scaling claim must come from `scripts/python/incremental_benchmark.py`, which
builds per-step **subset databases** that mimic the production server's growing
database. Do not reuse my table; it does not exist for a reason.

---

## 3. What `optimize_window_size` actually controls

Verified by reading the call chain, because two of the plan's original
assumptions about it were wrong.

`optimize_window_size` flows to exactly one place: it becomes
`IncrementalMapper::Options::ba_local_num_images` in
`SolveIncrementalWindowed`, which is the size of the local bundle-adjustment
problem. Per add: triangulate the new image against imported prior structure,
then `IterativeLocalRefinement` runs up to 2 rounds of `AdjustLocalBundle`.
`FindLocalBundle` (`incremental_mapper_impl.cc`) selects the window: the new
image plus its `N−1` most covisible registered images by shared-observation
count, subject to a triangulation-angle gate that relaxes through 8 successive
thresholds if too few images qualify.

Inside that BA:

- the N window images have free poses (minus `constant_frames`, i.e. gauge
  anchors);
- **out-of-window images do contribute.** `AddPointToProblem`
  (`bundle_adjustment_ceres.cc`) walks each variable point's *full* track and
  adds a `ReprojErrorConstantPoseCostFunctor` for every element whose image is
  not in the config. The far field is present as constant-pose residuals,
  as the original plan said;
- points are filtered by reprojection error and triangulation angle over the
  window's images *and* all touched points — this deletes observations from
  **imported prior points**, not just new ones. This is the decay engine
  §6 and §8 are about.

### 3.1 Track length ≤ 15 is not a discard, and does not contradict GLOMAP

`AdjustLocalBundle` has `constexpr size_t kMaxTrackLength = 15`, marking only
new (`!HasError()`) or short tracks as *variable*. This is upstream COLMAP, not
fork code. It looks arbitrary and like a conflict with GLOMAP's known
preference for long tracks. It is neither, because the two are different
operations.

GLOMAP does prefer long tracks, and this repo proves it: `EstablishTracks`
(`global_mapper.cc`) sorts candidate tracks by length **descending** and
greedily selects longest-first until every view meets its quota. Long tracks
are the preferred constraint for global positioning.

But a long prior track in a windowed add is not dropped. Trace it:

1. not added to `VariablePoints()`;
2. still enters the problem via `AddImageToProblem` for each window image that
   observes it;
3. `ParameterizePoints` then hits `point3D.track.Length() > num_observations`
   (only its in-window observations were counted) and calls
   `SetParameterBlockConstant`.

So it becomes a **fixed 3D anchor that still constrains the window's poses**.
Long, well-observed prior structure is the scaffolding the new image is fitted
against, and in practice it holds the gauge far more robustly than
`FixGauge(THREE_POINTS)` does. Newly triangulated points have no error yet, so
they stay variable. **No change needed here.**

---

## 4. Plan overview

| | |
|---|---|
| P0 | delete `full_solve_interval` |
| P1 | persisted incremental state (`incremental_state.txt`, `incremental_ledger.tsv`) |
| P2 | intrinsics-convergence gate: full solves until converged, windowed after |
| P3 | intrinsics-only global refresh + structure recovery on drift |
| P4 | non-lossy prior import (tripwire, not tolerance) |
| P5 | closed per-add ledger + per-add track completion |
| P6 | remove the dead track-establishment + BA work |
| P7 | server: pass `--optimize_window_size` |

Phase 0 of the original plan — **measure first, it gates everything** — still
applies and is unchanged:

```bash
python scripts/python/incremental_benchmark.py --database_path sessions/camsnap/database.db --image_path sessions/camsnap/images --output_dir /tmp/bench6 --colmap_exe C:/colmap-build/src/colmap/exe/Release/colmap.exe --variants full,window8,window16,window24
```

Record per-add wall time **as a function of N**, final point count, ATE/RPE,
and whether any image ends with zero observations. Expected: `full` grows with
N, `window*` stays flat. If windowed does **not** stay flat, stop and
re-diagnose before changing defaults. Confirm the plan-5 victim (`img_0008`)
stays healthy in every variant.

---

## 5. The intrinsics defect (P2 + P3)

**A windowed solve can never refine camera intrinsics.** There are two
independent freezes, either of which suffices:

1. `AdjustLocalBundle`: intrinsics are set constant when
   `num_images < reg_stats_.num_reg_images_per_camera.at(camera_id)`. With a
   single shared camera and any window smaller than the full model, this always
   fires.
2. `AddPointToProblem` (`bundle_adjustment_ceres.cc`): when a variable point
   has any track element in an image outside the config, that image's camera is
   force-set constant via `config_.SetConstantCamIntrinsics(image.CameraId())`.

Consequence: with one global camera — the normal case for this capture loop —
**whatever intrinsics the model holds when it switches to windowed mode are
frozen forever.** That is precisely the `7756a83c` failure mode: refined
geometry optimized against unrefined intrinsics, 2097 → 111 points.

This behavior is correct in the small. A global parameter cannot be estimated
from a local subproblem. The fix is to stop asking the window to do it, and
give the global parameter its own global pass.

### P2 — Intrinsics-convergence gate

The windowed/full decision becomes automatic and state-driven instead of a
tuned flag:

- while intrinsics are **not converged**, an add runs the **full** path, whose
  global BA refines intrinsics;
- intrinsics are **converged** when every camera's mean focal length changes by
  < `intrinsics_convergence_rel_tol` (default 1e-3) relative between two
  consecutive adds, for 2 consecutive adds, and `num_adds >= 3`;
- once converged, adds run **windowed**.

This also removes N as a correctness knob. With intrinsics correct and long
prior tracks anchoring the frame (§3.1), window size buys a marginally better
local fit, not correctness. Pick 20 and stop thinking about it.

### P3 — Intrinsics-only global refresh + recovery

Drift monitor: if an add's mean reprojection error exceeds
`best_mean_reproj * intrinsics_drift_factor` (default 1.25), then at the end of
that add:

1. **intrinsics-only global BA** — all registered images, every pose and every
   point held constant, only `camera.params` free. This is a ~4–8 DOF problem
   against all observations: cheap regardless of N, because the cost is
   residual evaluation, not the Schur complement;
2. **structure recovery** — `Retriangulate` + `CompleteAllTracks` +
   `MergeAllTracks`, which re-establish observations that were filtered while
   the intrinsics were wrong;
3. reset the convergence state so P2 re-validates.

This is the honest replacement for `full_solve_interval`: a periodic global
pass that refreshes the one thing a window structurally cannot.

---

## 6. Non-lossy prior import (P4)

The import loop in `SolveIncrementalWindowed` has four drop paths, and they are
**not** equally by-design:

| drop | condition | when it can fire |
|---|---|---|
| A | `!ExistsImage(track_el.image_id)` | prior image missing from the database cache — the upstream-corruption signal `LoadPriorPoses` already warns about |
| B | `point2D_idx >= NumPoints2D()` | keypoints changed under the prior — corruption |
| C | `Point2D().HasPoint3D()` | **unreachable** for a consistent prior: import runs into an empty point set, so this needs two prior points claiming one observation |
| D | `track.Length() < 2` | only ever as a *consequence* of A/B/C |

On a healthy prior the import is **already lossless**: zero elements drop and D
never fires. Every real loss is a symptom of prior/database disagreement, and
today it is silently absorbed into a lower point count.

So the fix is not smarter drops, it is refusing to hide them. Count each reason
separately, log each with the offending image names, and `return false` on any
nonzero tally unless `--allow_lossy_prior_import` is passed. A failed add
leaves the last good reconstruction in place; a silent haircut compounds
forever. Expected steady-state tally is exactly zero, so this is a tripwire,
not a tolerance knob — deliberately not a percentage threshold, because a
threshold is where silent decay hides.

---

## 7. `full_solve_interval` is deleted (P0)

It existed so every M-th add ran a full global solve as a "periodic global
refresh". Plan-5 set `skip_global_positioning = true` unconditionally, so those
periodic full solves stopped performing global positioning — the flag's name no
longer described its behavior. Re-enabling GP for periodic solves only measured
6/12 clean seeds versus 12/12 for the skip, so it is not worth reviving.

Delete the field, its CLI registration, its use in the `windowed` condition,
the `windowN.M` benchmark variant syntax, and the doc references. P3 provides
the periodic global refresh that actually matters.

---

## 8. Decay accounting and recovery (P5)

No invented abort threshold. Every add writes a **closed ledger** row:

```
imported / triangulated / merged / completed / filtered_reproj /
filtered_tri_angle / recovered / dropped_at_import(by reason)
```

with the invariant `in == out + accounted losses`. Anything that fails to
balance is a bug and asserts. Rows append to `incremental_ledger.tsv` beside
the model, so "did this session lose structure" is a query over the file rather
than a guess.

**Points lost to filtering are recoverable.** A filtered observation is not
destroyed — the feature match is still in the database — so it can be
re-established once poses and intrinsics improve. `IncrementalTriangulator`
already has the machinery (`CompleteAllTracks`, `MergeAllTracks`,
`Retriangulate`); the windowed path currently only calls the modified-points
variants. Therefore:

- **per add**: run `CompleteTracks` over all points in the window, not just
  modified ones — cheap, and re-attaches observations stripped by an earlier
  bad-intrinsics filter;
- **with the P3 refresh**: run the global `Retriangulate` + `CompleteAllTracks`
  + `MergeAllTracks`. Recovery is exactly the operation that becomes possible
  once the global parameter is corrected, so pairing them is natural.

Recovery then shows up as a positive term in the ledger.

---

## 9. Persisted state (P1)

The server invokes the binary once per add, so convergence tracking needs disk
state. Two files are written next to the model, mirroring the existing
`anchors.txt` pattern exactly (pipeline exposes an accessor, `exe/sfm.cc`
writes it, the server's model-directory move carries it along). The fork has no
JSON dependency, so both are line-oriented text.

- `incremental_state.txt` — `key value` lines: `num_adds`,
  `camera <id> <mean_focal>`, `intrinsics_converged`, `stable_adds`,
  `best_mean_reproj`, `last_refresh_add`
- `incremental_ledger.tsv` — header + one append-only row per add

Missing or unparsable state is treated as "not converged, no history", which
degrades to the full path — the safe direction.

---

## 10. Remove the dead work (P6)

On the incremental path, when global positioning is skipped, `EstablishTracks`
output feeds only bundle adjustment, and `IterativeRetriangulateAndRefine` then
deletes all of it. Gate both off together, beside the existing plan-5 lines in
`IncrementalGlobalPipeline::Run`:

```cpp
mapper_opts.skip_track_establishment = true;
mapper_opts.skip_bundle_adjustment = true;
```

Keep these tied to `skip_global_positioning` — if GP is ever re-enabled here it
needs established tracks.

**This must be measured, not assumed.** BA also refines poses and intrinsics
that persist into retriangulation, so removing it changes retriangulation's
starting point — and under P2 the full path is now specifically the path that
earns intrinsics convergence, which raises the stakes. Compare against the
plan-5 acceptance run: `img_0008` at offset ≈ 8.38 with ≈ 156 observations, no
zero-observation images, stable across all 12 seeds. If pose or intrinsics
quality regresses, fall back to keeping BA but triangulating the established
tracks from the known poses first, so BA starts from real structure instead of
the origin.

Expected saving if it holds: ~0.44 s of ~1.42 s per add, and the removal of a
degenerate-configuration hazard.

---

## 11. Server (P7)

Add `--optimize_window_size 20` to `run_incremental_global_mapper` in
`app/pipeline/colmap_runner.py`. The fork default stays `0` so the cold
`global_mapper` path and any other caller are provably untouched; the flag at
the call site is self-documenting. With P2 gating the actual windowed/full
decision, this value is a performance knob only.

---

## 11a. Measured results (2026-08-27, `sessions/camsnap`, 13 images)

Binary built from this branch; drip-fed from 5 images, 8 adds.

| variant | 3D points | mean reproj px | ATE rmse | RPE rot deg | mean s/add | total s |
|---|---|---|---|---|---|---|
| one-shot `global_mapper` | 2037 | 0.001 | 0 | 0 | — | 1.7 |
| drip-feed full | 2014 | 0.001 | 4.61 | 10.39 | 0.95 | 7.6 |
| drip-feed window20 | 1975 | 0.923 | 3.90 | 4.48 | **0.49** | **3.9** |

Windowed is ~2x faster per add and *more* accurate (ATE 3.90 vs 4.61, RPE
rotation 4.48 vs 10.39 deg).

**Read the reprojection-error column carefully.** The full path's 0.001 px is
not a quality result, it is an artifact: it deletes all structure and
re-triangulates from the current poses every add, so the points are fitted to
the poses and the residual is near-zero by construction. The windowed path's
0.92 px is an ordinary, honest SfM error over structure that has been carried
forward. The two numbers are not comparable, which is why the drift monitor
tracks them separately (sec. 9).

Ledger for the window20 run, showing the gate handing over at add 4:

```
add path      prior  imported  drops  triang  merged  completed  recovered  out   reproj    track_len  refreshed
1   full        760         0      0       0       0          0          0   918  0.001076   3.85      0
2   full        918         0      0       0       0          0          0  1048  0.001004   4.27      0
3   full       1048         0      0       0       0          0          0  1237  0.000961   4.20      0
4   windowed   1237      1237      0     320      38       1569          0  1390  0.560892   5.10      0
5   windowed   1390      1390      0     960     585       2408         13  1652  0.960594   5.39      1
6   windowed   1652      1652      0     793     202       2383          0  1885  0.943179   5.32      0
7   windowed   1885      1885      0     237      72       2117          0  1975  0.927478   5.18      0
8   windowed   1975      1975      0     134       0        442          0  1975  0.927444   5.18      0
```

- **Import is lossless on every add**: `imported == prior_points`, all four
  drop columns zero. P4's tripwire never fired, which is the expected
  steady state.
- **No structure decay**: points rise 1237 -> 1975 monotonically, mean track
  length holds at 5.1-5.4 (vs 4.0-4.5 on the full path). Criterion 4 holds.
- **Reprojection error plateaus** (0.56 -> 0.96 -> 0.94 -> 0.93 -> 0.927)
  rather than running away. One refresh fired at add 5 and none after.

### What the measurements changed about the design

1. **P6 is load-bearing, not an optimization.** With track establishment and
   BA re-enabled, the model does not merely waste 0.44 s -- it **collapses to
   0 points**, with focal length deviating by 20-84% per add (measured: 0.61,
   0.20, 0.81, 0.84, 0.77, 0.69, 0.53) and RPE rotation at 66.7 deg. BA over
   origin-initialized tracks actively destroys the intrinsics. Sec. 2(b)
   understated this.
2. **Convergence must be a stationarity test, not a decay test.** The focal
   estimate never settles: it jitters in a 3e-4..3e-3 relative band forever,
   because each full solve rebuilds all structure and re-lands slightly
   differently. A consecutive-change test against 1e-3 never fires -- measured
   directly, it oscillated stable/unstable indefinitely and the gate never
   opened. The test now compares against a running average with a 5e-3
   tolerance chosen to sit above the measured noise band.
3. **The drift monitor must not reset the convergence state.** Doing so
   forced the next add back onto the full path, producing a full/windowed
   oscillation that reintroduced exactly the O(N) rebuild this plan exists to
   remove. If intrinsics genuinely move, the stationarity test detects it
   directly; the refresh itself is cheap and its recovery pass is pure gain.
4. **The drift baseline must be a running average, not the best-ever value.**
   The first windowed add inherits pristine full-path structure and is
   systematically the best one, so a min-ever baseline fires on every
   subsequent add.

### Open items

- **Criterion 1 (flat in N) is NOT yet verified.** 8 adds over 13 images is
  far too short a lever arm to separate O(window) from O(N); the measured
  per-add times are dominated by fixed startup cost. This needs a
  substantially larger dataset -- and one with gravity priors, since
  `bootstrap_max_gravity_error_deg` gates the bootstrap and `camsnap` carries
  a `_gravity.json` per image that public datasets do not.
- **Criterion 3 (no zero-observation images) fails, pre-existing.**
  `img_0012` ends at zero observations in the one-shot reference too, so that
  image is bad data rather than a pipeline fault. But `img_0008` has 168
  observations one-shot and **0** in both drip-fed paths, and `img_0011` drops
  from 342 to 5 (full) / 20 (windowed). This affects the full and windowed
  paths equally -- windowed is slightly better -- so it is not caused by
  anything in this plan, but it is unresolved and it means the plan-5
  acceptance figure (`img_0008` at ~156 observations) does not currently
  reproduce. Worth its own investigation.

---

## 11b. PnP registration and the upright solver (added 2026-08-27)

The bootstrap averages one rotation candidate per pose-graph edge, which is
the wrong operator for a weakly-connected image: `img_0008` has nine edges, one
carrying 456 inliers, one 171, and seven carrying 16-20 (noise). Averaging let
the seven outvote the two. Replacing it with PnP against the prior structure
(`IncrementalMapper::RegisterNextImage`, absolute pose + RANSAC, with the
bootstrap kept only as an optional fallback) was the single largest quality
change in this plan:

| | ATE rmse | RPE rot | max rotation error vs one-shot |
|---|---|---|---|
| bootstrap (before) | 4.61 | 10.39 deg | **74.84 deg** |
| PnP (after) | 0.0125 | 0.064 deg | **0.38 deg** |

The bad bootstrap poses were poisoning the *whole* model, not only the images
that ended with zero observations. Every registered image is now within 0.38
deg of the one-shot reference.

**PnP's ability to decline is the point.** It accepted 6 of 8 images and
declined exactly the two that the bootstrap had been registering with garbage
poses and zero observations. Declining leaves the image unregistered and
retried on every later add (`RegisterNewImagesByPnP` sweeps all poseless
images), which is strictly better than a pose that can never recover -- an
image with zero observations gives bundle adjustment no residuals to pull it
back. So `bootstrap_fallback_when_pnp_declines` defaults to false, and an add
that registers nothing writes the prior model through unchanged (ledger path
`skipped`) rather than failing.

### The upright (up2p) solver

`Up2PEstimator` implements Kukelova et al.'s 2-point solver with known vertical
direction, plugged into the existing `LORANSAC<Up2PEstimator, EPNPEstimator>`
so local optimization and Ceres refinement are unchanged. Gravity comes from
the `pose_priors` table the server already populates. Verified by
`AbsolutePose.Up2PExactRecovery`: exact recovery to < 1e-6 over ~770 random
gravity-consistent configurations.

**It did not rescue the declined images, and the measurements say why.** The
speed argument for up2p does not apply here. Measured decline reasons:

| image | correspondences | inliers | ratio |
|---|---|---|---|
| `img_0008` | 35 / 50 / 64 | 9 / 14 / 18 | 0.26 / 0.28 / 0.28 |
| `img_0012` | 40 | 17 | 0.43 |

At a 0.28 inlier ratio P3P needs ~519 draws against a 10,000 budget -- RANSAC
was finding consensus fine. The constraint is the absolute correspondence
count, which no solver can increase. The follow-up conditioning hypothesis
(4 unknowns instead of 6 makes few inliers more informative) was also tested
and refuted: up2p put `img_0008` at 25.66 deg versus P3P's 28.78 deg at the
same threshold. These two images are genuinely unregisterable against this
structure.

### The dangerous knob

`gravity_uncertainty_deg` widens the RANSAC inlier threshold by
`focal * tan(deg)` to absorb the bias an imperfect gravity reading induces.
That term is much larger than it looks: at a 1000 px focal length, 1.5 deg adds
**26 px** on top of the default 12 px `max_error`, tripling the threshold.

Measured at `pnp_min_num_inliers = 30`:

| gravity_uncertainty_deg | images | max rotation error |
|---|---|---|
| 1.5 | 12 | **26.99 deg** (registered a bad pose P3P declined) |
| 0.5 | 11 | 0.35 deg |
| 0.0 | 11 | 0.38 deg |

So the widening, not the solver, caused the regression. **It defaults to 0.**
Raise it only against a measured gravity accuracy, and remember it scales with
focal length.

### Lowering the inlier threshold does not help either

Relaxing `pnp_min_num_inliers` from 30 to 15 registers both hard images but
with 28.78 deg and 38.30 deg rotation errors, and degrades `img_0011` from
0.19 to 0.79 deg. `img_0012` accumulates 301 observations while being
geometrically wrong -- worse than the zero-observation case, because it looks
healthy and pollutes its neighbours. The default of 30 is load-bearing.

---

## 12. Acceptance

1. Windowed per-add wall time **flat in N** while `full` grows.
2. Windowed ATE/RPE against the one-shot `global_mapper` reference no worse
   than the full path's.
3. No image in any add ends with zero 3D observations; `img_0008` stays at
   offset ≈ 8.4 with ≈ 160 observations.
4. Point count and mean track length do not decline monotonically across a
   full drip-fed session; the ledger balances on every add.
5. P6's stage removal does not regress the plan-5 12-seed acceptance run.
6. Cold `global_mapper` behavior unchanged — none of this may touch the
   from-scratch path.
7. A session that starts from an uncalibrated database converges intrinsics
   within a few adds and then stays windowed, and an induced intrinsics
   perturbation triggers exactly one P3 refresh that recovers the lost points.

---

## 13. Notes for whoever picks this up

- **Do not trust a single passing run.** Plan-5's failure was a PRNG lottery
  where 4 of 12 seeds passed; one green run proved nothing. The current fix is
  seed-stable, so a *variance* across seeds in any new variant is itself a bug
  signal.
- **The full test suite does not currently build on this machine** with
  `-DTESTS_ENABLED=ON`: 8 pre-existing `M_PI` undeclared errors in
  `src/colmap/feature/sift_test.cc` and `src/colmap/sfm/rotation_utils_test.cc`
  (MSVC needs `_USE_MATH_DEFINES`). Unrelated to plan-5 or plan-6, but it means
  "the suite passes" cannot currently be asserted. Worth fixing first so
  acceptance criterion 6 is checkable; it is likely a two-line include change.
- `incremental_global_pipeline_test.cc` already exercises
  `optimize_window_size = 4` — a working reference for the windowed path.
- `sessions/camsnap` in the server repo is the reproducer data; it is
  regenerable with `scripts/rerun_pipeline.py`. Its `priors/after_*` snapshots
  are **not** usable as single-add benchmarks (see §2).
