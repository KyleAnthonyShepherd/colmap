# Plan 5 — Incremental global mapper throws away prior camera positions

Agent plan for the COLMAP fork at `W:\Kyle_Shepherd\photogrammetry_app\colmap`,
branch `incremental-glomap` (at `0c3a579e` when this was written). Nothing in
this plan changes `home-server-app-updated`; the server-side guard is tracked
separately (see §7).

Goal: stop `incremental_global_mapper` from randomly discarding a good prior
reconstruction's camera positions on every incremental add, which currently
destroys a weakly-connected image roughly two times in three.

---

## 1. What is wrong, in one paragraph

Every incremental add runs a **full global solve**, and global positioning
begins by throwing away all camera positions — including the 12 already-solved
ones from the prior reconstruction — and replacing them with uniform random
points in a 200×200×200 cube. Well-connected cameras reconverge. A
weakly-connected one may not: it is left far outside the scene, the reprojection
filters then strip every one of its observations because they no longer agree
with that position, and once it has zero observations bundle adjustment has no
residuals for it and can never pull it back. The image is written to `sparse/0`
with a garbage pose. Whether this happens is decided by the PRNG draw.

---

## 2. Evidence

Measured on `sessions/camsnap` in the server repo (13 images, scene extent
≈ 26 units), the reconstruction of `sessions/30d37bfd` re-run through the real
pipeline. The victim is `img_0008_20260429_185440.jpg`, the session's
weakest-linked image: 9 match partners but only 749 verified inliers, and just
**two** CALIBRATED two-view edges (171 and 456 matches, to `img_0006` and
`img_0007`); its other edges are 16–18 matches and UNCALIBRATED.

**It is not a bootstrap failure.** `img_0008` registers correctly and stays
correct for three further adds, then dies on the fourth:

| state after adding | `img_0008` centre | offset from median camera | its 3D observations |
|---|---|---|---|
| `img_0008` | (-8.23, -9.12, -14.14) | 13.81 | 169 |
| `img_0009` | (-8.15, -9.08, -14.05) | 10.27 | 161 |
| `img_0010` | (-8.09, -9.03, -14.03) | 8.01 | 173 |
| `img_0011` | (-8.16, -9.04, -14.05) | 8.30 | 162 |
| `img_0012` | **(-608.62, 1248.15, 2485.29)** | **2859.18** | **0** |

**It is not a track-establishment failure.** Reimplementing
`GlobalMapper::EstablishTracks` in Python reproduces COLMAP's log exactly
(4066 tracks from 17238 observations, 1436 kept, 265 discarded as inconsistent)
and shows `img_0008` participating in **185** kept tracks going into the solve.
It has plenty of evidence; the solve discards it.

**It is global positioning.** Bisecting the stages on the single failing add:

| variant | `img_0008` offset | its observations |
|---|---|---|
| defaults | 2852.76 | 0 |
| `--GlobalMapper.skip_global_positioning 1` | **8.38** | **156** |
| `--GlobalMapper.max_angular_reproj_error_deg 10` | 1170.53 | 0 |
| `--GlobalMapper.max_normalized_reproj_error 0.5` | 2869.70 | 0 |
| `--GlobalMapper.skip_bundle_adjustment 1` | 9413.35 | 0 |
| `--GlobalMapper.skip_retriangulation 1` | 3111.80 | 0 |

Only skipping global positioning helps. Loosening the filters does not — they
are the mechanism that removes the evidence, not the cause of the bad position.

**It is a PRNG lottery.** Same command, same input, varying only
`--GlobalMapper.random_seed`:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| offset | 5446 | 42 | **8.5** | **8.5** | 9432 | 284 | **8.5** | 51 | 221 | 1102 | 12948 | **8.5** |
| obs | 0 | 0 | 162 | 163 | 0 | 0 | 161 | 0 | 0 | 0 | 1 | 164 |

**4 of 12 seeds succeed.** When it succeeds it lands on the same correct basin
every time (offset 8.46, ~162 observations). There is a right answer and the
solver finds it only about a third of the time.

**The data supports the correct answer easily.** The server's own live
capture-quality path (plan-4 Phase C) localizes this exact frame against a
descriptor cloud built from the *other twelve* images and puts it at
(-2.32, -7.29, -14.23) — 1.4 units from `img_0007`, squarely on the capture
trajectory — from only 33 correspondences and 11 PnP inliers. So a plain
gravity-constrained PnP with a fraction of the evidence gets it right while
global positioning, with 185 tracks available, does not.

---

## 3. Root cause, with references

`src/colmap/estimators/global_positioning.cc:98` —
`GlobalPositioner::InitializeRandomPositions`:

```cpp
if (options_.generate_random_positions && options_.optimize_positions) {
  frame_centers_[frame_id] = 100.0 * RandVector3d(-1, 1);
} else {
  frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
}
```

`generate_random_positions` defaults to `true`
(`src/colmap/estimators/global_positioning.h:14`), and
`IncrementalGlobalPipeline::Run` never turns it off — it only sets
`skip_rotation_averaging = true`
(`src/colmap/controllers/incremental_global_pipeline.cc:268`). So the
`else` branch, which would reuse the pose already in the reconstruction, is
dead on this path.

Random initialization is the right default for a **cold** global solve, where
there is nothing better. It is wrong for an **incremental add**, whose entire
premise is that a good prior reconstruction already exists. The prior positions
are an excellent initialization and are discarded.

Two secondary problems found on the way:

- **`random_seed = -1` does not do what its comment says.**
  `global_positioning.h:35` documents "If -1 (default), uses non-deterministic
  `random_device` seeding", but the constructor
  (`global_positioning.cc:22`) only calls `SetPRNGSeed` when
  `random_seed >= 0`. With the default, the PRNG is never seeded, so every
  fresh process replays the same draw. That is why this failure reproduces
  identically run after run rather than appearing intermittent. Note the fix
  direction: making it *genuinely* non-deterministic would be worse — it would
  turn a reproducible failure into a flaky one.
- **The `generate_*` flags are not exposed on the CLI.** `option_manager.cc`
  registers `gp_use_gpu`, `gp_gpu_index`, `gp_optimize_positions`,
  `gp_optimize_points`, `gp_optimize_scales`, `gp_loss_function_scale`,
  `gp_max_num_iterations` — but none of `generate_random_positions`,
  `generate_random_points`, `generate_scales`. They cannot be tested without
  recompiling, which is why this took a rebuild to pin down.

---

## 4. The reproducer

Minimal, ~2 seconds per run, no re-capture needed. `sessions/camsnap` in the
server repo already contains the per-add prior snapshots (delete it when done;
it is regenerable with `scripts/rerun_pipeline.py`).

```bash
colmap incremental_global_mapper \
  --database_path  sessions/camsnap/database.db \
  --image_path     sessions/camsnap/images \
  --prior_reconstruction_path sessions/camsnap/priors/after_img_0011_20260429_185443 \
  --output_path    /tmp/replay \
  --log_target stderr --log_severity 0
```

Then check `img_0008`'s centre and observation count in `/tmp/replay/0`.
Failing: offset ≈ 2853, 0 observations. Passing: offset ≈ 8.5, ≈ 162
observations, and **no** image in the output has zero observations.

If `sessions/camsnap` has been deleted, regenerate it with the snapshotting
driver described in the plan-4 CHANGELOG (`scripts/rerun_pipeline.py` plus a
`_run_incremental_mapping` wrapper that copies `sparse/0` before each add).

---

## 5. The fix

### Phase A — make it testable (do this first)

Register the three missing options in
`src/colmap/controllers/option_manager.cc`, next to the existing `gp_*` block
(~line 719):

```cpp
AddDefaultOption("GlobalMapper.gp_generate_random_positions",
                 &global_mapper->mapper.global_positioning.generate_random_positions);
AddDefaultOption("GlobalMapper.gp_generate_random_points",
                 &global_mapper->mapper.global_positioning.generate_random_points);
AddDefaultOption("GlobalMapper.gp_generate_scales",
                 &global_mapper->mapper.global_positioning.generate_scales);
```

This touches one `.cc` and rebuilds fast. Verify against the reproducer with
`--GlobalMapper.gp_generate_random_positions 0`. **Expect it to pass.** If it
does not, stop and re-diagnose before writing any more code — the rest of this
plan rests on that result.

### Phase B — use the prior positions on the incremental path

In `IncrementalGlobalPipeline::Run`, in the incremental-add branch beside the
existing `skip_rotation_averaging` line (`incremental_global_pipeline.cc:268`):

```cpp
// The prior reconstruction's camera centres are a far better initialization
// than a random point in a 200^3 cube, and discarding them is what loses
// weakly-connected images (see plan-5).
mapper_opts.global_positioning.generate_random_positions = false;
```

Leave `generate_random_points` alone unless Phase A shows it matters. Newly
established tracks have no 3D position yet, so seeding points from the
reconstruction risks initializing them all at the origin — check
`AddPoint3DToProblem` before touching it.

### Phase C — guard the non-posed frames

`InitializeRandomPositions` will now call `frame.RigFromWorld()` on every
constrained frame. Confirm every such frame actually has a pose; if not, fall
back to random for that frame only:

```cpp
const bool use_prior = !options_.generate_random_positions && frame.HasPose();
frame_centers_[frame_id] = use_prior
    ? frame.RigFromWorld().TgtOriginInSrc()
    : 100.0 * RandVector3d(-1, 1);
```

`Frame::HasPose()` is at `src/colmap/scene/frame.h:94`. On the incremental path
all 12 prior frames plus the bootstrapped one have poses, so this is
belt-and-braces — but the `else` branch is currently dead code and has never
run, so do not assume it is correct as written.

### Phase D — do not silently emit a camera with zero observations

Independent of the initialization fix, an image that ends a solve with **zero**
3D observations is unconstrained and its pose is meaningless. Today it is
written to `sparse/0` as if it were a normal registered image. After the solve
in `IncrementalGlobalPipeline::Run`, before writing output, either
de-register such images or fail the add loudly. Two of the three failure modes
observed (offset 42 and offset 51 at seeds 1 and 7) are only detectable this
way — they are too small to trip a distance threshold but still have zero
observations.

### Phase E — a fallback that is known to work

When an image fails to keep observations through the solve, fall back to
absolute-pose registration against the prior reconstruction's 3D points, which
is exactly what plan-4's live path does successfully for this frame (§2). The
fork already has the machinery in the incremental mapper's
`RegisterNextImage`. Optional, but it turns a hard failure into a graceful one.

### Phase F — fix the seed documentation

Either implement the documented `random_device` seeding or correct the comment
at `global_positioning.h:35`. Prefer correcting the comment and making `-1`
mean an explicit fixed default seed: reproducible failures are much cheaper to
debug than intermittent ones, and after Phase B the initialization should not
be load-bearing anyway.

---

## 6. Acceptance

1. The §4 reproducer passes: `img_0008` at offset ≈ 8.5 with ≈ 162
   observations, no zero-observation images in the output.
2. It passes for **all** of seeds 0–10 and 42, not just the four that pass
   today. This is the real bar — the current failure is a lottery, so a single
   passing run proves nothing.
3. A full 13-image re-capture through the server pipeline
   (`scripts/rerun_pipeline.py`) puts every camera within a few units of the
   median camera centre, and
   `scripts/live_replay.py sessions/<new> --mode track` reaches **13/13**
   (it is 12/13 today, held back by exactly this bug).
4. The fork's own test suite still passes (`-DTESTS_ENABLED=ON`).
5. From-scratch `global_mapper` behaviour is unchanged — Phase B must only
   touch the incremental path. Check a cold `global_mapper` run on the same
   database still produces a comparable reconstruction.

---

## 7. Notes for whoever picks this up

- **Build recipe** for a CLI-only COLMAP on this machine is in the plan-4
  CHANGELOG (vcpkg pinned to the fork's own baseline, `GUI/CUDA/CGAL=OFF`,
  `IPO_ENABLED=OFF`, target `colmap_main`, ~50 min cold, fast incrementally).
  A binary already exists at
  `C:\colmap-build\src\colmap\exe\Release\colmap.exe`.
- Rebuilding after touching `global_positioning.h` recompiles a lot, since it
  is pulled in via `global_mapper.h`. Phase A deliberately avoids that.
- This is **not** the same issue as the one-camera-per-session bug fixed in
  the server repo on 2026-08-26. That one is fixed and verified; this one is
  upstream of it and independent. Sharing one camera did reduce the blast
  radius (the victim's focal can no longer collapse to 122.9 px alongside its
  position), but it does not prevent the displacement.
- A complementary **server-side** guard — reject or flag a promoted
  reconstruction containing a camera far outside the scene — is tracked as its
  own task in the server repo. Worth having regardless: it protects the MEGS-2
  trainer ingest from a bad pose whatever the mapper does.
- Do not "fix" this by loosening `max_angular_reproj_error_deg` or
  `max_normalized_reproj_error`. The measurements in §2 show it does not work,
  and those filters are doing their job — the position they are filtering
  against is the thing that is wrong.
