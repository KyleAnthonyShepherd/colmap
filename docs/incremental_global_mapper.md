# `incremental_global_mapper`

Adds newly matched images to an already-solved sparse reconstruction without
re-running the full global SfM pipeline. Designed for a capture loop where a
server receives one photo at a time, matches it into a COLMAP database, and
wants the sparse model updated within seconds.

**Pinned base:** this feature branch is based on upstream COLMAP commit
`5b76f53` (the 4.0.4 changelog commit). Do not rebase without re-running the
full test suite; `PoseGraph::Edge::cam2_from_cam1` is guarded by a
`static_assert` that will fail loudly if the upstream type drifts to
`std::optional<Rigid3d>`.

## CLI

```
colmap incremental_global_mapper \
    --database_path session/database.db \
    --image_path session/images \
    --prior_reconstruction_path session/sparse/0 \
    --output_path session/sparse/0_incremental \
    [--optimize_window_size 20] \
    [--full_solve_interval 10] \
    [--bootstrap_min_inliers 10] \
    [--bootstrap_max_candidate_deg 10.0] \
    [--bootstrap_max_gravity_error_deg 15.0] \
    [--realign_to_prior 1] \
    [--use_prior_position 1] \
    [--use_robust_loss_on_prior_position 1] \
    [--prior_position_loss_scale 7.815] \
    [--prior_position_fallback_stddev 1.0] \
    [--prior_position_max_error_sigma 5.0] \
    [--prior_position_max_error_m 0.1] \
    [--overwrite_priors_covariance 0] \
    [--prior_position_std_x 1.0 --prior_position_std_y 1.0 \
     --prior_position_std_z 1.0]
```

The database must already contain features and verified matches for the new
image(s) (and decomposed relative poses — run `colmap view_graph_calibrator`
first, as the production server does). The prior reconstruction is read from
`--prior_reconstruction_path`; the updated reconstruction is written under
`--output_path` (numbered subdirectory `0`). With an empty
`--prior_reconstruction_path` the command falls back to a standard one-shot
global reconstruction.

## How it works

1. **Seed** — `GlobalMapper::LoadPriorPoses` copies every registered prior
   pose into the fresh reconstruction (matched by image id) and records the
   prior image set. Prior images missing from the database are reported with
   a warning (a symptom of upstream database corruption).
2. **Bootstrap** — `GlobalMapper::BootstrapNewImagePoses` walks the pose
   graph for edges connecting unregistered images to prior images. Each edge
   with at least `bootstrap_min_inliers` verified matches contributes one
   absolute-rotation candidate and one translation ray. The pure solver
   (`SolvePoseFromPriorEdges`) combines them:
   - rotation: weighted Karcher mean on SO(3) with an outlier re-pass that
     discards candidates further than `bootstrap_max_candidate_deg` from the
     mean;
   - gravity gate: when the new image has a gravity prior in the database's
     `pose_priors` table and the prior images define a world gravity
     direction, candidates whose implied gravity disagrees with the measured
     one by more than `bootstrap_max_gravity_error_deg` are rejected first
     (a stronger outlier test than distance-from-mean when only 2–3
     candidates exist; `<= 0` disables). If every candidate fails the gate,
     solving proceeds ungated with a warning.
   - translation: weighted least-squares intersection of the rays, with a
     prior-centre-centroid fallback for near-singular (single-ray/colinear)
     systems. A cheirality check warns when the resulting pose sees almost
     none of the prior 3D points in front of the camera.
3. **Solve** — one of two paths:
   - **Windowed** (`--optimize_window_size N > 0`, recommended 15–25 for
     capture loops): prior 3D points are imported with their 2D–3D links,
     only the new image is triangulated against them, and iterative local
     bundle adjustment optimizes the new image plus its `N` most covisible
     images. All other poses participate only as constant-pose residuals,
     so per-add cost is ~O(window) instead of O(total images), and the
     output stays in the prior coordinate frame by construction — no
     realignment, no scale drift. With `--full_solve_interval M`, every
     M-th image (by registered count) runs the full path below instead, as
     a periodic drift-correcting refresh.
   - **Full** (`--optimize_window_size 0`, previous behaviour): track
     establishment from scratch, global positioning, iterative bundle
     adjustment, and retriangulation over all frames, with rotation
     averaging skipped in favour of the bootstrapped rotations.

## Position priors (`--use_prior_position`)

Off by default. When on, the database's `pose_priors` table stops being read
only for pair selection and becomes a constraint on the solve.

Why it matters: a priorless model has no scale of its own, so a wrong Sim3 and
a wrong model are indistinguishable. Measured on `sessions/test1-2` (69 images,
identical database), the priorless `mapper` produced a Sim3 scale of 1.7037
against `pose_prior_mapper`'s 1.0053, a camera-height IQR of 0.075 m against
0.036 m, and a worst GNSS/SfM step ratio of 4.17 against 1.47. With priors the
model is metric *before* georeferencing runs, and the Sim3 becomes a check
rather than a correction.

Three things happen when the flag is set:

1. **The priors are moved into the reconstruction's frame**, not the other way
   round (`sfm/prior_positions.h`). A Sim3 is fitted robustly from the
   registered cameras' solved centres to their priors and inverted; the priors
   and their covariances are transformed by it. Transforming the *model*, as
   the cold `pose_prior_mapper` path does, would move every previously solved
   camera out of the prior reconstruction's gauge — which an incremental add
   must never do. If the fit is impossible (fewer than 3 registered images
   carry a prior) or fails, the add runs unconstrained and says so.

2. **Registration is gated.** After PnP + refinement and *before* anything is
   committed, a candidate pose whose camera centre disagrees with its prior by
   more than both `--prior_position_max_error_sigma` sigmas (Mahalanobis,
   under the prior's own covariance) and `--prior_position_max_error_m` metres
   is declined. Both conditions together: a 1.6 cm vertical sigma would
   otherwise reject on ordinary SfM noise, and a metric-only test has no idea
   how good the fix is. A declined image is retried on the next add, which is
   strictly better than a confident wrong pose — at `img_0059` the priorless
   solve moved the camera 1.238 m where the operator had moved 0.297 m (~8σ),
   and every later frame inherited the displacement.

3. **Bundle adjustment gets a position residual.** On the windowed path the
   residual is added to the images the covisibility window leaves free, with
   alignment and normalization disabled so the gauge stays exactly where the
   constant poses put it. On the full path the existing pose-prior bundle
   adjuster runs (the same one `pose_prior_mapper` uses), which aligns the
   model to the priors and leaves it metric; `Reconstruction::Normalize()` is
   skipped throughout, since it would rescale away the metric scale the priors
   just established. For the same reason, realignment to `anchors.txt` is
   skipped when priors are active: the priors are the gauge.

Covariance is **read from the database** by default. The site pipeline writes
real per-frame covariance, and the 1 m default would throw away a 1.6 cm
vertical measurement — so `--prior_position_fallback_stddev` is a last resort,
and `--overwrite_priors_covariance` (which rewrites the covariance of every
prior in the database) is for datasets that have none.

Only `CARTESIAN` (`coordinate_system = 1`) priors are used. `WGS84` priors are
converted to Cartesian ENU on load when the flag is set; anything still
`UNDEFINED` after that has no metric meaning and is dropped with a warning
rather than mixed in.

## Prior residual report (`prior_residuals.tsv`)

Written beside the model on every add that uses position priors, one row per
registered image with a prior:

```
image_id  name  prior_x/y/z  solved_x/y/z  residual_x/y/z  residual_norm
sigma_x/y/z  mahalanobis  robust_weight  used_in_ba
```

`sigma_*` is what the cost function was actually weighted by (the covariance
diagonal in world frame, after the frame fit). `robust_weight` is the weight
the Cauchy loss gave the residual, 1 when the loss is trivial. `used_in_ba` is
1 when this image's pose was free *and* constrained by its prior in this add's
bundle adjustment — the prior pulled on it, rather than merely existing.

The point is that a stuck or displaced frame is identifiable from this file
alone. Everything in it is already known to bundle adjustment; before, finding
`img_0059` meant reading `residuals.csv`, re-deriving `R_enu_from_cam` per
frame from the model and the Sim3, rotating residuals into each camera's own
frame, and comparing consecutive steps.

The ledger (`incremental_ledger.tsv`) carries the same signal per add in five
new columns — `prior_residuals`, `prior_rms`, `prior_worst`,
`prior_worst_sigma`, `prior_worst_image` — so it is visible *during* the walk.
A ledger written by an older build is migrated on carry-forward: the header is
widened and old rows are padded with empty fields (empty, not zero, because
the measurement did not exist).

## The antenna as a rig sensor

A GNSS antenna is not at the camera centre. Turning an antenna position into a
camera-centre prior needs the camera's full orientation including yaw, which
with no magnetometer only exists after the model is solved — hence the
three-fidelity prior ladder, the two-pass georeferencing and the stripe
correlation diagnostic the site repo carries.

The antenna is, however, exactly a second sensor rigidly mounted to the
camera, and this build has rigs. A pose prior whose `corr_sensor_type` names a
non-camera sensor (`SensorType::GNSS = 2`) that belongs to the frame's rig now
gets its residual attached through that sensor:

```
residual = P_antenna_measured - (C_cam + R_world_from_cam * l_sensor)
```

evaluated by `AbsoluteRigPosePositionPriorCostFunctor` over
(`sensor_from_rig`, `rig_from_world`). Under
`--Mapper.ba_refine_sensor_from_rig` the `sensor_from_rig` translation is a
free parameter, so **bundle adjustment estimates the lever arm** instead of
requiring it to be measured — and keeps estimating it, for free, as the mount
changes. Robust alignment (`AlignReconstructionToPosePriors`) resolves such
priors through the frame as well, so a cold solve can use antenna positions
directly.

To use it, the caller declares the antenna in the database: a rig sensor of
type `GNSS` with a `sensor_from_rig` transform (the measured lever arm, e.g.
`l = [0, 0.1524, 0]`, as the starting value), and pose priors whose
`corr_sensor_type` is that type. A sensor with no `sensor_from_rig` is
rejected with a warning: bundle adjustment can refine a lever arm, not invent
one.

Verify against the recovered `sensor_from_rig` translation, which is a direct
measurement — *not* against the stripe correlation. On `test1-2` the stripe did
not collapse under `pose_prior_mapper` (0.674 → 0.744) and a distributed ~7 cm
camera-frame systematic survives on both solvers, with the lever arm already
eliminated as its cause on physical grounds. A rig term may fit the arm
correctly and still leave that 7 cm behind.

## Gauge anchors (`anchors.txt`)

Full solves leave the output in an arbitrary similarity frame, so it must be
realigned to the prior. Realigning to the previous output each time
("rolling" Sim3) compounds scale error over hundreds of photos. Instead:

- The first successful reconstruction selects up to 4 well-conditioned
  anchor images (many 3D points, mutually wide baselines) and the CLI
  writes `anchors.txt` (`image_id cx cy cz` per line) next to
  `cameras.bin`. Because the file lives inside the model directory it
  survives the server's `0_incremental/0 → sparse/0` promotion.
- Later runs read `prior_reconstruction_path/anchors.txt`. Full solves
  realign onto the **fixed** anchor centres; the rolling Sim3 over all
  prior centres remains only as a fallback when fewer than 3 anchors are
  registered. Windowed solves additionally keep anchor poses constant even
  when anchors fall inside the covisibility window.
- The anchor set propagates unchanged across adds; it is the permanent
  session reference frame.

Deleting `anchors.txt` is safe: the next run falls back to rolling
realignment and re-selects anchors.

## Benchmark

`scripts/python/incremental_benchmark.py` replays a session database through
the drip-feed loop (per-step subset databases mimic the production server's
growing database), compares against a one-shot `global_mapper` reference,
and emits a markdown table with ATE/RPE, point counts, mean reprojection
error, and per-add wall times:

```
python3 scripts/python/incremental_benchmark.py \
    --database_path session/database.db \
    --output_dir /tmp/bench \
    --colmap_exe build/src/colmap/exe/colmap \
    --variants full,window8,window20.10 \
    --start_images 5
```

`full` = drip-feed with full solves; `windowN` = windowed with window N;
`windowN.M` = windowed with a full solve every M images. Requires numpy
only (standalone binary model readers, no pycolmap).

Reference numbers from a real 13-image phone session (ALIKED+LightGlue,
CPU-only container, drip-feed from 5 images; ATE/RPE vs the one-shot
model after Sim3 alignment):

| variant | 3D points | mean reproj px | RPE rot deg | mean s/add | total s |
|---|---|---|---|---|---|
| one-shot global_mapper | 2067 | 0.001 | — | — | 2.5 |
| drip-feed full | 2074 | 0.001 | 11.7 | 1.73 | 13.9 |
| drip-feed window8 | 1850 | 1.055 | 3.6 | 0.32 | 2.5 |
| drip-feed window8.4 | 2068 | 0.379 | 0.24 | 0.68 | 5.4 |

Two caveats: per-add wall time for the full variant grows with N even at
this size (1.3 s → 2.5 s across 8 adds) while the windowed variant stays
flat (~0.3 s); and solves are nondeterministic without a fixed seed, so
single runs on small sessions vary — the full drip-feed scored ATE 0.06 /
RPE 0.12° on one run and ATE 3.5 / RPE 11.7° on another (a single bad
small-subset solve poisons the rolling chain; exactly the failure mode the
anchor mechanism and periodic-refresh configuration are designed to
bound). `windowN.M` was the best and most stable configuration measured.
Re-run on target hardware at N = 50/150/300 for real curves.

## Server integration contract (spacemapper)

- The server must keep calling `view_graph_calibrator` before each add.
- `bootstrap_min_inliers` (default 10) is the C++-side floor an accepted
  image must clear; the server's go/no-go gate should be at least as
  strict so accepted uploads cannot fail bootstrap.
- Gravity priors written to `pose_priors` (camera-frame "direction gravity
  pulls", COLMAP 4.x schema) are consumed by the gravity gate
  automatically; no new flags needed.
- To use GNSS on the live path, add `--use_prior_position 1` to the
  `incremental_global_mapper` invocation. The server already writes
  `pose_priors` rows with real per-frame covariance and
  `coordinate_system = 1` (CARTESIAN ENU metres), which is exactly what this
  consumes — no schema change, and **do not** pass
  `--overwrite_priors_covariance`, which would replace those measurements
  with a constant. Consider `--use_robust_loss_on_prior_position 1` so one
  bad fix cannot drag the window.
- With priors on, the model is metric and `anchors.txt` realignment is
  skipped; the georeferencing Sim3 becomes a check (expect scale within 1% of
  1.0) rather than a correction.
- `prior_residuals.tsv` rides along with the model directory promotion, like
  `anchors.txt`, and the ledger gains five `prior_*` columns.
- To enable the windowed path in production, add
  `--optimize_window_size 20 --full_solve_interval 10` to the
  `incremental_global_mapper` invocation in `colmap_runner.py`. No other
  server change is required; `anchors.txt` rides along with the model
  directory promotion.

## Tests

- `sfm/rotation_utils_test` — Karcher mean and candidate selection.
- `sfm/global_mapper_bootstrap_test` — pure solver (both edge orderings,
  outlier rejection, gravity gate, degenerate translation, fallbacks),
  database-level hold-one-out bootstraps, windowed solve, and a drip-feed
  regression mirroring the production loop.
- `controllers/incremental_global_pipeline_test` — end-to-end adds (full +
  windowed), anchor selection/IO/realignment, a 100-step anchored-vs-rolling
  drift simulation, and the no-prior fallback.

Known limitations:

- Track establishment in the full-solve path still rebuilds from scratch
  (O(total matches)); the windowed path avoids it entirely.
- The windowed path assumes the prior model's 2D–3D links are consistent
  with the database (true when features are extracted once per image, as
  the production server does).
- `LoadPriorPoses` imports the prior model's refined camera intrinsics
  (overriding the database's coarse focal priors), and the windowed path
  additionally seeds each new image's camera from a matching prior camera.
  This matters a lot in practice: on a real phone session the database
  focal prior was ~30% below the bundle-adjusted value, and without the
  import the windowed solve collapsed the model within a few adds.
- Even with correct intrinsics, windowed solves accumulate small pose drift
  relative to a full solve (the relative-pose evidence is decomposed with
  approximate intrinsics and never globally re-averaged). **Always pair
  `--optimize_window_size` with `--full_solve_interval` (5–10) in
  production**; the periodic full solve plus anchor realignment bounds the
  drift. Benchmark on your own data with
  `--variants full,windowN,windowN.M` before choosing N and M.
- Multi-camera rigs take the same code paths but have not been benchmarked
  on real rig data; the synthetic rig tests in the upstream suite stay
  green.
