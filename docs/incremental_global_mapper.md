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
    [--realign_to_prior 1]
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
