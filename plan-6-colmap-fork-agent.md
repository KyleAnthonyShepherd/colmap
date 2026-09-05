# Plan 6 — COLMAP fork work package

For an agent working in `W:\Kyle_Shepherd\photogrammetry_app\colmap` (the
`incremental-glomap` fork). Written 2026-09-05 from the first session that
reconstructed and finalized end to end (`test1-2`, 69 frames). Every item below
is a thing that session actually hit; none of it is speculative.

**Read first:** `HANDOFF-CAPTURE-REVIEW.md` §4b and §4c in `site_mapper/`. They
are the evidence for all four items and give the exact commands that produced
the numbers quoted here. `PLAN.md` §4.3 and §4.5 give the prior design.

**Build in use:** COLMAP 4.1.0.dev0, commit `44171a33`, 2026-08-27, with CUDA,
at `/home/dragonsmith/colmap/build/src/colmap/exe/colmap` on the server.

---

## 0. The one-paragraph summary

The site pipeline hands COLMAP high-quality RTK positions (1.6 cm vertical,
12 cm horizontal, `fix_quality: 4` on every frame) and COLMAP's incremental
path cannot use them. `pose_prior_mapper` can — and when it does, the model
comes out metric to 0.5% with no external scale, the camera-height IQR halves,
and an 8σ pose blunder is suppressed. But `pose_prior_mapper` is a *cold*
solver, and this system is a live-walk system whose whole point is
`incremental_global_mapper`. So the constraint that works is unavailable to the
path that matters. That is item 1, and it is worth more than the other three
together.

---

## 1. Position priors in `incremental_global_mapper` — **highest value**

### What exists now

| binary | prior options |
|---|---|
| `mapper` | **none** |
| `global_mapper` | **none** |
| `incremental_global_mapper` | `--prior_reconstruction_path`, `--bootstrap_min_inliers`, `--realign_to_prior`, `--gravity_uncertainty_deg`, `--allow_lossy_prior_import` |
| `pose_prior_mapper` | `--overwrite_priors_covariance`, `--prior_position_std_{x,y,z}`, `--use_robust_loss_on_prior_position`, `--prior_position_loss_scale` |

`incremental_global_mapper`'s "prior" is a prior *reconstruction* — a seed
model. It has nothing to do with GNSS. The `pose_priors` table is read by
`SpatialPairGenerator::ReadPositionPriorData` for **pair selection only**.
`site_mapper/server/app/site/db_priors.py` says so in its own docstring, which
calls `pose_prior_mapper` "a future `pose_prior_mapper` A/B".

### The measurement that justifies the work

Identical database (`sessions/test1-2`, 69 images, 675 verified pairs,
64,268 inliers), two solvers:

| | `mapper` (priorless) | `pose_prior_mapper` |
|---|---|---|
| registered | 69/69 @ 0.633 px | 69/69 @ 0.630 px |
| Sim3 scale | 1.7037 | **1.0053** |
| camera height IQR | 0.075 m | **0.036 m** |
| worst GNSS/SfM step ratio | **4.17** | **1.47** |

**The scale line is the point.** A priorless model has no scale of its own, so
a wrong Sim3 and a wrong model are indistinguishable — which is exactly how
this session lost an afternoon. With position priors the model is metric before
georeferencing runs at all, and the Sim3 becomes a *check* rather than a
correction.

The step-ratio line is the second point. At `img_0059` the priorless solve moved
the camera 1.238 m where the operator had moved 0.297 m, and every subsequent
frame inherited the displacement. Against a horizontal σ of 0.12 m that is ~8σ:
a position-prior cost term makes it impossible, and with priors it fell to
1.47×.

### The task

Port the prior-position residual from `pose_prior_mapper` into
`incremental_global_mapper`, so a **windowed add** can be constrained by GNSS.

* Same option names, so the two binaries stay interchangeable in scripts:
  `--use_prior_position`, `--use_robust_loss_on_prior_position`,
  `--prior_position_loss_scale`, `--overwrite_priors_covariance`,
  `--prior_position_std_{x,y,z}`.
* Default the covariance to **read from the database**. The site pipeline
  writes real per-frame covariance (`db_priors.pack_doubles_le`), and the 1 m
  default std would throw away a 1.6 cm vertical measurement.
* The cost must apply during **PnP acceptance**, not only in the final bundle
  adjustment. The failure mode is a pose that is admitted and then displaces
  its successors; rejecting it at registration is worth far more than
  optimising it afterwards.
* Respect `pose_priors.coordinate_system`. The site path writes CARTESIAN (1)
  ENU metres deliberately — see the `db_priors` docstring on why UNDEFINED (-1)
  is wrong here.

### Acceptance

Re-run `sessions/test1-2` through the live pipeline (not a cold solve) and get
scale within 1% of 1.0, camera-height IQR under 0.05 m, and no step-ratio
outlier above 2.0. The site repo now computes that last number itself —
`site_mapper.georef.step_consistency`, surfaced in `site/qa/metrics.json` as
`step_ratio_worst`.

---

## 2. Report per-image prior residuals

### Why

Finding the `img_0059` blunder took six rounds of ad-hoc analysis: reading
`residuals.csv`, re-deriving `R_enu_from_cam` per frame from the model and the
Sim3, rotating residuals into each camera's own frame, and comparing
consecutive steps. COLMAP already knows all of this at the end of every bundle
adjustment — the gap between each registered camera and its position prior, and
whether the robust loss down-weighted it.

Everything upstream saw only *consequences*: `seam_step_mean` 23.6 against a
limit of 10, `mean_agreement` 0.259 against 0.6, and a `stripe correlation 0.67:
lever arm or tilt wrong` warning that sent the investigation after the lever arm
— which turned out to be innocent, and took a sweep, a direct measurement and an
iterative fit to exonerate.

### The task

Emit, per registered image, alongside the model:

```
image_id, name, prior_position(xyz), solved_position(xyz),
residual(xyz), residual_norm, sigma_used(xyz), robust_weight, used_in_ba
```

A TSV beside `cameras.bin` is fine; extending `incremental_ledger.tsv` is
better, since the site pipeline already reads that per batch and it would make
the signal available *during the walk* rather than at finalize.

### Acceptance

A stuck or displaced frame is identifiable from that file alone, with no
external reconstruction of the geometry.

---

## 3. The antenna as a rig sensor — **the structural fix**

### The problem it removes

The GNSS antenna is 0.1524 m from the camera (measured, not nominal — the
0.165 m in `PLAN.md` §5.1, `FIELD_GUIDE.md` and `HANDOFF-WP3.md` is stale). To
turn an antenna position into a camera-centre prior you need the camera's full
orientation including yaw, and with no magnetometer allowed that only exists
*after* the model is solved. So the site repo carries an entire apparatus for
this chicken-and-egg:

* `priors.py`'s three-fidelity ladder — `vertical_only` →
  `gyro_dead_reckoned` → `solved_rotation`, with sigma inflated by
  `UNKNOWN_YAW_SIGMA_M = 0.12` while yaw is unknown;
* `georef.georef_two_pass` — fit, re-derive every prior from the fitted
  rotations, fit again, up to four passes;
* `georef.stripe_correlation` — a diagnostic whose only job is to say whether
  the above worked.

All of it exists because COLMAP wants a camera-centre prior.

### What makes it tractable now

This build has rigs: `rigs.bin`, `frames.bin`,
`--Mapper.ba_refine_sensor_from_rig`. **The antenna is exactly a second sensor
rigidly mounted to the camera.** If a rig could contain a non-camera sensor
carrying position priors, the site path would feed raw antenna ENU straight in,
the lever arm would *be* `sensor_from_rig`, and bundle adjustment would solve
the pose and the arm together.

### The task

1. Allow a rig sensor of a non-camera type with position priors attached, its
   `sensor_from_rig` a fixed transform.
2. Let the prior-position residual attach to that sensor rather than to the
   camera centre: `residual = P_antenna_measured − (C_cam + R_enu_from_cam ·
   l_sensor)`.
3. Make `sensor_from_rig` refinable under `--Mapper.ba_refine_sensor_from_rig`,
   so BA *estimates* the lever arm.

Point 3 is the prize. It would have answered in one run what this session spent
an entire investigation on — and it would keep answering it every session, for
free, as the mount changes.

### Acceptance

Feed antenna positions with `l = [0, 0.1524, 0]` declared, and BA recovers a
`sensor_from_rig` translation within a couple of centimetres of it, with the
stripe correlation collapsing below 0.3.

### Caveat, and it is a real one

The stripe on `test1-2` did **not** collapse under `pose_prior_mapper` — it
went from 0.674 to 0.744, and a distributed ~7 cm camera-frame systematic
survives on both solvers. The lever arm has been eliminated as its cause on
physical grounds (see §4c: no vector an S23 can host fits, and an iterative fit
runs the arm away to 0.337 m while inliers fall 14 → 9). So a rig term may fit
the arm correctly and still leave that 7 cm behind. **Do not treat a residual
stripe as proof this item failed** — verify against the recovered
`sensor_from_rig` translation, which is a direct measurement.

---

## 4. Documentation only, no code

* `mapper --help | grep -i prior` returns **nothing**, so the natural guess
  `--Mapper.use_prior_position` fails to parse and looks like a version problem.
  Worth a one-line note in `mapper`'s help pointing at `pose_prior_mapper`.
* `pose_prior_mapper`'s prior options are **un-prefixed**
  (`--use_robust_loss_on_prior_position`) while its others are `Mapper.`-prefixed
  (`--Mapper.ba_refine_focal_length`). That inconsistency cost this session one
  failed run. Either prefix them or document them.

---

## 5. Order of work

1. **Item 1** — the live path cannot use GNSS at all today. Everything else is
   an improvement on a system that already works; this is a capability gap.
2. **Item 2** — small, and it makes items 1 and 3 debuggable. Do it early.
3. **Item 4** — minutes.
4. **Item 3** — the largest change, and it retires the most code, but only
   sensible once 1 and 2 are in and the live path is producing metric models.

## 6. What the site repo already does, so don't duplicate it

Landed 2026-09-05, in `site_mapper/`:

* `georef.step_consistency` + `STEP_RATIO_LIMIT` / `STEP_ABS_MIN_M` — the
  GNSS-step vs SfM-step check, written into `sim3.json` and checked in
  `qa.py`'s verdict. This is the acceptance test for item 1.
* `colmap_site.run_pose_prior_mapper` and `pipeline._cold_solve` — every cold
  solve now goes through `pose_prior_mapper`, falling back to `mapper` if the
  binary or its flags are missing (`SITE_COLD_MAPPER`).
* `pipeline._promote_largest_component` — `mapper` writes one model per
  connected component numbered by *discovery order*, not size; on `test1-2` a
  3-image fragment landed at `sparse/0` while the complete 69-image model was
  at `sparse/1`.
* `pipeline._rebootstrap` + `SITE_STALL_BATCHES` — recovery from a wedged seed.

The staging tree is `site_mapper/server/`, **not applied** to the spacemapper
repo. See `HANDOFF-CAPTURE-REVIEW.md` §1 on the mirror rule, and diff before
copying anything anywhere.
