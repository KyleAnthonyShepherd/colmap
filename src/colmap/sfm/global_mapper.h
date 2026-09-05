#pragma once

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/sfm/prior_positions.h"

#include <filesystem>
#include <limits>
// ── INCREMENTAL-ADD (new includes) ────────────────────────────────────────
#include <optional>
#include <unordered_set>
// ─────────────────────────────────────────────────────────────────────────

namespace colmap {

struct GlobalMapperOptions {
  // Number of threads.
  int num_threads = -1;

  // PRNG seed for all stochastic methods during reconstruction.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the pipeline is deterministic with the given
  // seed.
  int random_seed = -1;

  // The image path at which to find the images to extract point colors.
  // If not specified, all point colors will be black.
  std::filesystem::path image_path;

  // Options for each component
  RotationEstimatorOptions rotation_averaging;
  GlobalPositionerOptions global_positioning;
  BundleAdjustmentOptions bundle_adjustment = [] {
    BundleAdjustmentOptions options;
    options.refine_sensor_from_rig = false;
    options.min_track_length = 3;
    options.print_summary = false;
    if (options.ceres) {
      options.ceres->loss_function_type =
          CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
      options.ceres->use_gpu = true;
      // TODO: Investigate whether disabling auto solver selection and using
      // explicit SPARSE_SCHUR + CLUSTER_TRIDIAGONAL is necessary for global
      // SfM, or if we can just rely on COLMAP's auto selection.
      options.ceres->auto_select_solver_type = false;
      options.ceres->solver_options.function_tolerance = 1e-5;
      options.ceres->solver_options.max_num_iterations = 200;
      options.ceres->solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.ceres->solver_options.preconditioner_type =
          ceres::CLUSTER_TRIDIAGONAL;
    }
    return options;
  }();

  IncrementalTriangulator::Options retriangulation = [] {
    IncrementalTriangulator::Options opts;
    opts.complete_max_reproj_error = 15.0;
    opts.merge_max_reproj_error = 15.0;
    opts.min_angle = 1.0;
    return opts;
  }();

  // Track establishment options.
  // Max pixel distance between observations of the same track within one image.
  double track_intra_image_consistency_threshold = 10.;

  // Required number of tracks per view before early stopping.
  int track_required_tracks_per_view = std::numeric_limits<int>::max();

  // Minimum number of views per track.
  int track_min_num_views_per_track = 3;

  // Thresholds for each component.
  double max_angular_reproj_error_deg = 1.;  // for global positioning
  double max_normalized_reproj_error = 1e-2;  // for bundle adjustment
  double min_tri_angle_deg = 1.;              // for triangulation

  // Control the number of iterations for bundle adjustment.
  int ba_num_iterations = 3;

  // Whether to skip the fixed-rotation stage in bundle adjustment.
  // By default, BA runs in two stages: first with fixed rotations (position
  // only), then with full optimization. Setting this to true skips the first
  // stage and runs full optimization directly.
  bool ba_skip_fixed_rotation_stage = false;

  // Whether to skip the joint optimization stage in bundle adjustment.
  // When set to true, only the fixed-rotation stage is run (optimizing
  // positions only). This is mutually exclusive with
  // ba_skip_fixed_rotation_stage.
  bool ba_skip_joint_optimization_stage = false;

  // Control the flow of the global sfm
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;

  // ── INCREMENTAL-ADD OPTIONS ────────────────────────────────────────────
  //
  // These options are only active when LoadPriorPoses() +
  // BootstrapNewImagePoses() are called before Solve().  The typical
  // single-new-image workflow sets skip_rotation_averaging = true and lets
  // global positioning + BA handle the rest.

  // Minimum number of PoseGraph inliers required on a prior→new edge before
  // it contributes a rotation candidate.  Raise this if noisy matches produce
  // bad relative poses.
  int bootstrap_min_inliers = 10;

  // Maximum rotation error (degrees) between a bootstrapped candidate and
  // the weighted Karcher mean before the candidate is considered an outlier
  // and discarded from a second averaging pass.
  double bootstrap_max_candidate_deg = 10.0;

  // After BootstrapNewImagePoses() and Solve(), re-align the output
  // reconstruction back to the prior coordinate frame using a Sim3 fit over
  // the prior image centres.  Always recommended when skip_rotation_averaging
  // = true to correct any metric drift from global positioning.
  bool realign_to_prior_after_solve = true;

  // Maximum disagreement (degrees) between the measured gravity direction of
  // a new image (from its pose prior) and the gravity direction implied by a
  // rotation candidate before the candidate is rejected. Requires gravity
  // priors on both the new image and the prior images. This is a stronger
  // outlier test than distance-from-mean when only 2-3 candidates exist.
  // Set <= 0 to disable.
  double bootstrap_max_gravity_error_deg = 15.0;

  // When > 0, incremental adds run a windowed local solve instead of the
  // full global pipeline: prior 3D points are imported, only the new
  // image(s) are triangulated, and bundle adjustment optimizes just the new
  // image(s) plus the `optimize_window_size` most covisible registered
  // images. All other poses stay constant, so per-add cost is ~O(window)
  // instead of O(N) and the output remains in the prior coordinate frame by
  // construction (no Sim3 realignment needed). Typical values: 15-25.
  // 0 (default) keeps the previous full-solve behavior.
  int optimize_window_size = 0;

  // A windowed solve structurally cannot refine camera intrinsics: both
  // AdjustLocalBundle (num_images < num_reg_images_per_camera) and
  // AddPointToProblem (any track element outside the config) force the
  // camera's intrinsics constant. With a single shared camera, whatever
  // intrinsics the model holds when it goes windowed are frozen forever --
  // which is how a model ends up optimizing refined geometry against
  // unrefined intrinsics and collapsing. So the pipeline runs full solves
  // (which do refine intrinsics globally) until intrinsics converge, and
  // only then switches to windowed adds. See plan-6 sec. 5.

  // Maximum relative deviation of a camera's mean focal length from its
  // running average for that add to count as "stable".
  //
  // This is a stationarity test, not a decay test. The focal estimate does
  // not settle to a fixed value: each full solve rebuilds all structure and
  // re-lands slightly differently, so on real data it jitters in a band
  // (measured: 3e-4 to 3e-3 relative, non-decreasing, around 985.7 px). A
  // consecutive-change test against that noise floor never fires. What the
  // gate actually needs to distinguish is "roughly right" from "grossly
  // wrong" -- the failure it exists to prevent is freezing the intrinsics at
  // something like the database's coarse field-of-view prior, which can be
  // 30-40% off. 5e-3 sits above the measured rebuild-noise band and two
  // orders of magnitude below a gross error.
  double intrinsics_convergence_rel_tol = 5e-3;

  // Number of consecutive stable adds required to declare convergence.
  int intrinsics_min_stable_adds = 2;

  // Minimum number of adds before convergence may be declared at all.
  int intrinsics_min_adds = 3;

  // Drift trigger: when an add's mean reprojection error exceeds the best
  // ever recorded times this factor, run an intrinsics-only global bundle
  // adjustment plus a global structure-recovery pass, and re-validate
  // convergence. <= 0 disables the drift monitor.
  double intrinsics_drift_factor = 1.25;

  // Register new images by absolute pose (PnP + RANSAC) against the prior
  // reconstruction's 3D points, falling back to BootstrapNewImagePoses only
  // for images PnP declines.
  //
  // The bootstrap averages one rotation candidate per pose-graph edge. For a
  // weakly-connected image that is the wrong operator: measured on
  // sessions/camsnap, img_0008 has nine edges of which one carries 456
  // inliers, one 171, and seven carry 16-20 (i.e. noise). Averaging lets the
  // seven outvote the two, and the gravity gate then discards candidates
  // weight-blind on top of that. The result was a rotation off by 2.8 deg
  // (and 45-75 deg for two other images), after which triangulation finds no
  // consistent rays and the image is written out registered with zero
  // observations -- with no residuals left for bundle adjustment to pull it
  // back. PnP uses the prior 3D structure directly and RANSAC *rejects*
  // outliers rather than averaging them, and it can decline (returning
  // false) instead of emitting a confident wrong pose.
  bool use_pnp_registration = true;

  // Whether an image PnP declines should still be handed to
  // BootstrapNewImagePoses. Off by default, because the decline is the
  // useful signal: measured on sessions/camsnap, PnP accepted 6 of 8 images
  // and declined exactly the two that the bootstrap went on to register with
  // a garbage pose and zero observations. An unregistered image is strictly
  // better than a mis-registered one -- it is retried automatically on every
  // later add (RegisterNewImagesByPnP sweeps all poseless images, not just
  // the newest), and by then there is usually more structure to match
  // against, whereas a bad pose with zero observations gives bundle
  // adjustment no residuals to recover from and is permanent.
  bool bootstrap_fallback_when_pnp_declines = false;

  // Minimum number of inlier 2D-3D correspondences for PnP registration
  // (IncrementalMapper::Options::abs_pose_min_num_inliers). COLMAP's default
  // of 30 is tuned for from-scratch incremental mapping; a weakly-connected
  // image on the incremental path can have only 35-64 correspondences in
  // total, so 30 inliers is unreachable for it even when RANSAC finds a clean
  // consensus at a healthy inlier ratio.
  int pnp_min_num_inliers = 30;

  // Expected worst-case error of the gravity priors, in degrees. Widens the
  // PnP inlier threshold by focal*tan(this) while the upright solver is in
  // use. Kept small deliberately -- the term scales with focal length and
  // admitting outliers is worse than rejecting a few correct
  // correspondences. See
  // AbsolutePoseEstimationOptions::gravity_uncertainty_deg.
  double gravity_uncertainty_deg = 0.5;

  // The windowed prior-structure import is lossless on a healthy prior: its
  // only drop paths signal prior/database disagreement. By default any drop
  // fails the add, leaving the last good reconstruction in place rather than
  // silently writing a degraded one. Set true to import what it can and
  // continue anyway.
  bool allow_lossy_prior_import = false;

  // ── POSITION PRIORS ────────────────────────────────────────────────────
  //
  // The live path is the one that matters for a walk, and until now it could
  // not use GNSS at all: `prior_reconstruction_path` is a prior *model*, and
  // the `pose_priors` table was read only for pair selection. These options
  // are named to match `pose_prior_mapper` exactly, so the two binaries stay
  // interchangeable in scripts. See plan-6 item 1.

  // Whether to constrain the solve with the database's position priors.
  bool use_prior_position = false;

  // Whether to down-weight prior-position residuals with a Cauchy loss.
  bool use_robust_loss_on_prior_position = false;

  // Threshold on the (covariance-whitened) residual for the robust loss.
  // chi2 for 3 DOF at 95% = 7.815.
  double prior_position_loss_scale = 7.815;

  // Standard deviation, in metres, assumed for a prior that carries no
  // covariance. The covariance is read from the database by default -- the
  // site pipeline writes real per-frame values, and a 1 m default would throw
  // away a 1.6 cm vertical measurement -- so this is a last resort, not a
  // setting to reach for.
  double prior_position_fallback_stddev = 1.0;

  // Absolute-pose acceptance gate: a PnP registration is declined when its
  // camera centre is further from its prior than BOTH this many sigmas and
  // `prior_position_max_error_m` metres. Rejecting a blunder at registration
  // is worth far more than optimising it afterwards, because a pose that is
  // admitted has already displaced every frame registered after it. <= 0
  // disables the gate. See IncrementalMapper::Options for the reasoning
  // behind requiring both conditions.
  double prior_position_max_error_sigma = 5.0;
  double prior_position_max_error_m = 0.1;

  // ─────────────────────────────────────────────────────────────────────
};

// ── INCREMENTAL-ADD ACCOUNTING ─────────────────────────────────────────────

// Per-add structure accounting for an incremental windowed solve.
//
// Every 3D point that enters an add is accounted for on exit. Filtering
// inside local bundle adjustment deletes observations from *imported prior*
// points, not just new ones, so a drip-fed session can bleed structure
// steadily while every individual add "succeeds". The ledger makes that
// visible as data instead of a guess: it is appended to
// `incremental_ledger.tsv` beside the model, one row per add.
//
// The import counters are all zero on a healthy prior -- they only become
// non-zero when the prior reconstruction and the database disagree.
struct IncrementalAddLedger {
  // Prior structure import.
  size_t prior_points_total = 0;
  size_t imported = 0;
  size_t dropped_missing_image = 0;
  size_t dropped_bad_point2D_idx = 0;
  size_t dropped_already_linked = 0;
  size_t dropped_short_track = 0;

  // New images registered by PnP rather than by the bootstrap fallback.
  size_t pnp_registered = 0;

  // Structure changes during the solve.
  size_t triangulated = 0;
  size_t merged = 0;
  size_t completed = 0;
  size_t recovered = 0;  // from the global recovery pass, when it ran
  size_t filtered = 0;

  // Outcome.
  size_t points_out = 0;
  double mean_reproj_error = 0.0;
  double mean_track_length = 0.0;
  bool intrinsics_refreshed = false;

  // Position-prior agreement for this add (plan-6 item 2). All zero when
  // priors are not in use. `worst_sigma` is the Mahalanobis distance of the
  // largest disagreement -- the number that would have named `img_0059` on
  // the add that broke it, instead of six rounds of re-deriving the geometry
  // from `residuals.csv` afterwards.
  size_t prior_residuals = 0;
  double prior_residual_rms = 0.0;
  double prior_residual_worst = 0.0;
  double prior_residual_worst_sigma = 0.0;
  image_t prior_residual_worst_image = kInvalidImageId;

  // Total prior points lost at import time. Zero unless the prior and the
  // database disagree.
  size_t DroppedAtImport() const {
    return dropped_missing_image + dropped_bad_point2D_idx +
           dropped_already_linked + dropped_short_track;
  }
};

// ── INCREMENTAL-ADD FREE FUNCTIONS ─────────────────────────────────────────

// Evidence from a single pose-graph edge connecting a new (unregistered)
// image to a prior (registered) image.
struct BootstrapEdgeObservation {
  // Relative pose of the edge, in the pose graph's stored orientation.
  Rigid3d cam2_from_cam1;
  // True if the prior image is cam1 of the edge (and the new image cam2).
  bool prior_is_cam1 = true;
  // Absolute pose of the prior image.
  Rigid3d prior_cam_from_world;
  // Non-negative weight, typically the edge's verified match count.
  double weight = 1.0;
};

// Solves the absolute pose of one new image from edges to prior images.
//
// Rotation: each edge contributes one candidate absolute rotation
//   R_new = R_rel * R_prior        (prior is cam1)
//   R_new = R_rel^{-1} * R_prior   (prior is cam2)
// combined by a weighted Karcher mean on SO(3). When more than one candidate
// exists, candidates further than `max_candidate_deg` from the mean are
// discarded and the mean is recomputed.
//
// Translation: each edge with a non-degenerate relative translation
// contributes a world-space ray from the prior camera centre toward the new
// camera centre; the centre is the weighted least-squares point closest to
// all rays. Falls back to the weighted centroid of the prior centres when
// the ray system is near-singular (e.g. a single ray or colinear priors).
//
// Gravity consistency gate for rotation candidates.
struct BootstrapGravityGate {
  // Measured gravity direction in the new image's camera frame (unit norm),
  // e.g. from the image's pose prior.
  Eigen::Vector3d gravity_in_new_cam;
  // Gravity direction in the world frame of the prior reconstruction (unit
  // norm), e.g. averaged from the prior images' gravity priors and poses.
  Eigen::Vector3d gravity_in_world;
  // Maximum angle (degrees) between measured and candidate-implied gravity.
  double max_error_deg = 15.0;
};

// Returns std::nullopt when `observations` is empty.
//
// When `gravity_gate` is set, observations whose implied camera-frame
// gravity (R_candidate * gravity_in_world) disagrees with the measured
// gravity by more than max_error_deg are discarded before averaging. If all
// observations fail the gate, solving proceeds ungated with a warning (a
// single bad accelerometer sample must not block registration).
std::optional<Rigid3d> SolvePoseFromPriorEdges(
    const std::vector<BootstrapEdgeObservation>& observations,
    double max_candidate_deg,
    const std::optional<BootstrapGravityGate>& gravity_gate = std::nullopt);

class GlobalMapper {
 public:
  explicit GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache);

  // Prepare the mapper for a new reconstruction. This will initialize the
  // reconstruction and view graph from the database.
  void BeginReconstruction(
      const std::shared_ptr<Reconstruction>& reconstruction);

  // Run the global SfM pipeline.
  bool Solve(const GlobalMapperOptions& options);

  // Run rotation averaging to estimate global rotations.
  bool RotationAveraging(const RotationEstimatorOptions& options);

  // Establish tracks from feature matches.
  void EstablishTracks(const GlobalMapperOptions& options);

  // Estimate global camera positions.
  bool GlobalPositioning(const GlobalPositionerOptions& options,
                         double max_angular_reproj_error_deg,
                         double max_normalized_reproj_error,
                         double min_tri_angle_deg);

  // Run iterative bundle adjustment to refine poses and structure.
  bool IterativeBundleAdjustment(const BundleAdjustmentOptions& options,
                                 double max_normalized_reproj_error,
                                 double min_tri_angle_deg,
                                 int num_iterations,
                                 bool skip_fixed_rotation_stage = false,
                                 bool skip_joint_optimization_stage = false);

  // Iteratively retriangulate tracks and refine to improve structure.
  bool IterativeRetriangulateAndRefine(
      const IncrementalTriangulator::Options& options,
      const BundleAdjustmentOptions& ba_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  // ── INCREMENTAL-ADD PUBLIC API ─────────────────────────────────────────

  // Seed the current reconstruction with poses from `prior_reconstruction`.
  //
  // For every image found in BOTH reconstructions (matched by image_t id),
  // this method:
  //   1. Sets cam_from_world to the prior's value.
  //   2. Marks the image as registered.
  //   3. Records its id in prior_image_ids_ for later use.
  //
  // Must be called AFTER BeginReconstruction() and BEFORE Solve().
  // Images present only in the prior (not in the new database) are silently
  // skipped.
  void LoadPriorPoses(const class Reconstruction& prior_reconstruction);

  // Derive initial rotations for images not yet registered.
  //
  // For each unregistered image in the current reconstruction, walks the
  // PoseGraph looking for edges to already-registered (prior) images.  Each
  // valid edge contributes one candidate absolute rotation:
  //
  //   R_new = R_rel * R_prior     (if the pair is prior → new)
  //   R_new = R_rel^{-1} * R_prior  (if the pair is new → prior)
  //
  // Candidates are weighted by their inlier count and averaged via a
  // weighted Karcher mean on SO(3).  A second pass optionally discards
  // outlier candidates before the final mean.
  //
  // Must be called AFTER LoadPriorPoses().
  //
  // Returns the set of image ids for which both rotation and translation were bootstrapped.
  // Images for which no valid prior neighbour exists remain unregistered.
  std::unordered_set<image_t> BootstrapNewImagePoses(
      const GlobalMapperOptions& options);

  // Returns the set of image ids loaded from the prior reconstruction.
  // Empty until LoadPriorPoses() has been called.
  const std::unordered_set<image_t>& PriorImageIds() const {
    return prior_image_ids_;
  }

  // Imports the prior reconstruction's 3D points into the current
  // reconstruction, preserving point3D_t ids and re-linking each track
  // element's 2D-3D association. Must be called AFTER LoadPriorPoses().
  //
  // On a healthy prior this is lossless; every drop path signals that the
  // prior and the database disagree. Returns false on any loss unless
  // options.allow_lossy_prior_import is set. Accumulates into `ledger`.
  bool ImportPriorPoints3D(const GlobalMapperOptions& options,
                           const class Reconstruction& prior_reconstruction,
                           IncrementalAddLedger* ledger = nullptr);

  // Registers every currently unregistered image by absolute pose against
  // the already-imported prior structure, and returns the set that was
  // registered. Requires ImportPriorPoints3D() to have run first; returns
  // empty (with a warning) if there is no structure to match against.
  //
  // Images this declines are left unregistered so BootstrapNewImagePoses()
  // can still try them -- it skips images that already have a pose, so the
  // two compose as a fallback chain without further bookkeeping.
  std::unordered_set<image_t> RegisterNewImagesByPnP(
      const GlobalMapperOptions& options,
      IncrementalAddLedger* ledger = nullptr);

  // Windowed local solve for incremental adds (see
  // GlobalMapperOptions::optimize_window_size). Imports the prior
  // reconstruction's 3D points, triangulates the new image(s) against them,
  // and runs iterative local bundle adjustment around each new image with a
  // covisibility window of `options.optimize_window_size` images. Poses
  // outside the window contribute constant-pose residuals only, so the
  // output stays in the prior coordinate frame.
  //
  // Must be called AFTER LoadPriorPoses() and BootstrapNewImagePoses();
  // `new_image_ids` is the return value of the latter. Images listed in
  // `constant_image_ids` (e.g. gauge anchors) keep their poses constant
  // even when they fall inside the covisibility window.
  //
  // Fails (returns false) if any prior 3D structure could not be imported
  // faithfully, unless options.allow_lossy_prior_import is set. When
  // `ledger` is non-null it receives this add's structure accounting.
  bool SolveIncrementalWindowed(
      const GlobalMapperOptions& options,
      const class Reconstruction& prior_reconstruction,
      const std::unordered_set<image_t>& new_image_ids,
      const std::unordered_set<image_t>& constant_image_ids = {},
      IncrementalAddLedger* ledger = nullptr);

  // Intrinsics-only global bundle adjustment followed by global structure
  // recovery. A local window can never refine intrinsics (see
  // GlobalMapperOptions::intrinsics_drift_factor), so this gives that one
  // global parameter its own global pass: all registered images, every pose
  // and every 3D point held constant, only camera params free. That is a
  // ~4-8 DOF problem regardless of N.
  //
  // Then re-establishes structure that was filtered away while the
  // intrinsics were wrong: a filtered observation is not destroyed, the
  // feature match is still in the database, so it can be recovered once the
  // intrinsics are corrected. Returns the number of observations recovered
  // and accumulates into `ledger` when non-null.
  size_t RefreshIntrinsicsGlobally(const GlobalMapperOptions& options,
                                   IncrementalAddLedger* ledger = nullptr);

  // Reads the database's position priors and expresses them in the current
  // reconstruction's world frame, caching the result for the rest of this
  // solve. Must be called AFTER LoadPriorPoses(), so the fit has registered
  // camera centres to work from, and BEFORE RegisterNewImagesByPnP() /
  // SolveIncrementalWindowed(), which both consume it.
  //
  // Returns false when no usable priors could be placed; the solve then
  // proceeds unconstrained, exactly as before. No-op returning false when
  // options.use_prior_position is not set.
  bool ComputeWorldPositionPriors(const GlobalMapperOptions& options);

  // The priors computed above. Empty until ComputeWorldPositionPriors() has
  // run successfully.
  const WorldPositionPriors& PositionPriors() const {
    return position_priors_;
  }

  // Images whose pose was free AND constrained by a position prior during
  // this solve's bundle adjustment -- the `used_in_ba` column of the
  // per-image residual report. See plan-6 item 2.
  const std::unordered_set<image_t>& PriorConstrainedImageIds() const {
    return prior_constrained_image_ids_;
  }

  // Whether this solve is constrained by position priors. When it is, the
  // reconstruction is metric and must not be renormalized: Reconstruction::
  // Normalize() rescales the model, which is exactly the metric scale the
  // priors just established.
  bool UsePositionPriors() const { return !position_priors_.Empty(); }
  // ──────────────────────────────────────────────────────────────────────

  // Getter functions.
  std::shared_ptr<class Reconstruction> Reconstruction() const;

 private:
  // Collects per-image gravity priors from the database and derives the
  // gravity direction in the reconstruction's world frame by averaging the
  // prior images' measurements through their known poses. Returns nullopt
  // when no registered prior image carries a gravity prior.
  std::optional<Eigen::Vector3d> CollectGravityPriors(
      std::unordered_map<image_t, Eigen::Vector3d>* image_to_gravity) const;

  std::shared_ptr<const DatabaseCache> database_cache_;
  std::shared_ptr<class PoseGraph> pose_graph_;
  std::shared_ptr<class Reconstruction> reconstruction_;

  // ── INCREMENTAL-ADD PRIVATE STATE ─────────────────────────────────────
  // Image ids whose poses were loaded from a prior reconstruction.
  // Populated by LoadPriorPoses(); consumed by BootstrapNewImagePoses()
  // and by the optional realignment step in IncrementalGlobalPipeline::Run().
  std::unordered_set<image_t> prior_image_ids_;

  // Subsampled prior 3D points (world coordinates) used for a cheap
  // cheirality sanity check on bootstrapped poses.
  std::vector<Eigen::Vector3d> prior_points_sample_;

  // Database position priors expressed in this reconstruction's world frame.
  // Populated by ComputeWorldPositionPriors(); consumed by PnP registration
  // and by the windowed solve's local bundle adjustment.
  WorldPositionPriors position_priors_;

  // See PriorConstrainedImageIds().
  std::unordered_set<image_t> prior_constrained_image_ids_;
  // ──────────────────────────────────────────────────────────────────────
};

}  // namespace colmap
