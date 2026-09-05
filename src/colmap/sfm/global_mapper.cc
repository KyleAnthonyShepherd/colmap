#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/math/union_find.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"
#include "colmap/sfm/rotation_utils.h"

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <type_traits>

namespace colmap {
namespace {

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);

  return ba->Solve()->IsSolutionUsable();
}

GlobalMapperOptions InitializeOptions(const GlobalMapperOptions& options) {
  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions opts = options;
  if (opts.random_seed >= 0) {
    opts.rotation_averaging.random_seed = opts.random_seed;
    opts.global_positioning.random_seed = opts.random_seed;
    opts.global_positioning.use_parameter_block_ordering = false;
    opts.retriangulation.random_seed = opts.random_seed;
  }
  opts.global_positioning.solver_options.num_threads = opts.num_threads;
  if (opts.bundle_adjustment.ceres) {
    opts.bundle_adjustment.ceres->solver_options.num_threads = opts.num_threads;
  }
  return opts;
}

// Copies the position-prior settings onto an IncrementalMapper::Options, so
// PnP registration and local bundle adjustment are governed by exactly the
// same constraint. A no-op when priors are off or none could be placed.
void ApplyPositionPriorOptions(const WorldPositionPriors& position_priors,
                               IncrementalMapper::Options* mapper_options) {
  if (position_priors.Empty()) return;
  const PriorPositionSettings& settings = position_priors.settings;
  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = settings.use_robust_loss;
  mapper_options->prior_position_loss_scale = settings.loss_scale;
  mapper_options->prior_position_fallback_stddev = settings.fallback_stddev;
  mapper_options->prior_position_max_error_sigma = settings.max_error_sigma;
  mapper_options->prior_position_max_error_m = settings.max_error_m;
  mapper_options->pose_priors_in_world = position_priors.priors;
}

}  // namespace

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  using Observation = std::pair<image_t, point2D_t>;
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  for (const auto image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.xy);
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  auto corr_graph = database_cache_->CorrespondenceGraph();

  // Union all matching observations.
  UnionFind<Observation> uf;
  FeatureMatches matches;
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    corr_graph->ExtractMatchesBetweenImages(image_id1, image_id2, matches);
    for (const auto& match : matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  std::unordered_map<Observation, std::vector<Observation>> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, and collect valid ones with lengths.
  std::unordered_map<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  for (const auto& [track_id, observations] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options.track_intra_image_consistency_threshold *
              options.track_intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.track.AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images < static_cast<size_t>(options.track_min_num_views_per_track))
      continue;

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  LOG(INFO) << "Kept " << candidate_points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and select for problem.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  std::unordered_map<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  for (const auto& [track_length, point3D_id] : track_lengths) {
    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.track.Elements().begin(),
        point3D.track.Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options.track_required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.track.Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count == static_cast<size_t>(options.track_required_tracks_per_view))
        --images_left;
      ++count;
    }

    // Add track after updating counts so we can move.
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before filtering: " << candidate_points3D.size()
            << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angular_reproj_error_deg,
                                     double max_normalized_reproj_error,
                                     double min_tri_angle_deg) {
  if (!RunGlobalPositioning(options, *pose_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  ObservationManager obs_manager(*reconstruction_);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angular_reproj_error_deg,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<std::pair<image_t, point2D_t>> obs_to_delete;
  for (const auto point3D_id : reconstruction_->Point3DIds()) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      if (!camera.has_prior_focal_length) {
        continue;
      }
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (error > max_angular_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : obs_to_delete) {
    if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(image_id, point2D_idx);
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  // Normalize the structure for numerical stability.
  // Skipped under position priors: the priors have made the model metric,
  // and Normalize() would rescale exactly that away.
  if (!UsePositionPriors()) {
    reconstruction_->Normalize();
  }

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage) {
  for (int ite = 0; ite < num_iterations; ite++) {
    // Optional fixed-rotation stage: optimize positions only
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      if (!RunBundleAdjustment(opts_position_only, *reconstruction_)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << num_iterations << ", fixed-rotation stage finished";
    }

    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      if (!RunBundleAdjustment(options, *reconstruction_)) {
        return false;
      }
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << " finished";

    // Normalize the structure for numerical stability.
    // Skipped under position priors: the priors have made the model metric,
    // and Normalize() would rescale exactly that away.
    if (!UsePositionPriors()) {
      reconstruction_->Normalize();
    }

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    LOG(INFO) << "Filtering tracks by reprojection ...";

    ObservationManager obs_manager(*reconstruction_);
    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          reconstruction_->Point3DIds(),
          ReprojectionErrorType::NORMALIZED);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation
  LOG(INFO) << "Filtering tracks by reprojection ...";
  {
    ObservationManager obs_manager(*reconstruction_);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        reconstruction_->Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
  }

  // Iterative global refinement.
  //
  // On the full path the priors go through AdjustGlobalBundle's existing
  // pose-prior adjuster (the same one `pose_prior_mapper` uses), which aligns
  // the model to the priors and therefore leaves it metric. Normalization is
  // skipped in that case for the same reason the incremental mapper skips it:
  // it would rescale the metric solution away.
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  ApplyPositionPriorOptions(position_priors_, &mapper_options);
  const bool use_prior_position = mapper_options.use_prior_position;
  if (use_prior_position) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      if (position_priors_.Find(image_id) != nullptr) {
        prior_constrained_image_ids_.insert(image_id);
      }
    }
  }
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/
                                   !use_prior_position);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  if (!RunBundleAdjustment(ba_options, *reconstruction_)) {
    return false;
  }

  // Normalize the structure for numerical stability.
  // Skipped under position priors: the priors have made the model metric,
  // and Normalize() would rescale exactly that away.
  if (!UsePositionPriors()) {
    reconstruction_->Normalize();
  }

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions opts = InitializeOptions(options);

  // Run rotation averaging
  if (!opts.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(opts.rotation_averaging)) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!opts.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(opts);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!opts.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(opts.global_positioning,
                           opts.max_angular_reproj_error_deg,
                           opts.max_normalized_reproj_error,
                           opts.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Bundle adjustment
  if (!opts.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(opts.bundle_adjustment,
                                   opts.max_normalized_reproj_error,
                                   opts.min_tri_angle_deg,
                                   opts.ba_num_iterations,
                                   opts.ba_skip_fixed_rotation_stage,
                                   opts.ba_skip_joint_optimization_stage)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!opts.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(opts.retriangulation,
                                         opts.bundle_adjustment,
                                         opts.max_normalized_reproj_error,
                                         opts.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  return true;
}

// ── GlobalMapper::LoadPriorPoses ────────────────────────────────────────────

void GlobalMapper::LoadPriorPoses(const class Reconstruction& prior_reconstruction) {
  DCHECK(reconstruction_ != nullptr)
      << "Call BeginReconstruction() before LoadPriorPoses().";

  prior_image_ids_.clear();
  size_t loaded = 0;

  // Import the prior's refined camera intrinsics. The database may only
  // hold coarse initial values (e.g. a focal prior from the field of view);
  // the prior reconstruction's intrinsics were bundle-adjusted and are
  // consistent with the prior poses and 3D points imported below. Without
  // this, the windowed solve optimizes refined geometry against unrefined
  // intrinsics and warps the model.
  size_t loaded_cameras = 0;
  for (const auto& [camera_id, prior_camera] : prior_reconstruction.Cameras()) {
    if (!reconstruction_->ExistsCamera(camera_id)) continue;
    struct Camera& camera = reconstruction_->Camera(camera_id);
    if (camera.model_id != prior_camera.model_id ||
        camera.width != prior_camera.width ||
        camera.height != prior_camera.height) {
      LOG(WARNING) << "LoadPriorPoses: camera " << camera_id
                   << " differs in model/size between prior and database; "
                      "keeping database intrinsics.";
      continue;
    }
    camera.params = prior_camera.params;
    camera.has_prior_focal_length = true;
    ++loaded_cameras;
  }

  for (const auto& [image_id, prior_image] : prior_reconstruction.Images()) {
    if (!prior_image.HasPose()) continue;
    if (!reconstruction_->ExistsImage(image_id)) {
      // A registered prior image that is missing from the database cache
      // points at upstream database corruption (e.g. an image-deletion bug
      // in the producing service) — do not hide it.
      LOG(WARNING) << "LoadPriorPoses: prior image id " << image_id << " ('"
                   << prior_image.Name()
                   << "') is registered in the prior reconstruction but "
                      "missing from the database cache; skipping it.";
      continue;
    }

    Image& image = reconstruction_->Image(image_id);
    image.FramePtr()->SetCamFromWorld(image.CameraId(), prior_image.CamFromWorld());
    reconstruction_->RegisterFrame(image.FrameId());
    prior_image_ids_.insert(image_id);
    ++loaded;
  }

  // Keep a sample of prior 3D points for the post-bootstrap cheirality check.
  prior_points_sample_.clear();
  constexpr size_t kMaxSamplePoints = 1000;
  const size_t num_points = prior_reconstruction.NumPoints3D();
  const size_t stride = std::max<size_t>(1, num_points / kMaxSamplePoints);
  size_t point_idx = 0;
  for (const auto& [point3D_id, point3D] : prior_reconstruction.Points3D()) {
    if (point_idx++ % stride == 0) {
      prior_points_sample_.push_back(point3D.xyz);
    }
  }

  LOG(INFO) << "LoadPriorPoses: seeded " << loaded
            << " image(s) and imported intrinsics for " << loaded_cameras
            << " camera(s) from prior reconstruction.";
}

// ── SolvePoseFromPriorEdges ──────────────────────────────────────────────────────

std::optional<Rigid3d> SolvePoseFromPriorEdges(
    const std::vector<BootstrapEdgeObservation>& all_observations,
    const double max_candidate_deg,
    const std::optional<BootstrapGravityGate>& gravity_gate) {
  if (all_observations.empty()) return std::nullopt;

  // ── Pass 0: gravity gate ──────────────────────────────────────────────
  //
  // A rotation candidate implies a camera-frame gravity direction
  // R_candidate * g_world. Candidates that disagree with the measured
  // gravity beyond the threshold are outliers regardless of their weight —
  // discard the whole observation (its translation ray depends on the same
  // bad relative pose).
  std::vector<BootstrapEdgeObservation> gated_observations;
  const std::vector<BootstrapEdgeObservation>* observations_ptr =
      &all_observations;
  if (gravity_gate.has_value()) {
    const Eigen::Vector3d g_meas =
        gravity_gate->gravity_in_new_cam.normalized();
    const Eigen::Vector3d g_world = gravity_gate->gravity_in_world.normalized();
    const double max_dot_angle_rad =
        gravity_gate->max_error_deg * M_PI / 180.0;

    for (const BootstrapEdgeObservation& obs : all_observations) {
      const Eigen::Quaterniond R_prior = obs.prior_cam_from_world.rotation();
      const Eigen::Quaterniond R_rel = obs.cam2_from_cam1.rotation();
      const Eigen::Quaterniond R_new_cand =
          obs.prior_is_cam1 ? (R_rel * R_prior).normalized()
                            : (R_rel.inverse() * R_prior).normalized();
      const Eigen::Vector3d g_implied = R_new_cand * g_world;
      const double angle = std::acos(
          std::clamp(g_implied.dot(g_meas), -1.0, 1.0));
      if (angle <= max_dot_angle_rad) {
        gated_observations.push_back(obs);
      }
    }

    if (gated_observations.empty()) {
      LOG(WARNING)
          << "SolvePoseFromPriorEdges: all " << all_observations.size()
          << " candidate(s) violate the gravity prior by more than "
          << gravity_gate->max_error_deg
          << " deg; proceeding without the gravity gate (measured gravity "
             "may be unreliable).";
    } else {
      if (gated_observations.size() < all_observations.size()) {
        LOG(INFO) << "SolvePoseFromPriorEdges: gravity gate discarded "
                  << all_observations.size() - gated_observations.size()
                  << " of " << all_observations.size() << " candidate(s).";
      }
      observations_ptr = &gated_observations;
    }
  }
  const std::vector<BootstrapEdgeObservation>& observations =
      *observations_ptr;

  // Rigid3d convention:
  //   p_cam2 = R_rel * p_cam1 + t_rel
  //   R_rel  = R_cam2 * R_cam1^{-1}
  //   t_rel  = position of cam1's world-origin in cam2's frame
  //            (unit-norm from essential-matrix decomposition; direction ok)

  // ── Pass 1: rotation candidates + translation ray bundle ─────────────

  std::vector<Eigen::Quaterniond> rot_candidates;
  std::vector<double> rot_weights;
  rot_candidates.reserve(observations.size());
  rot_weights.reserve(observations.size());

  // Rays are stored as (origin, t_rel_unit, prior_is_cam1, R_prior); the
  // world-space direction is derived in Pass 2 from the final mean rotation.
  struct Ray {
    Eigen::Vector3d origin;      // C_prior in world space
    Eigen::Vector3d t_rel_unit;  // normalized edge translation
    bool prior_is_cam1;
    Eigen::Matrix3d R_prior;     // rotation of prior_cam_from_world
    double weight;
  };
  std::vector<Ray> rays;
  rays.reserve(observations.size());

  for (const BootstrapEdgeObservation& obs : observations) {
    const Eigen::Quaterniond R_prior = obs.prior_cam_from_world.rotation();

    // ── Rotation candidate ───────────────────────────────────────────────────
    //
    //   pair (prior=cam1, new=cam2):  R_rel = R_new * R_prior^{-1}
    //                                 ⟹  R_new = R_rel * R_prior
    //
    //   pair (new=cam1, prior=cam2):  R_rel = R_prior * R_new^{-1}
    //                                 ⟹  R_new = R_rel^{-1} * R_prior
    const Eigen::Quaterniond R_rel = obs.cam2_from_cam1.rotation();
    const Eigen::Quaterniond R_new_cand =
        obs.prior_is_cam1 ? (R_rel * R_prior).normalized()
                          : (R_rel.inverse() * R_prior).normalized();

    rot_candidates.push_back(R_new_cand);
    rot_weights.push_back(obs.weight);

    // ── Ray for translation LS ─────────────────────────────────────────────
    //
    // A near-zero relative translation (pure-rotation edge) contributes no
    // usable ray; the rotation candidate above is still kept.
    const Eigen::Vector3d t_rel_raw = obs.cam2_from_cam1.translation();
    const double t_norm = t_rel_raw.norm();
    if (t_norm < 1e-9) continue;

    Ray ray;
    ray.origin = Inverse(obs.prior_cam_from_world).translation();
    ray.t_rel_unit = t_rel_raw / t_norm;
    ray.prior_is_cam1 = obs.prior_is_cam1;
    ray.R_prior = R_prior.toRotationMatrix();
    ray.weight = obs.weight;
    rays.push_back(std::move(ray));
  }

  // ── Pass 2a: rotation via weighted Karcher mean on SO(3) ─────────────

  const int best_idx = BestCandidateIdxByWeight(rot_weights);
  Eigen::Quaterniond q_mean = KarcherMeanSO3(
      rot_candidates, rot_weights, rot_candidates[best_idx]);

  // Outlier pass: discard candidates more than max_candidate_deg from the
  // current mean, then recompute the mean over the inliers.
  if (rot_candidates.size() > 1) {
    const double max_rad = max_candidate_deg * M_PI / 180.0;

    std::vector<Eigen::Quaterniond> inlier_rots;
    std::vector<double> inlier_rot_w;
    for (size_t i = 0; i < rot_candidates.size(); ++i) {
      Eigen::Quaterniond dq = (q_mean.inverse() * rot_candidates[i]).normalized();
      if (dq.w() < 0.0) dq.coeffs() = -dq.coeffs();
      const double angle = 2.0 * std::acos(std::clamp(std::abs(dq.w()), 0.0, 1.0));
      if (angle <= max_rad) {
        inlier_rots.push_back(rot_candidates[i]);
        inlier_rot_w.push_back(rot_weights[i]);
      }
    }
    if (!inlier_rots.empty() && inlier_rots.size() < rot_candidates.size()) {
      const int bi = BestCandidateIdxByWeight(inlier_rot_w);
      q_mean = KarcherMeanSO3(inlier_rots, inlier_rot_w, inlier_rots[bi]);
    }
  }

  const Eigen::Matrix3d R_new_mat = q_mean.toRotationMatrix();
  const Eigen::Matrix3d R_new_mat_T = R_new_mat.transpose();

  // ── Pass 2b: translation via weighted multi-ray least squares ─────────
  //
  // World-space ray direction from C_prior toward C_new:
  //
  //   pair (prior=cam1, new=cam2):
  //     t_rel = C_prior in new-cam frame.
  //     direction prior→new  =  −R_new^T * t̂_rel
  //     (uses the final mean rotation, so all rays are consistent with the
  //      averaged pose rather than their per-edge candidates)
  //
  //   pair (new=cam1, prior=cam2):
  //     t_rel = C_new in prior-cam frame.
  //     direction prior→new  =  +R_prior^T * t̂_rel
  //
  // Each ray (o_i, d_i, w_i) contributes to:
  //   A  +=  w_i * (I − d_i d_i^T)
  //   b  +=  w_i * (I − d_i d_i^T) * o_i
  //
  // Solve A * C_new = b  for the world-space camera centre C_new.

  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b_vec = Eigen::Vector3d::Zero();

  for (const Ray& ray : rays) {
    Eigen::Vector3d d = ray.prior_is_cam1
                            ? Eigen::Vector3d(-(R_new_mat_T * ray.t_rel_unit))
                            : Eigen::Vector3d(ray.R_prior.transpose() *
                                              ray.t_rel_unit);
    d.normalize();

    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d * d.transpose();
    A += ray.weight * P;
    b_vec += ray.weight * P * ray.origin;
  }

  Eigen::Vector3d C_new;
  if (rays.empty() || std::abs(A.determinant()) < 1e-9) {
    // Near-singular: rays are nearly parallel (e.g. a single ray, or all
    // priors colinear with the new camera) or no usable rays at all.  Fall
    // back to the weighted centroid of the prior centres.  Global
    // positioning will correct this.
    LOG(WARNING) << "SolvePoseFromPriorEdges: near-singular ray system; "
                    "using prior-centre centroid as translation fallback.";
    C_new = Eigen::Vector3d::Zero();
    double w_sum = 0.0;
    for (const BootstrapEdgeObservation& obs : observations) {
      C_new += obs.weight * Inverse(obs.prior_cam_from_world).translation();
      w_sum += obs.weight;
    }
    C_new /= w_sum;
  } else {
    C_new = A.colPivHouseholderQr().solve(b_vec);
  }

  // Convert world centre to COLMAP's cam_from_world translation:
  //   t = -R_new * C_new
  Rigid3d pose;
  pose.rotation() = q_mean;
  pose.translation() = -(R_new_mat * C_new);
  return pose;
}

// ── GlobalMapper::BootstrapNewImagePoses ────────────────────────────────────

std::unordered_set<image_t> GlobalMapper::BootstrapNewImagePoses(
    const GlobalMapperOptions& options) {
  DCHECK(reconstruction_ != nullptr)
      << "Call BeginReconstruction() before BootstrapNewImagePoses().";
  DCHECK(!prior_image_ids_.empty())
      << "Call LoadPriorPoses() before BootstrapNewImagePoses().";

  // Guard against the upstream signature drift where
  // PoseGraph::Edge::cam2_from_cam1 becomes std::optional<Rigid3d>: fail
  // loudly at compile time after a rebase instead of invoking UB.
  static_assert(
      std::is_same_v<decltype(PoseGraph::Edge::cam2_from_cam1), Rigid3d>,
      "PoseGraph::Edge::cam2_from_cam1 is assumed to be a plain Rigid3d. "
      "If it is std::optional<Rigid3d> on this base, guard the accesses "
      "below with has_value().");

  // Walk the PoseGraph and collect, for every unregistered image, all valid
  // edges that connect it to a prior (registered) image.
  std::unordered_map<image_t, std::vector<BootstrapEdgeObservation>> evidence;

  for (const auto& [pair_id, rel_pose] : pose_graph_->Edges()) {
    if (!rel_pose.valid) continue;
    if (rel_pose.num_matches < options.bootstrap_min_inliers) continue;

    const auto [id1, id2] = PairIdToImagePair(pair_id);

    const bool id1_prior = prior_image_ids_.count(id1) > 0;
    const bool id2_prior = prior_image_ids_.count(id2) > 0;
    if (id1_prior == id2_prior) continue;  // both or neither — skip

    const image_t prior_id = id1_prior ? id1 : id2;
    const image_t new_id = id1_prior ? id2 : id1;

    if (!reconstruction_->ExistsImage(new_id)) continue;
    if (reconstruction_->Image(new_id).HasPose()) continue;

    BootstrapEdgeObservation obs;
    obs.cam2_from_cam1 = rel_pose.cam2_from_cam1;
    obs.prior_is_cam1 = id1_prior;
    obs.prior_cam_from_world = reconstruction_->Image(prior_id).CamFromWorld();
    obs.weight = static_cast<double>(rel_pose.num_matches);
    evidence[new_id].push_back(std::move(obs));
  }

  // Gravity gate setup: collect per-image gravity priors and derive the
  // world gravity direction from the prior images' priors and poses.
  std::unordered_map<image_t, Eigen::Vector3d> image_to_gravity;
  std::optional<Eigen::Vector3d> world_gravity;
  if (options.bootstrap_max_gravity_error_deg > 0) {
    world_gravity = CollectGravityPriors(&image_to_gravity);
  }

  // Solve each new image's pose from its accumulated evidence.
  std::unordered_set<image_t> bootstrapped;

  for (const auto& [new_id, observations] : evidence) {
    std::optional<BootstrapGravityGate> gravity_gate;
    if (world_gravity.has_value()) {
      const auto it = image_to_gravity.find(new_id);
      if (it != image_to_gravity.end()) {
        BootstrapGravityGate gate;
        gate.gravity_in_new_cam = it->second;
        gate.gravity_in_world = *world_gravity;
        gate.max_error_deg = options.bootstrap_max_gravity_error_deg;
        gravity_gate = gate;
      }
    }

    const std::optional<Rigid3d> pose = SolvePoseFromPriorEdges(
        observations, options.bootstrap_max_candidate_deg, gravity_gate);
    if (!pose.has_value()) continue;

    Image& new_image = reconstruction_->Image(new_id);
    new_image.FramePtr()->SetCamFromWorld(new_image.CameraId(), *pose);
    reconstruction_->RegisterFrame(new_image.FrameId());
    bootstrapped.insert(new_id);

    LOG(INFO) << "BootstrapNewImagePoses: image " << new_id << " — pose from "
              << observations.size() << " prior edge(s).";

    // Cheap cheirality sanity check: a plausibly-posed camera should see a
    // reasonable fraction of the prior scene points in front of it.  Warn
    // only — Solve() may still fix a marginal pose.
    if (!prior_points_sample_.empty()) {
      size_t num_in_front = 0;
      for (const Eigen::Vector3d& point : prior_points_sample_) {
        if ((*pose * point).z() > 0) ++num_in_front;
      }
      const double frac =
          static_cast<double>(num_in_front) / prior_points_sample_.size();
      if (frac < 0.1) {
        LOG(WARNING) << "BootstrapNewImagePoses: only "
                     << 100.0 * frac << "% of sampled prior 3D points have "
                     << "positive depth in bootstrapped image " << new_id
                     << "; the pose may be unreliable.";
      }
    }
  }

  if (bootstrapped.empty()) {
    LOG(WARNING)
        << "BootstrapNewImagePoses: no new images could be bootstrapped. "
           "Ensure the new image has verified matches against prior images "
           "and that bootstrap_min_inliers is not too high.";
  }

  return bootstrapped;
}

// ── GlobalMapper::ImportPriorPoints3D ────────────────────────────────

bool GlobalMapper::ImportPriorPoints3D(
    const GlobalMapperOptions& options,
    const class Reconstruction& prior_reconstruction,
    IncrementalAddLedger* ledger) {
  THROW_CHECK_NOTNULL(reconstruction_);

  // 1. Import the prior 3D points, re-linking their 2D-3D associations.
  //
  //    On a healthy prior this import is *lossless*: every drop path below
  //    signals that the prior reconstruction and the database disagree.
  //    - missing image / bad point2D_idx: upstream corruption, the same
  //      signal LoadPriorPoses already warns about;
  //    - already linked: unreachable for a consistent prior, since we import
  //      into an empty point set -- it needs two prior points claiming one
  //      observation;
  //    - short track: only ever a consequence of the three above.
  //
  //    So we count each reason separately and refuse to hide any of them. A
  //    silent haircut here compounds across every subsequent add; failing the
  //    add instead leaves the last good reconstruction in place.
  IncrementalAddLedger local_ledger;
  IncrementalAddLedger& acc = (ledger != nullptr) ? *ledger : local_ledger;
  acc.prior_points_total = prior_reconstruction.NumPoints3D();

  std::set<image_t> missing_image_ids;
  std::set<image_t> bad_idx_image_ids;
  for (const auto& [point3D_id, prior_point3D] :
       prior_reconstruction.Points3D()) {
    struct Point3D point3D;
    point3D.xyz = prior_point3D.xyz;
    point3D.color = prior_point3D.color;
    point3D.error = prior_point3D.error;
    for (const TrackElement& track_el : prior_point3D.track.Elements()) {
      if (!reconstruction_->ExistsImage(track_el.image_id)) {
        ++acc.dropped_missing_image;
        missing_image_ids.insert(track_el.image_id);
        continue;
      }
      const Image& image = reconstruction_->Image(track_el.image_id);
      if (track_el.point2D_idx >= image.NumPoints2D()) {
        ++acc.dropped_bad_point2D_idx;
        bad_idx_image_ids.insert(track_el.image_id);
        continue;
      }
      if (image.Point2D(track_el.point2D_idx).HasPoint3D()) {
        ++acc.dropped_already_linked;
        continue;
      }
      point3D.track.AddElement(track_el);
    }
    if (point3D.track.Length() < 2) {
      ++acc.dropped_short_track;
      continue;
    }
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));
    ++acc.imported;
  }

  if (acc.DroppedAtImport() > 0) {
    LOG(ERROR) << "ImportPriorPoints3D: prior structure import was "
                  "lossy. This means the prior reconstruction and the "
                  "database disagree; it is not normal wear.";
    if (acc.dropped_missing_image > 0) {
      std::ostringstream ids;
      for (const image_t id : missing_image_ids) ids << ' ' << id;
      LOG(ERROR) << "  " << acc.dropped_missing_image
                 << " track element(s) reference image(s) missing from the "
                    "database cache:"
                 << ids.str();
    }
    if (acc.dropped_bad_point2D_idx > 0) {
      std::ostringstream ids;
      for (const image_t id : bad_idx_image_ids) ids << ' ' << id;
      LOG(ERROR) << "  " << acc.dropped_bad_point2D_idx
                 << " track element(s) have an out-of-range keypoint index; "
                    "keypoints changed under the prior for image(s):"
                 << ids.str();
    }
    if (acc.dropped_already_linked > 0) {
      LOG(ERROR) << "  " << acc.dropped_already_linked
                 << " track element(s) target a Point2D already claimed by "
                    "another prior point (inconsistent prior).";
    }
    if (acc.dropped_short_track > 0) {
      LOG(ERROR) << "  " << acc.dropped_short_track
                 << " prior point(s) lost entirely (< 2 views left).";
    }
    if (!options.allow_lossy_prior_import) {
      LOG(ERROR) << "Failing this add rather than writing a degraded model. "
                    "Pass --allow_lossy_prior_import to import anyway.";
      return false;
    }
    LOG(WARNING) << "allow_lossy_prior_import is set; continuing with "
                 << acc.imported << " of " << acc.prior_points_total
                 << " prior point(s).";
  }

  LOG(INFO) << "ImportPriorPoints3D: imported " << acc.imported << " of "
            << acc.prior_points_total << " prior 3D point(s).";
  return true;
}

// ── GlobalMapper::CollectGravityPriors ─────────────────────────

std::optional<Eigen::Vector3d> GlobalMapper::CollectGravityPriors(
    std::unordered_map<image_t, Eigen::Vector3d>* image_to_gravity) const {
  THROW_CHECK_NOTNULL(image_to_gravity);
  image_to_gravity->clear();

  for (const PosePrior& pose_prior : database_cache_->PosePriors()) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        pose_prior.HasGravity()) {
      (*image_to_gravity)[pose_prior.corr_data_id.id] =
          pose_prior.gravity.normalized();
    }
  }

  // Average the prior images' measured gravity, rotated into the world frame
  // by their known poses: g_world = R_cam_from_world^T * g_cam.
  Eigen::Vector3d world_gravity_sum = Eigen::Vector3d::Zero();
  size_t num_prior_gravities = 0;
  for (const image_t prior_id : prior_image_ids_) {
    const auto it = image_to_gravity->find(prior_id);
    if (it == image_to_gravity->end()) continue;
    if (!reconstruction_->ExistsImage(prior_id)) continue;
    if (!reconstruction_->Image(prior_id).HasPose()) continue;
    const Eigen::Quaterniond R_prior =
        reconstruction_->Image(prior_id).CamFromWorld().rotation();
    world_gravity_sum += R_prior.inverse() * it->second;
    ++num_prior_gravities;
  }
  if (num_prior_gravities > 0 && world_gravity_sum.norm() > 1e-6) {
    return world_gravity_sum.normalized();
  }
  return std::nullopt;
}

// ── GlobalMapper::RegisterNewImagesByPnP ────────────────────────

bool GlobalMapper::ComputeWorldPositionPriors(
    const GlobalMapperOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  position_priors_ = WorldPositionPriors();

  if (!options.use_prior_position) return false;

  if (database_cache_->PosePriors().empty()) {
    LOG(WARNING) << "Position priors requested but the database has none; "
                    "solving without them.";
    return false;
  }

  WorldPositionPriorOptions prior_options;
  prior_options.fallback_stddev = options.prior_position_fallback_stddev;
  prior_options.alignment_ransac_options.random_seed = options.random_seed;
  position_priors_ = CollectWorldPositionPriors(
      *database_cache_, *reconstruction_, prior_options);

  position_priors_.settings.use_robust_loss =
      options.use_robust_loss_on_prior_position;
  position_priors_.settings.loss_scale = options.prior_position_loss_scale;
  position_priors_.settings.fallback_stddev =
      options.prior_position_fallback_stddev;
  position_priors_.settings.max_error_sigma =
      options.prior_position_max_error_sigma;
  position_priors_.settings.max_error_m = options.prior_position_max_error_m;

  return !position_priors_.Empty();
}

std::unordered_set<image_t> GlobalMapper::RegisterNewImagesByPnP(
    const GlobalMapperOptions& options, IncrementalAddLedger* ledger) {
  THROW_CHECK_NOTNULL(reconstruction_);

  std::unordered_set<image_t> registered;

  if (reconstruction_->NumPoints3D() == 0) {
    LOG(WARNING) << "RegisterNewImagesByPnP: no prior structure imported; "
                    "nothing to match against. Falling back to bootstrap.";
    return registered;
  }

  // Sorted for determinism: RegisterNextImage mutates the model (it adds the
  // inlier observations), so registration order is observable.
  std::vector<image_t> candidates;
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    if (!image.HasPose()) candidates.push_back(image_id);
  }
  std::sort(candidates.begin(), candidates.end());
  if (candidates.empty()) return registered;

  // TearDown() (called by EndReconstruction) erases every image without a
  // pose, which is exactly the set the bootstrap fallback still needs, and
  // can also drop cameras belonging to rigs left with no registered frame.
  // Snapshot the refined intrinsics so a re-Load cannot quietly replace them
  // with the database's coarse values.
  std::unordered_map<camera_t, struct Camera> camera_snapshot;
  for (const auto& [camera_id, camera] : reconstruction_->Cameras()) {
    camera_snapshot.emplace(camera_id, camera);
  }

  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  IncrementalMapper::Options mapper_options;
  mapper_options.num_threads = options.num_threads;
  mapper_options.random_seed = options.random_seed;
  // Never let a single-view PnP move the shared camera's intrinsics. The
  // incremental path imports the prior's bundle-adjusted intrinsics, which
  // are estimated from the whole model; re-fitting them against one image's
  // correspondences would be strictly worse information.
  mapper_options.abs_pose_refine_focal_length = false;
  mapper_options.abs_pose_refine_extra_params = false;
  mapper_options.abs_pose_min_num_inliers = options.pnp_min_num_inliers;

  // Supply the known vertical, when the database carries gravity priors, so
  // absolute pose estimation can use the 2-point upright solver: 4 unknowns
  // instead of 6 means the few correspondences a weakly-connected image has
  // constrain the pose far more tightly. RefineAbsolutePose still polishes
  // without the constraint, so IMU error does not enter the output pose.
  // Position priors gate the registration: a pose that disagrees with its
  // GNSS fix is declined here rather than admitted and left to displace every
  // frame registered after it. See plan-6 item 1.
  ApplyPositionPriorOptions(position_priors_, &mapper_options);
  if (mapper_options.use_prior_position) {
    LOG(INFO) << "RegisterNewImagesByPnP: position-prior acceptance gate "
                 "active over "
              << mapper_options.pose_priors_in_world.size()
              << " prior(s) at " << mapper_options.prior_position_max_error_sigma
              << " sigma / " << mapper_options.prior_position_max_error_m
              << " m.";
  }

  std::unordered_map<image_t, Eigen::Vector3d> image_to_gravity;
  mapper_options.gravity_in_world = CollectGravityPriors(&image_to_gravity);
  mapper_options.gravity_uncertainty_deg = options.gravity_uncertainty_deg;
  if (mapper_options.gravity_in_world.has_value()) {
    mapper_options.gravity_in_cam = std::move(image_to_gravity);
    LOG(INFO) << "RegisterNewImagesByPnP: using gravity-constrained absolute "
                 "pose (up2p) with "
              << mapper_options.gravity_in_cam.size()
              << " image gravity prior(s).";
  }

  for (const image_t image_id : candidates) {
    const std::string& name = reconstruction_->Image(image_id).Name();
    if (mapper.RegisterNextImage(mapper_options, image_id)) {
      registered.insert(image_id);
      LOG(INFO) << "RegisterNewImagesByPnP: registered image " << image_id
                << " ('" << name << "') with "
                << reconstruction_->Image(image_id).NumPoints3D()
                << " observation(s).";
    } else {
      // Not an error: declining is the point. A weakly-connected image gets
      // handed to the bootstrap rather than being given a guessed pose.
      LOG(WARNING) << "RegisterNewImagesByPnP: absolute pose estimation "
                      "declined image "
                   << image_id << " ('" << name
                   << "'); leaving it for the bootstrap fallback.";
    }
  }
  mapper.EndReconstruction(/*discard=*/false);

  // Restore the images TearDown removed, then the refined intrinsics.
  reconstruction_->Load(*database_cache_);
  for (const auto& [camera_id, camera] : camera_snapshot) {
    if (reconstruction_->ExistsCamera(camera_id)) {
      reconstruction_->Camera(camera_id).params = camera.params;
    }
  }

  if (ledger != nullptr) {
    ledger->pnp_registered += registered.size();
  }
  LOG(INFO) << "RegisterNewImagesByPnP: registered " << registered.size()
            << " of " << candidates.size() << " new image(s) by PnP.";
  return registered;
}

// ── GlobalMapper::SolveIncrementalWindowed ─────────────────────────────────

bool GlobalMapper::SolveIncrementalWindowed(
    const GlobalMapperOptions& options,
    const class Reconstruction& prior_reconstruction,
    const std::unordered_set<image_t>& new_image_ids,
    const std::unordered_set<image_t>& constant_image_ids,
    IncrementalAddLedger* ledger) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_GT(options.optimize_window_size, 0);
  THROW_CHECK(!new_image_ids.empty());

  const GlobalMapperOptions opts = InitializeOptions(options);

  // 0. Seed the new images' camera intrinsics from refined prior cameras of
  //    identical model and size. The database typically only carries a
  //    coarse focal prior (e.g. from the field of view); starting local
  //    bundle adjustment 30-40% off in focal length biases the refined
  //    poses and the error compounds across adds. With a same-device
  //    capture loop the refined prior intrinsics are a far better initial
  //    value. (LoadPriorPoses already imported the prior cameras' params
  //    for the prior images themselves.)
  for (const image_t new_id : new_image_ids) {
    if (!reconstruction_->ExistsImage(new_id)) continue;
    struct Camera& camera = *reconstruction_->Image(new_id).CameraPtr();
    if (prior_reconstruction.ExistsCamera(camera.camera_id)) continue;
    std::vector<double> focals;
    const struct Camera* donor = nullptr;
    for (const auto& [prior_camera_id, prior_camera] :
         prior_reconstruction.Cameras()) {
      if (prior_camera.model_id == camera.model_id &&
          prior_camera.width == camera.width &&
          prior_camera.height == camera.height) {
        focals.push_back(prior_camera.MeanFocalLength());
        donor = &prior_camera;
      }
    }
    if (donor == nullptr) continue;
    // Use the params of the prior camera with the median mean focal.
    std::nth_element(
        focals.begin(), focals.begin() + focals.size() / 2, focals.end());
    const double median_focal = focals[focals.size() / 2];
    for (const auto& [prior_camera_id, prior_camera] :
         prior_reconstruction.Cameras()) {
      if (prior_camera.model_id == camera.model_id &&
          prior_camera.width == camera.width &&
          prior_camera.height == camera.height &&
          prior_camera.MeanFocalLength() == median_focal) {
        donor = &prior_camera;
        break;
      }
    }
    camera.params = donor->params;
    LOG(INFO) << "SolveIncrementalWindowed: seeded intrinsics of new camera "
              << camera.camera_id << " from prior camera "
              << donor->camera_id << " (focal "
              << donor->MeanFocalLength() << ").";
  }

  // 1. Import the prior 3D points (unless a caller already did, e.g. to
  //    give PnP registration something to match against).
  IncrementalAddLedger local_ledger;
  IncrementalAddLedger& acc = (ledger != nullptr) ? *ledger : local_ledger;
  if (reconstruction_->NumPoints3D() == 0) {
    if (!ImportPriorPoints3D(options, prior_reconstruction, &acc)) {
      return false;
    }
  }


  // 2. Triangulate the new image(s) against the prior structure and run
  //    iterative local bundle adjustment with a covisibility window. Poses
  //    outside the window enter the problem as constant-pose residuals
  //    only, so the prior frame is preserved by construction.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  IncrementalMapper::Options mapper_options;
  mapper_options.ba_local_num_images = opts.optimize_window_size;
  mapper_options.num_threads = opts.num_threads;
  mapper_options.random_seed = opts.random_seed;
  for (const image_t image_id : constant_image_ids) {
    if (reconstruction_->ExistsImage(image_id)) {
      mapper_options.constant_frames.insert(
          reconstruction_->Image(image_id).FrameId());
    }
  }

  // The windowed add's whole point is that it is the live path; without this
  // it is also the one path that cannot use GNSS. The priors are already in
  // this reconstruction's frame, so local bundle adjustment can add them as
  // position residuals on the images the window leaves free -- no alignment,
  // no normalization, and therefore no movement of the prior model's gauge.
  // See plan-6 item 1.
  ApplyPositionPriorOptions(position_priors_, &mapper_options);
  if (mapper_options.use_prior_position) {
    LOG(INFO) << "SolveIncrementalWindowed: constraining the window with "
              << mapper_options.pose_priors_in_world.size()
              << " position prior(s).";
  }

  BundleAdjustmentOptions ba_options = opts.bundle_adjustment;
  ba_options.print_summary = false;
  if (ba_options.ceres) {
    ba_options.ceres->solver_options.num_threads = opts.num_threads;
  }

  IncrementalTriangulator::Options tri_options = opts.retriangulation;

  for (const image_t image_id : new_image_ids) {
    THROW_CHECK(reconstruction_->Image(image_id).HasPose())
        << "New image " << image_id << " must be bootstrapped before the "
        << "windowed solve.";
    acc.triangulated += mapper.TriangulateImage(tri_options, image_id);
    mapper.IterativeLocalRefinement(/*max_num_refinements=*/2,
                                    /*max_refinement_change=*/0.001,
                                    mapper_options,
                                    ba_options,
                                    tri_options,
                                    image_id);
  }

  // Recover observations that local filtering stripped from prior points on
  // this or an earlier add. A filtered observation is not destroyed -- the
  // feature match is still in the database -- so it can be re-attached now
  // that the poses have been refined. IterativeLocalRefinement only completes
  // the *modified* points; this sweeps all of them, which is cheap and is
  // what stops a drip-fed session from bleeding structure.
  acc.completed += mapper.CompleteTracks(tri_options);
  acc.merged += mapper.MergeTracks(tri_options);

  // Record which poses the priors actually pulled on, for the residual
  // report's `used_in_ba` column (plan-6 item 2).
  const std::unordered_set<image_t>& prior_constrained =
      mapper.PriorConstrainedImageIds();
  prior_constrained_image_ids_.insert(prior_constrained.begin(),
                                      prior_constrained.end());

  mapper.EndReconstruction(/*discard=*/false);

  acc.points_out = reconstruction_->NumPoints3D();
  acc.mean_reproj_error = reconstruction_->ComputeMeanReprojectionError();
  acc.mean_track_length = reconstruction_->ComputeMeanTrackLength();

  LOG(INFO) << "SolveIncrementalWindowed: triangulated " << acc.triangulated
            << " observation(s) for " << new_image_ids.size()
            << " new image(s); completed " << acc.completed << ", merged "
            << acc.merged << "; " << acc.points_out
            << " point(s) out; window size " << opts.optimize_window_size
            << ".";
  return true;
}

// ── GlobalMapper::RefreshIntrinsicsGlobally ────────────────────────────────

size_t GlobalMapper::RefreshIntrinsicsGlobally(
    const GlobalMapperOptions& options, IncrementalAddLedger* ledger) {
  THROW_CHECK_NOTNULL(reconstruction_);

  const GlobalMapperOptions opts = InitializeOptions(options);
  const std::vector<image_t> reg_image_ids = reconstruction_->RegImageIds();
  if (reg_image_ids.empty()) return 0;

  // 1. Intrinsics-only global bundle adjustment.
  //
  //    Everything except the camera parameters is held constant, so this is
  //    a ~4-8 DOF problem no matter how large N grows: the cost is residual
  //    evaluation, not the Schur complement. The gauge is fixed by
  //    construction (all poses and points constant), so no gauge fixing is
  //    needed or wanted.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  BundleAdjustmentOptions ba_options = opts.bundle_adjustment;
  ba_options.refine_points3D = false;
  ba_options.refine_rig_from_world = false;
  ba_options.refine_sensor_from_rig = false;
  ba_options.refine_focal_length = true;
  ba_options.refine_extra_params = true;
  ba_options.print_summary = false;
  if (ba_options.ceres) {
    ba_options.ceres->solver_options.num_threads = opts.num_threads;
  }

  const double before_error = reconstruction_->ComputeMeanReprojectionError();
  std::vector<double> before_focals;
  for (const auto& [camera_id, camera] : reconstruction_->Cameras()) {
    before_focals.push_back(camera.MeanFocalLength());
  }

  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction_);
  bundle_adjuster->Solve();

  std::ostringstream focal_change;
  size_t focal_idx = 0;
  for (const auto& [camera_id, camera] : reconstruction_->Cameras()) {
    focal_change << " cam" << camera_id << ": " << before_focals[focal_idx++]
                 << " -> " << camera.MeanFocalLength();
  }
  LOG(INFO) << "RefreshIntrinsicsGlobally: intrinsics-only BA over "
            << reg_image_ids.size() << " image(s);" << focal_change.str();

  // 2. Recover structure that was filtered away while the intrinsics were
  //    wrong. This is the operation that only becomes possible once the
  //    global parameter is corrected, which is why it is paired with the BA
  //    rather than run on its own schedule.
  const size_t points_before = reconstruction_->NumPoints3D();
  const size_t obs_before = reconstruction_->ComputeNumObservations();

  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);
  const IncrementalTriangulator::Options tri_options = opts.retriangulation;
  const size_t num_retriangulated = mapper.Retriangulate(tri_options);
  const size_t num_completed = mapper.CompleteTracks(tri_options);
  const size_t num_merged = mapper.MergeTracks(tri_options);
  mapper.EndReconstruction(/*discard=*/false);

  const size_t obs_after = reconstruction_->ComputeNumObservations();
  const size_t recovered =
      (obs_after > obs_before) ? (obs_after - obs_before) : 0;

  LOG(INFO) << "RefreshIntrinsicsGlobally: mean reproj error " << before_error
            << " -> " << reconstruction_->ComputeMeanReprojectionError()
            << "; points " << points_before << " -> "
            << reconstruction_->NumPoints3D() << "; retriangulated "
            << num_retriangulated << ", completed " << num_completed
            << ", merged " << num_merged << "; " << recovered
            << " observation(s) recovered.";

  if (ledger != nullptr) {
    ledger->recovered += recovered;
    ledger->completed += num_completed;
    ledger->merged += num_merged;
    ledger->intrinsics_refreshed = true;
    ledger->points_out = reconstruction_->NumPoints3D();
    ledger->mean_reproj_error = reconstruction_->ComputeMeanReprojectionError();
    ledger->mean_track_length = reconstruction_->ComputeMeanTrackLength();
  }
  return recovered;
}

}  // namespace colmap
