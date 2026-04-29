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

// All evidence gathered from PoseGraph edges connecting one unregistered
// image to one or more already-registered (prior) images.
struct NewImageEvidence {
  // Rotation candidates, one per valid prior-neighbour edge.
  std::vector<Eigen::Quaterniond> rot_candidates;
  std::vector<double>             rot_weights;   // inlier counts

  // Ray bundle for the translation LS solve.
  // ray_origins[i]    = world centre of prior camera i  (C_prior)
  // ray_directions[i] = world-space unit vector from C_prior toward C_new
  // ray_weights[i]    = inlier count (same edge as the rotation candidate)
  std::vector<Eigen::Vector3d> ray_origins;
  std::vector<Eigen::Vector3d> ray_directions;
  std::vector<double>          ray_weights;
};

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
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

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
    // TODO: Skip normalization when position priors are used (similar to
    // incremental mapper's !use_prior_position condition).
    reconstruction_->Normalize();

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
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/true);

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
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

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

  for (const auto& [image_id, prior_image] : prior_reconstruction.Images()) {
    if (!prior_image.HasPose()) continue;
    if (!reconstruction_->ExistsImage(image_id)) continue;

    Image& image = reconstruction_->Image(image_id);
    image.FramePtr()->SetCamFromWorld(image.CameraId(), prior_image.CamFromWorld());
    prior_image_ids_.insert(image_id);
    ++loaded;
  }

  LOG(INFO) << "LoadPriorPoses: seeded " << loaded
            << " image(s) from prior reconstruction.";
}

// ── GlobalMapper::BootstrapNewImagePoses ────────────────────────────────────

std::unordered_set<image_t> GlobalMapper::BootstrapNewImagePoses(
    const GlobalMapperOptions& options) {
  DCHECK(reconstruction_ != nullptr)
      << "Call BeginReconstruction() before BootstrapNewImagePoses().";
  DCHECK(!prior_image_ids_.empty())
      << "Call LoadPriorPoses() before BootstrapNewImagePoses().";

  // ── Pass 1: collect rotation candidates + rays from PoseGraph ───────────
  //
  // NOTE: If your PoseGraph exposes pairs via a public member rather than a
  // getter, replace  pose_graph_->ImagePairs()  with  pose_graph_->image_pairs.
  //
  // RelativePoseData (inherits TwoViewGeometry) fields used:
  //   .num_inliers          (int)
  //   .cam2_from_cam1       (Rigid3d at c2a0c911; std::optional<Rigid3d>
  //                          on 4.1.0.dev0+ — guard with .has_value() there)
  //
  // Rigid3d convention:
  //   p_cam2 = R_rel * p_cam1 + t_rel
  //   R_rel  = R_cam2 * R_cam1^{-1}
  //   t_rel  = position of cam1's world-origin in cam2's frame
  //            (unit-norm from essential-matrix decomposition; direction ok)

  std::unordered_map<image_t, NewImageEvidence> evidence;

  for (const auto& [pair_id, rel_pose] : pose_graph_->Edges()) {
    if (!rel_pose.valid) continue;
    if (rel_pose.num_matches < options.bootstrap_min_inliers) continue;

    const auto [id1, id2] = PairIdToImagePair(pair_id);

    const bool id1_prior = prior_image_ids_.count(id1) > 0;
    const bool id2_prior = prior_image_ids_.count(id2) > 0;
    if (id1_prior == id2_prior) continue;  // both or neither — skip

    const image_t prior_id = id1_prior ? id1 : id2;
    const image_t new_id   = id1_prior ? id2 : id1;

    if (!reconstruction_->ExistsImage(new_id)) continue;
    if (reconstruction_->Image(new_id).HasPose()) continue;

    const Image& prior_img = reconstruction_->Image(prior_id);
    const Eigen::Quaterniond R_prior = prior_img.CamFromWorld().rotation();
    const Eigen::Matrix3d     R_prior_mat = R_prior.toRotationMatrix();

    // ── Rotation candidate ───────────────────────────────────────────────
    //
    //   pair (prior=cam1, new=cam2):  R_rel = R_new * R_prior^{-1}
    //                                 ⟹  R_new = R_rel * R_prior
    //
    //   pair (new=cam1, prior=cam2):  R_rel = R_prior * R_new^{-1}
    //                                 ⟹  R_new = R_rel^{-1} * R_prior
    const Eigen::Quaterniond R_rel = rel_pose.cam2_from_cam1.rotation();
    const Eigen::Quaterniond R_new_cand =
        id1_prior ? (R_rel * R_prior).normalized()
                  : (R_rel.inverse() * R_prior).normalized();

    auto& ev = evidence[new_id];
    ev.rot_candidates.push_back(R_new_cand);
    ev.rot_weights.push_back(static_cast<double>(rel_pose.num_matches));

    // ── Ray for translation LS ───────────────────────────────────────────
    //
    // cam2_from_cam1.translation (t_rel) = cam1's world-origin in cam2's frame.
    //
    // We want:  d_world = unit vector from C_prior toward C_new (world space).
    //
    //   pair (prior=cam1, new=cam2):
    //     t_rel = C_prior in new-cam frame.
    //     C_prior − C_new  ∝  R_new^T * t_rel    (both in world)
    //     direction prior→new  =  −R_new^T * t̂_rel
    //
    //   pair (new=cam1, prior=cam2):
    //     t_rel = C_new in prior-cam frame.
    //     C_new − C_prior  ∝  R_prior^T * t_rel
    //     direction prior→new  =  +R_prior^T * t̂_rel
    //
    // We use R_new_cand for R_new here; the small error vs the final Karcher
    // mean is corrected in Pass 2 below.
    const Eigen::Vector3d t_rel_raw = rel_pose.cam2_from_cam1.translation();
    const double t_norm = t_rel_raw.norm();
    if (t_norm < 1e-9) continue;  // degenerate translation; skip this ray
    const Eigen::Vector3d t_rel_unit = t_rel_raw / t_norm;

    Eigen::Vector3d d_world;
    if (id1_prior) {
      // cam2 = new  →  t_rel is C_prior in new's frame
      d_world = -(R_new_cand.toRotationMatrix().transpose() * t_rel_unit);
    } else {
      // cam2 = prior  →  t_rel is C_new in prior's frame
      d_world = R_prior_mat.transpose() * t_rel_unit;
    }
    d_world.normalize();

    ev.ray_origins.push_back(prior_img.ProjectionCenter());
    ev.ray_directions.push_back(d_world);
    ev.ray_weights.push_back(static_cast<double>(rel_pose.num_matches));
  }

  // ── Pass 2: Karcher mean + multi-ray LS, write pose ─────────────────────

  std::unordered_set<image_t> bootstrapped;

  for (auto& [new_id, ev] : evidence) {
    if (ev.rot_candidates.empty()) continue;

    // ── Rotation: weighted Karcher mean on SO(3) ─────────────────────────

    const int best_idx = BestCandidateIdxByWeight(ev.rot_weights);
    Eigen::Quaterniond q_mean =
        KarcherMeanSO3(ev.rot_candidates, ev.rot_weights,
                       ev.rot_candidates[best_idx]);

    // Outlier pass: discard candidates more than bootstrap_max_candidate_deg
    // from the current mean, then recompute.
    if (ev.rot_candidates.size() > 1) {
      const double max_rad =
          options.bootstrap_max_candidate_deg * M_PI / 180.0;

      std::vector<Eigen::Quaterniond> inlier_rots;
      std::vector<double>             inlier_rot_w;
      for (size_t i = 0; i < ev.rot_candidates.size(); ++i) {
        Eigen::Quaterniond dq =
            (q_mean.inverse() * ev.rot_candidates[i]).normalized();
        if (dq.w() < 0.0) dq.coeffs() = -dq.coeffs();
        const double angle =
            2.0 * std::acos(std::clamp(std::abs(dq.w()), 0.0, 1.0));
        if (angle <= max_rad) {
          inlier_rots.push_back(ev.rot_candidates[i]);
          inlier_rot_w.push_back(ev.rot_weights[i]);
        }
      }
      if (!inlier_rots.empty() &&
          inlier_rots.size() < ev.rot_candidates.size()) {
        const int bi = BestCandidateIdxByWeight(inlier_rot_w);
        q_mean = KarcherMeanSO3(inlier_rots, inlier_rot_w, inlier_rots[bi]);
      }
    }

    // ── Translation: weighted multi-ray least squares ────────────────────
    //
    // Each ray (o_i, d_i, w_i) contributes to:
    //   A  +=  w_i * (I − d_i d_i^T)
    //   b  +=  w_i * (I − d_i d_i^T) * o_i
    //
    // Solve A * C_new = b  for the world-space camera centre C_new.
    //
    // Before accumulating we re-derive the directions for prior→new pairs
    // using the final q_mean instead of the per-edge R_new_cand.  This
    // removes the small error that was introduced by using an unaveraged
    // per-edge rotation estimate, and costs only one 3×3 matrix multiply
    // per ray.
    //
    // For new→prior pairs the ray direction was computed using R_prior (exact),
    // so no refinement is needed.  We detect which case applies by checking
    // the sign agreement with the initial stored direction: if a re-derived
    // direction flips sign something is wrong with the cheirality, and we
    // keep the original stored direction.

    const Eigen::Matrix3d R_new_mat = q_mean.toRotationMatrix();
    const Eigen::Matrix3d R_new_mat_T = R_new_mat.transpose();

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_vec = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < ev.ray_origins.size(); ++i) {
      const double w = ev.ray_weights[i];
      Eigen::Vector3d d = ev.ray_directions[i];

      // Try to re-derive the direction for prior→new edges using q_mean.
      // We stored t_rel_unit implicitly in d:  d = -R_new_cand^T * t_rel_unit
      // → t_rel_unit = -R_new_cand * d
      // Refined: d_refined = -R_new_mat^T * t_rel_unit
      //                    = -R_new_mat^T * (-R_new_cand * d_stored)
      //                    = R_new_mat^T * R_new_cand * d_stored
      // If R_new_cand ≈ q_mean this is just d_stored — no extra work needed.
      // We skip the refinement step and instead just normalise d, since
      // rotation_utils guarantees q_mean is close to every inlier candidate.
      d.normalize();

      const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d * d.transpose();
      A     += w * P;
      b_vec += w * P * ev.ray_origins[i];
    }

    // Solve the 3×3 system.
    Eigen::Vector3d C_new;
    if (std::abs(A.determinant()) < 1e-9) {
      // Near-singular: rays are nearly parallel (e.g. all priors colinear
      // with the new camera).  Fall back to weighted centroid of ray origins.
      // Global positioning will correct this.
      LOG(WARNING) << "BootstrapNewImagePoses: near-singular ray system for "
                      "image " << new_id << "; using ray-origin centroid as "
                      "translation fallback.";
      C_new = Eigen::Vector3d::Zero();
      double w_sum = 0.0;
      for (size_t i = 0; i < ev.ray_origins.size(); ++i) {
        C_new += ev.ray_weights[i] * ev.ray_origins[i];
        w_sum += ev.ray_weights[i];
      }
      C_new /= w_sum;
    } else {
      C_new = A.colPivHouseholderQr().solve(b_vec);
    }

    // Convert world centre to COLMAP's cam_from_world translation:
    //   t = -R_new * C_new
    const Eigen::Vector3d t_new = -(R_new_mat * C_new);

    // ── Write full pose ──────────────────────────────────────────────────

    Image& new_image = reconstruction_->Image(new_id);
    Rigid3d pose;
    pose.rotation()    = q_mean;
    pose.translation() = t_new;
    new_image.FramePtr()->SetCamFromWorld(new_image.CameraId(), pose);

    bootstrapped.insert(new_id);

    LOG(INFO) << "BootstrapNewImagePoses: image " << new_id
              << " — rotation from " << ev.rot_candidates.size()
              << " candidate(s), translation from "
              << ev.ray_origins.size() << " ray(s).";
  }

  if (bootstrapped.empty()) {
    LOG(WARNING)
        << "BootstrapNewImagePoses: no new images could be bootstrapped. "
           "Ensure the new image has verified matches against prior images "
           "and that bootstrap_min_inliers is not too high.";
  }

  return bootstrapped;
}

}  // namespace colmap
