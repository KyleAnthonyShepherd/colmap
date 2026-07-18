// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// (Same BSD licence header as other COLMAP files.)

#include "colmap/controllers/incremental_global_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {

namespace {

// Warns when too few cameras have prior focal lengths (copied from
// global_pipeline.cc).
constexpr double kMinPriorFocalLengthRatio = 0.5;

bool HasInsufficientPriorFocalLengths(const DatabaseCache& database_cache) {
  const auto& cameras = database_cache.Cameras();
  if (cameras.empty()) return false;
  const size_t num_with_prior =
      std::count_if(cameras.begin(), cameras.end(), [](const auto& camera) {
        return camera.second.has_prior_focal_length;
      });
  return num_with_prior < kMinPriorFocalLengthRatio * cameras.size();
}

void WarnInsufficientPriorFocalLengths() {
  LOG(WARNING)
      << "Less than " << kMinPriorFocalLengthRatio * 100
      << "% of cameras have prior focal lengths. The global mapper depends on "
         "reasonably good focal length priors. Consider running "
         "'colmap view_graph_calibrator' first.";
}

}  // namespace

// ── Constructor ─────────────────────────────────────────────────────────────

IncrementalGlobalPipeline::IncrementalGlobalPipeline(
    IncrementalGlobalPipelineOptions options,
    std::shared_ptr<Database> database,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {
  THROW_CHECK_NOTNULL(database);

  DatabaseCache::Options cache_opts;
  cache_opts.min_num_matches = options_.min_num_matches;
  cache_opts.ignore_watermarks = options_.ignore_watermarks;
  cache_opts.image_names = {options_.image_names.begin(),
                             options_.image_names.end()};

  database_cache_ = DatabaseCache::Create(*database, cache_opts);

  if (options_.decompose_relative_pose) {
    MaybeDecomposeRelativePoses(database_cache_.get());
  }
}

// ── Run ─────────────────────────────────────────────────────────────────────

void IncrementalGlobalPipeline::Run() {
  const bool has_prior_path = !options_.prior_reconstruction_path.empty();

  if (!has_prior_path) {
    // Fall back: behave exactly like a standard GlobalPipeline run.
    LOG(INFO) << "IncrementalGlobalPipeline: no prior path set, running "
                 "standard global reconstruction.";
    auto reconstruction = std::make_shared<Reconstruction>();
    GlobalMapperOptions mapper_opts = options_.mapper;
    mapper_opts.image_path = options_.image_path;
    mapper_opts.num_threads = options_.num_threads;
    mapper_opts.random_seed = options_.random_seed;

    GlobalMapper mapper(database_cache_);
    mapper.BeginReconstruction(reconstruction);

    Timer t;
    t.Start();
    mapper.Solve(mapper_opts);
    LOG(INFO) << "Reconstruction done in " << t.ElapsedSeconds() << " s";

    AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                       reconstruction.get());

    Reconstruction& out =
        *reconstruction_manager_->Get(reconstruction_manager_->Add());
    out = *reconstruction;
    if (!options_.image_path.empty()) {
      out.ExtractColorsForAllImages(options_.image_path);
    }
    return;
  }

  // ── Incremental-add path ───────────────────────────────────────────────

  // 1. Load the prior reconstruction.
  LOG(INFO) << "IncrementalGlobalPipeline: loading prior reconstruction from "
            << options_.prior_reconstruction_path;
  Reconstruction prior_reconstruction;
  prior_reconstruction.Read(options_.prior_reconstruction_path);

  THROW_CHECK_GT(prior_reconstruction.NumRegImages(), 0)
      << "Prior reconstruction contains no registered images.";

  LOG(INFO) << "Prior reconstruction: " << prior_reconstruction.NumRegImages()
            << " registered images, " << prior_reconstruction.NumPoints3D()
            << " 3D points.";

  // 2. Collect the prior camera centres for the post-solve realignment.
  //    We'll use these to build a Sim3 that undoes any drift introduced by
  //    global positioning / bundle adjustment.
  const std::vector<image_t> prior_reg_ids =
      prior_reconstruction.RegImageIds();

  std::vector<Eigen::Vector3d> prior_centres;
  prior_centres.reserve(prior_reg_ids.size());
  for (const image_t id : prior_reg_ids) {
    prior_centres.push_back(
        prior_reconstruction.Image(id).ProjectionCenter());
  }

  // 3. Optionally warn about focal lengths.
  const bool warn_focal = HasInsufficientPriorFocalLengths(*database_cache_);
  if (warn_focal) WarnInsufficientPriorFocalLengths();

  // 4. Build mapper options: force skip_rotation_averaging since we are
  //    providing bootstrapped rotations.
  GlobalMapperOptions mapper_opts = options_.mapper;
  mapper_opts.image_path = options_.image_path;
  mapper_opts.num_threads = options_.num_threads;
  mapper_opts.random_seed = options_.random_seed;
  mapper_opts.skip_rotation_averaging = true;

  // 5. Create the mapper, initialise, and inject prior poses.
  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction);

  mapper.LoadPriorPoses(prior_reconstruction);

  LOG(INFO) << "Prior images loaded: " << mapper.PriorImageIds().size();

  // 6. Bootstrap rotations for every unregistered image (should be 1 here).
  const std::unordered_set<image_t> bootstrapped =
      mapper.BootstrapNewImagePoses(mapper_opts);

  if (bootstrapped.empty()) {
    LOG(ERROR)
        << "IncrementalGlobalPipeline: failed to bootstrap any new image "
           "rotation.  Ensure the new image has verified matches against at "
           "least one prior image in the database and that "
           "bootstrap_min_inliers is not too high.";
    return;
  }

  LOG(INFO) << "Bootstrapped " << bootstrapped.size() << " new image(s): ";
  for (const image_t id : bootstrapped) {
    LOG(INFO) << "  image_id=" << id;
  }

  // 7. Solve. When optimize_window_size > 0 run a windowed local solve
  //    (O(window) per add, output stays in the prior frame by
  //    construction), except every full_solve_interval-th image where a
  //    full global solve acts as a periodic refresh.
  const bool windowed =
      mapper_opts.optimize_window_size > 0 &&
      !(mapper_opts.full_solve_interval > 0 &&
        reconstruction->NumRegImages() % mapper_opts.full_solve_interval == 0);

  Timer run_timer;
  run_timer.Start();
  if (windowed) {
    LOG(INFO) << "Running windowed incremental solve (window size "
              << mapper_opts.optimize_window_size << ").";
    if (!mapper.SolveIncrementalWindowed(
            mapper_opts, prior_reconstruction, bootstrapped)) {
      LOG(ERROR) << "Windowed incremental solve failed.";
      return;
    }
  } else {
    mapper.Solve(mapper_opts);
  }
  LOG(INFO) << "Incremental global solve done in "
            << run_timer.ElapsedSeconds() << " s";

  // 8. Rig-scale alignment (matches standard GlobalPipeline behaviour).
  //    Skipped for windowed solves, which keep the prior scales.
  if (!windowed) {
    AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                       reconstruction.get());
  }

  // 9. Optional: re-align the full output to the prior coordinate frame.
  //    Global positioning may have drifted the prior cameras slightly.
  //    We compute a Sim3 from (output prior centres) → (original prior
  //    centres) and apply it to the whole reconstruction.
  //    Windowed solves never leave the prior frame, so realignment is
  //    unnecessary there.
  if (!windowed && mapper_opts.realign_to_prior_after_solve &&
      prior_reg_ids.size() >= 3) {
    // Collect the output positions of the prior images.
    std::vector<Eigen::Vector3d> new_prior_centres;
    new_prior_centres.reserve(prior_reg_ids.size());
    size_t missing = 0;
    for (const image_t id : prior_reg_ids) {
      if (reconstruction->Image(id).HasPose()) {
        new_prior_centres.push_back(
            reconstruction->Image(id).ProjectionCenter());
      } else {
        ++missing;
      }
    }

    if (missing > 0) {
      LOG(WARNING) << "Realignment: " << missing
                   << " prior image(s) not registered in output.";
    }

    if (new_prior_centres.size() >= 3) {
      Sim3d prior_from_output;
      if (EstimateSim3d(new_prior_centres, prior_centres, prior_from_output)) {
        reconstruction->Transform(prior_from_output);
        LOG(INFO) << "Realigned output to prior frame.  "
                     "Scale factor: " << prior_from_output.scale();
      } else {
        LOG(WARNING)
            << "Realignment: Sim3 estimation failed; output may be in a "
               "different scale/frame than the prior reconstruction.";
      }
    } else {
      LOG(WARNING)
          << "Realignment skipped: fewer than 3 prior images are registered "
             "in the output reconstruction.";
    }
  }

  // 10. Write output.
  Reconstruction& out =
      *reconstruction_manager_->Get(reconstruction_manager_->Add());
  out = *reconstruction;

  if (!options_.image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    out.ExtractColorsForAllImages(options_.image_path);
  }
}

}  // namespace colmap
