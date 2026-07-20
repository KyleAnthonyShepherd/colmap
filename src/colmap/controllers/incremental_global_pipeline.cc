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

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

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

// ── Anchor-based gauge handling ────────────────────────────────────────────

std::vector<image_t> SelectAnchorImages(const Reconstruction& reconstruction,
                                        const int num_anchors) {
  THROW_CHECK_GT(num_anchors, 0);

  // Candidates sorted by number of 3D points (descending): well-observed
  // images make stable anchors.
  std::vector<std::pair<size_t, image_t>> candidates;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    candidates.emplace_back(reconstruction.Image(image_id).NumPoints3D(),
                            image_id);
  }
  std::sort(candidates.begin(), candidates.end(), std::greater<>());

  // Greedy max-min-distance selection for wide mutual baselines.
  std::vector<image_t> anchors;
  std::vector<Eigen::Vector3d> anchor_centers;
  for (int round = 0; round < num_anchors && !candidates.empty(); ++round) {
    int best_idx = -1;
    double best_min_dist = -1.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Eigen::Vector3d center =
          reconstruction.Image(candidates[i].second).ProjectionCenter();
      double min_dist = std::numeric_limits<double>::max();
      for (const Eigen::Vector3d& anchor_center : anchor_centers) {
        min_dist = std::min(min_dist, (center - anchor_center).norm());
      }
      if (anchor_centers.empty()) {
        // First anchor: the best-observed image.
        best_idx = 0;
        break;
      }
      if (min_dist > best_min_dist) {
        best_min_dist = min_dist;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0) break;
    const image_t image_id = candidates[best_idx].second;
    anchors.push_back(image_id);
    anchor_centers.push_back(
        reconstruction.Image(image_id).ProjectionCenter());
    candidates.erase(candidates.begin() + best_idx);
  }
  return anchors;
}

bool ReadAnchors(const std::filesystem::path& path, AnchorSet* anchor_set) {
  THROW_CHECK_NOTNULL(anchor_set);
  anchor_set->anchors.clear();
  std::ifstream file(path);
  if (!file.is_open()) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    image_t image_id;
    Eigen::Vector3d center;
    if (!(ss >> image_id >> center.x() >> center.y() >> center.z())) {
      LOG(WARNING) << "ReadAnchors: unparsable line in " << path << ": '"
                   << line << "'";
      anchor_set->anchors.clear();
      return false;
    }
    anchor_set->anchors.emplace_back(image_id, center);
  }
  return !anchor_set->anchors.empty();
}

void WriteAnchors(const std::filesystem::path& path,
                  const AnchorSet& anchor_set) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK(file.is_open()) << "Cannot write anchors to " << path;
  file << "# Gauge anchors: image_id cx cy cz (fixed reference centres)\n";
  file << std::setprecision(17);
  for (const auto& [image_id, center] : anchor_set.anchors) {
    file << image_id << " " << center.x() << " " << center.y() << " "
         << center.z() << "\n";
  }
}

bool RealignToAnchors(const AnchorSet& anchor_set,
                      Reconstruction* reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  for (const auto& [image_id, stored_center] : anchor_set.anchors) {
    if (!reconstruction->ExistsImage(image_id) ||
        !reconstruction->Image(image_id).HasPose()) {
      continue;
    }
    src.push_back(reconstruction->Image(image_id).ProjectionCenter());
    tgt.push_back(stored_center);
  }
  if (src.size() < 3) {
    LOG(WARNING) << "RealignToAnchors: only " << src.size()
                 << " anchor(s) registered in the reconstruction; need 3.";
    return false;
  }
  Sim3d anchors_from_output;
  if (!EstimateSim3d(src, tgt, anchors_from_output)) {
    LOG(WARNING) << "RealignToAnchors: Sim3 estimation failed.";
    return false;
  }
  reconstruction->Transform(anchors_from_output);
  LOG(INFO) << "Realigned output to " << src.size()
            << " fixed anchor(s). Scale factor: "
            << anchors_from_output.scale();
  return true;
}

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

    // Select gauge anchors for the first successful reconstruction so that
    // subsequent incremental adds have a fixed realignment reference.
    anchors_.anchors.clear();
    for (const image_t image_id : SelectAnchorImages(*reconstruction)) {
      anchors_.anchors.emplace_back(
          image_id, reconstruction->Image(image_id).ProjectionCenter());
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

  // 7. Load the gauge anchors persisted next to the prior reconstruction,
  //    if any (see AnchorSet).
  AnchorSet prior_anchors;
  const bool have_prior_anchors = ReadAnchors(
      options_.prior_reconstruction_path / "anchors.txt", &prior_anchors);
  if (have_prior_anchors) {
    LOG(INFO) << "Loaded " << prior_anchors.anchors.size()
              << " gauge anchor(s) from prior reconstruction.";
  }

  // 8. Solve. When optimize_window_size > 0 run a windowed local solve
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
    // Anchor poses stay constant even when they fall inside the window, so
    // the gauge is fixed by construction.
    if (!mapper.SolveIncrementalWindowed(mapper_opts,
                                         prior_reconstruction,
                                         bootstrapped,
                                         prior_anchors.ImageIds())) {
      LOG(ERROR) << "Windowed incremental solve failed.";
      return;
    }
  } else {
    mapper.Solve(mapper_opts);
  }
  LOG(INFO) << "Incremental global solve done in "
            << run_timer.ElapsedSeconds() << " s";

  // 9. Rig-scale alignment (matches standard GlobalPipeline behaviour).
  //    Skipped for windowed solves, which keep the prior scales.
  if (!windowed) {
    AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                       reconstruction.get());
  }

  // 10. Realign full-solve output back to the prior coordinate frame.
  //     Preferred: Sim3 onto the FIXED anchor centres (no rolling-drift
  //     compounding). Fallback: legacy rolling Sim3 over all prior centres.
  //     Windowed solves never leave the prior frame, so realignment is
  //     unnecessary there.
  if (!windowed && mapper_opts.realign_to_prior_after_solve) {
    const bool anchor_aligned =
        have_prior_anchors &&
        RealignToAnchors(prior_anchors, reconstruction.get());

    if (!anchor_aligned && prior_reg_ids.size() >= 3) {
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
        if (EstimateSim3d(
                new_prior_centres, prior_centres, prior_from_output)) {
          reconstruction->Transform(prior_from_output);
          LOG(INFO) << "Realigned output to prior frame (rolling).  "
                       "Scale factor: " << prior_from_output.scale();
        } else {
          LOG(WARNING)
              << "Realignment: Sim3 estimation failed; output may be in a "
                 "different scale/frame than the prior reconstruction.";
        }
      } else {
        LOG(WARNING)
            << "Realignment skipped: fewer than 3 prior images are "
               "registered in the output reconstruction.";
      }
    }
  }

  // 11. Propagate or create the gauge anchors to persist with the output:
  //     keep the prior anchors as the fixed reference when they exist,
  //     otherwise select fresh ones from this output.
  if (have_prior_anchors) {
    anchors_ = prior_anchors;
  } else {
    anchors_.anchors.clear();
    for (const image_t image_id : SelectAnchorImages(*reconstruction)) {
      anchors_.anchors.emplace_back(
          image_id, reconstruction->Image(image_id).ProjectionCenter());
    }
    LOG(INFO) << "Selected " << anchors_.anchors.size()
              << " new gauge anchor(s).";
  }

  // 12. Write output.
  Reconstruction& out =
      *reconstruction_manager_->Get(reconstruction_manager_->Add());
  out = *reconstruction;

  if (!options_.image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    out.ExtractColorsForAllImages(options_.image_path);
  }
}

}  // namespace colmap
