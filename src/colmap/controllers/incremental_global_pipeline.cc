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
#include "colmap/sfm/prior_positions.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

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

// ── Incremental ledger schema ──────────────────────────────────────────────

constexpr const char* kLedgerHeader =
    "add\tpath\tprior_points\timported\tdrop_missing_image\t"
    "drop_bad_idx\tdrop_already_linked\tdrop_short_track\t"
    "triangulated\tmerged\tcompleted\trecovered\tpoints_out\t"
    "mean_reproj\tmean_track_len\tintrinsics_refreshed\t"
    "prior_residuals\tprior_rms\tprior_worst\tprior_worst_sigma\t"
    "prior_worst_image";

size_t CountColumns(const std::string& line) {
  return static_cast<size_t>(std::count(line.begin(), line.end(), '\t')) + 1;
}

void StripCarriageReturn(std::string* line) {
  if (!line->empty() && line->back() == '\r') line->pop_back();
}

// Copies a ledger written by a previous add into `path`, widening it to the
// current schema. Rows from an older build are padded with empty fields --
// empty rather than zero, because the column did not exist then and a zero
// would read as a measurement.
void CarryForwardLedger(const std::filesystem::path& prior_path,
                        const std::filesystem::path& path) {
  std::ifstream in(prior_path);
  if (!in.is_open()) {
    LOG(WARNING) << "AppendIncrementalLedger: could not read " << prior_path;
    return;
  }
  std::string header;
  if (!std::getline(in, header)) {
    // Empty prior ledger: nothing to carry, the fresh header is written by
    // the caller.
    return;
  }
  StripCarriageReturn(&header);

  std::ofstream out(path);
  if (!out.is_open()) {
    LOG(WARNING) << "AppendIncrementalLedger: could not write " << path;
    return;
  }

  const size_t num_columns = CountColumns(kLedgerHeader);
  if (CountColumns(header) > num_columns) {
    // A ledger from a *newer* build. Copying it unchanged keeps its data
    // readable; the rows this build appends will simply be narrower.
    LOG(WARNING) << "AppendIncrementalLedger: " << prior_path
                 << " has more columns than this build writes; carrying it "
                    "forward unchanged.";
    out << header << "\n" << in.rdbuf();
    return;
  }

  out << kLedgerHeader << "\n";
  std::string line;
  while (std::getline(in, line)) {
    StripCarriageReturn(&line);
    if (line.empty()) continue;
    out << line;
    for (size_t i = CountColumns(line); i < num_columns; ++i) out << "\t";
    out << "\n";
  }
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

// ── Cross-add incremental state ────────────────────────────────────────────

bool ReadIncrementalState(const std::filesystem::path& path,
                          IncrementalState* state) {
  THROW_CHECK_NOTNULL(state);
  *state = IncrementalState();
  std::ifstream file(path);
  if (!file.is_open()) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string key;
    if (!(ss >> key)) continue;
    bool ok = true;
    if (key == "num_adds") {
      ok = static_cast<bool>(ss >> state->num_adds);
    } else if (key == "stable_adds") {
      ok = static_cast<bool>(ss >> state->stable_adds);
    } else if (key == "intrinsics_converged") {
      int value = 0;
      ok = static_cast<bool>(ss >> value);
      state->intrinsics_converged = (value != 0);
    } else if (key == "reproj_ema_full") {
      ok = static_cast<bool>(ss >> state->reproj_ema_full);
    } else if (key == "reproj_ema_windowed") {
      ok = static_cast<bool>(ss >> state->reproj_ema_windowed);
    } else if (key == "last_refresh_add") {
      ok = static_cast<bool>(ss >> state->last_refresh_add);
    } else if (key == "camera") {
      camera_t camera_id = 0;
      double mean_focal = 0.0;
      ok = static_cast<bool>(ss >> camera_id >> mean_focal);
      if (ok) state->camera_mean_focal[camera_id] = mean_focal;
    } else {
      LOG(WARNING) << "ReadIncrementalState: unknown key '" << key << "' in "
                   << path << "; ignoring.";
      continue;
    }
    if (!ok) {
      LOG(WARNING) << "ReadIncrementalState: unparsable line in " << path
                   << ": '" << line
                   << "'; discarding state (treated as not converged).";
      *state = IncrementalState();
      return false;
    }
  }
  return true;
}

void WriteIncrementalState(const std::filesystem::path& path,
                           const IncrementalState& state) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK(file.is_open()) << "Cannot write incremental state to " << path;
  file << "# Cross-add incremental state. Delete this file to force the\n"
          "# pipeline back to full solves until intrinsics re-converge.\n";
  file << std::setprecision(17);
  file << "num_adds " << state.num_adds << "\n";
  file << "stable_adds " << state.stable_adds << "\n";
  file << "intrinsics_converged " << (state.intrinsics_converged ? 1 : 0)
       << "\n";
  file << "reproj_ema_full " << state.reproj_ema_full << "\n";
  file << "reproj_ema_windowed " << state.reproj_ema_windowed << "\n";
  file << "last_refresh_add " << state.last_refresh_add << "\n";
  for (const auto& [camera_id, mean_focal] : state.camera_mean_focal) {
    file << "camera " << camera_id << " " << mean_focal << "\n";
  }
}

void AppendIncrementalLedger(const std::filesystem::path& path,
                             const std::filesystem::path& prior_path,
                             const int add_index,
                             const char* path_label,
                             const IncrementalAddLedger& ledger) {
  // Carry the prior session's history forward, since each add writes into a
  // fresh output directory that the server then swaps into place. A ledger
  // written by an older build has fewer columns, so it is migrated rather
  // than copied: appending wider rows to a narrower file would produce a
  // ragged TSV that the site pipeline cannot parse.
  if (!std::filesystem::exists(path) && std::filesystem::exists(prior_path)) {
    CarryForwardLedger(prior_path, path);
  }

  const bool need_header = !std::filesystem::exists(path);
  std::ofstream file(path, std::ios::app);
  THROW_CHECK(file.is_open()) << "Cannot write ledger to " << path;
  if (need_header) {
    file << kLedgerHeader << "\n";
  }
  file << add_index << "\t" << path_label << "\t"
       << ledger.prior_points_total << "\t" << ledger.imported << "\t"
       << ledger.dropped_missing_image << "\t" << ledger.dropped_bad_point2D_idx
       << "\t" << ledger.dropped_already_linked << "\t"
       << ledger.dropped_short_track << "\t" << ledger.triangulated << "\t"
       << ledger.merged << "\t" << ledger.completed << "\t"
       << ledger.recovered << "\t" << ledger.points_out << "\t"
       << ledger.mean_reproj_error << "\t" << ledger.mean_track_length << "\t"
       << (ledger.intrinsics_refreshed ? 1 : 0) << "\t"
       << ledger.prior_residuals << "\t" << ledger.prior_residual_rms << "\t"
       << ledger.prior_residual_worst << "\t"
       << ledger.prior_residual_worst_sigma << "\t"
       << (ledger.prior_residual_worst_image == kInvalidImageId
               ? std::string("")
               : std::to_string(ledger.prior_residual_worst_image))
       << "\n";
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
  // WGS84 priors are useless to a metric solver as latitude/longitude, so
  // convert them to Cartesian ENU on load, exactly as the incremental
  // pipeline does. Priors already written as CARTESIAN (which is what the
  // site pipeline does, deliberately) pass through untouched.
  cache_opts.convert_pose_priors_to_enu = options_.mapper.use_prior_position;

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

    // Report prior residuals even on this path. This solve does not use the
    // priors as a constraint (that is `pose_prior_mapper`'s job), so every
    // residual is reported with used_in_ba = 0 -- which is exactly the
    // measurement that says how far a priorless model has drifted from the
    // GNSS track.
    if (mapper_opts.use_prior_position) {
      WorldPositionPriorOptions prior_opts;
      prior_opts.fallback_stddev = mapper_opts.prior_position_fallback_stddev;
      prior_opts.alignment_ransac_options.random_seed = mapper_opts.random_seed;
      prior_residuals_ = ComputePriorPositionResiduals(
          *reconstruction,
          CollectWorldPositionPriors(
              *database_cache_, *reconstruction, prior_opts),
          /*used_in_ba=*/{},
          mapper_opts.prior_position_fallback_stddev,
          mapper_opts.use_robust_loss_on_prior_position,
          mapper_opts.prior_position_loss_scale);
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

  // Global positioning re-solves camera positions from scratch, starting from
  // uniformly random points in a 200^3 cube. On an incremental add it has no
  // new information to contribute -- every prior camera already has a solved
  // position and the new image's position comes from BootstrapNewImagePoses --
  // so all it does is re-roll a good answer as a PRNG lottery. A
  // weakly-connected image that lands outside the scene has its observations
  // stripped by the reprojection filters, after which bundle adjustment has no
  // residuals left to pull it back and it is written out with a garbage pose.
  // Skip it and let bundle adjustment refine the positions against real
  // reprojection residuals instead. See plan-5.
  mapper_opts.skip_global_positioning = true;

  // With global positioning skipped, EstablishTracks' output feeds only the
  // bundle adjustment stage, and IterativeRetriangulateAndRefine then opens
  // by deleting every one of those points. Worse, EstablishTracks creates
  // tracks with default-constructed Point3D (xyz = origin) and nothing
  // assigns them a position now that GP is gone, so that BA is
  // triangulation-by-gradient-descent from a degenerate all-at-origin
  // configuration -- roughly 31% of an add spent on structure that is
  // discarded unused. Skip both. These stay tied to skip_global_positioning:
  // if GP is ever re-enabled here it needs established tracks. See plan-6.
  mapper_opts.skip_track_establishment = true;
  mapper_opts.skip_bundle_adjustment = true;

  // If global positioning is deliberately re-enabled on this path, at least
  // seed it from the prior reconstruction's camera centers rather than
  // discarding them.
  mapper_opts.global_positioning.generate_random_positions = false;

  // 5. Create the mapper, initialise, and inject prior poses.
  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction);

  mapper.LoadPriorPoses(prior_reconstruction);

  LOG(INFO) << "Prior images loaded: " << mapper.PriorImageIds().size();

  // 5b. Place the database's position priors in this reconstruction's frame.
  //
  //     This has to happen after LoadPriorPoses (the fit needs solved camera
  //     centres to work from) and before registration (which the priors
  //     gate). If it fails -- too few priors, or a fit that will not
  //     converge -- the add simply runs unconstrained, exactly as before.
  if (mapper_opts.use_prior_position) {
    if (mapper.ComputeWorldPositionPriors(mapper_opts)) {
      LOG(INFO) << "Position priors active for this add: "
                << mapper.PositionPriors().priors.size() << " prior(s), "
                << "alignment rmse "
                << mapper.PositionPriors().alignment_rmse << " m.";
    } else {
      LOG(WARNING) << "Position priors requested but unusable for this add; "
                      "solving without them.";
    }
  }

  // 6. Give the new image(s) a pose.
  //
  //    Preferred: PnP against the prior reconstruction's 3D points. Import
  //    the prior structure first so there is something to match against --
  //    the full path would otherwise not need it, but IterativeRetriangulate-
  //    AndRefine deletes all points before rebuilding, so carrying them here
  //    costs nothing and buys a far better pose.
  //
  //    Fallback: BootstrapNewImagePoses, which averages one rotation
  //    candidate per pose-graph edge. That is the wrong operator for a
  //    weakly-connected image -- noise edges outvote the few real ones --
  //    so it now only handles what PnP declines. See plan-6.
  std::unordered_set<image_t> new_image_ids;
  if (mapper_opts.use_pnp_registration) {
    if (mapper.ImportPriorPoints3D(mapper_opts, prior_reconstruction,
                                   &ledger_)) {
      new_image_ids = mapper.RegisterNewImagesByPnP(mapper_opts, &ledger_);
    } else {
      LOG(ERROR) << "Prior structure import failed; cannot register by PnP.";
      return;
    }
  }

  // BootstrapNewImagePoses skips images that already have a pose, so this
  // composes as a fallback chain with no further bookkeeping. It is off by
  // default once PnP has run: see
  // GlobalMapperOptions::bootstrap_fallback_when_pnp_declines.
  std::unordered_set<image_t> bootstrapped;
  if (!mapper_opts.use_pnp_registration ||
      mapper_opts.bootstrap_fallback_when_pnp_declines) {
    bootstrapped = mapper.BootstrapNewImagePoses(mapper_opts);
    new_image_ids.insert(bootstrapped.begin(), bootstrapped.end());
  }

  if (new_image_ids.empty()) {
    // Not a failure. Registering nothing is the correct outcome when the new
    // image cannot be placed against the existing structure: the prior model
    // is written through unchanged and the image is retried on the next add,
    // when there is more structure to match against. Forcing a pose here is
    // what produces permanently broken images.
    LOG(WARNING)
        << "IncrementalGlobalPipeline: no new image could be registered "
           "against the prior structure. Writing the prior model through "
           "unchanged; the image will be retried on the next add.";

    Reconstruction& passthrough =
        *reconstruction_manager_->Get(reconstruction_manager_->Add());
    passthrough = *reconstruction;
    // Keep the prior anchors as the fixed gauge reference.
    ReadAnchors(options_.prior_reconstruction_path / "anchors.txt", &anchors_);
    ReadIncrementalState(
        options_.prior_reconstruction_path / "incremental_state.txt", &state_);
    // Still counts as an add so ledger rows stay uniquely indexed; the label
    // is what distinguishes it.
    state_.num_adds += 1;
    path_label_ = "skipped";
    ledger_.points_out = reconstruction->NumPoints3D();
    ledger_.mean_reproj_error = reconstruction->ComputeMeanReprojectionError();
    ledger_.mean_track_length = reconstruction->ComputeMeanTrackLength();
    return;
  }

  LOG(INFO) << "Registered " << new_image_ids.size() << " new image(s): "
            << ledger_.pnp_registered << " by PnP, " << bootstrapped.size()
            << " by bootstrap fallback.";
  for (const image_t id : new_image_ids) {
    LOG(INFO) << "  image_id=" << id
              << (bootstrapped.count(id) ? " (bootstrap)" : " (pnp)");
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

  // 8. Decide windowed vs full, then solve.
  //
  //    A windowed solve structurally cannot refine camera intrinsics: both
  //    AdjustLocalBundle and AddPointToProblem force the camera constant
  //    whenever the window is smaller than the model, which with one shared
  //    camera is always. So whatever intrinsics we hold when we first go
  //    windowed would be frozen for the rest of the session -- the exact
  //    setup that collapses a model (refined geometry vs. unrefined
  //    intrinsics). Run full solves, which do refine intrinsics globally,
  //    until they converge; only then switch to windowed adds. See plan-6.
  ReadIncrementalState(
      options_.prior_reconstruction_path / "incremental_state.txt", &state_);

  const bool windowed =
      mapper_opts.optimize_window_size > 0 && state_.intrinsics_converged;
  path_label_ = windowed ? "windowed" : "full";

  if (mapper_opts.optimize_window_size > 0 && !windowed) {
    LOG(INFO) << "Windowed solves requested but intrinsics have not "
                 "converged yet (add " << state_.num_adds << ", "
              << state_.stable_adds
              << " consecutive stable add(s)); running a full solve so "
                 "intrinsics can be refined globally.";
  }

  Timer run_timer;
  run_timer.Start();
  if (windowed) {
    LOG(INFO) << "Running windowed incremental solve (window size "
              << mapper_opts.optimize_window_size << ").";
    // Anchor poses stay constant even when they fall inside the window, so
    // the gauge is fixed by construction.
    if (!mapper.SolveIncrementalWindowed(mapper_opts,
                                         prior_reconstruction,
                                         new_image_ids,
                                         prior_anchors.ImageIds(),
                                         &ledger_)) {
      LOG(ERROR) << "Windowed incremental solve failed.";
      return;
    }
  } else {
    mapper.Solve(mapper_opts);
    ledger_.prior_points_total = prior_reconstruction.NumPoints3D();
    ledger_.points_out = reconstruction->NumPoints3D();
    ledger_.mean_reproj_error = reconstruction->ComputeMeanReprojectionError();
    ledger_.mean_track_length = reconstruction->ComputeMeanTrackLength();
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
  //
  //     Also skipped when position priors are in use. The priors are then the
  //     gauge: a full solve has aligned the model to them and it is metric,
  //     and a Sim3 onto anchor centres recorded from an earlier solve would
  //     pull that back -- undoing the correction rather than checking it.
  const bool prior_positions_active = mapper.UsePositionPriors();
  if (prior_positions_active && !windowed &&
      mapper_opts.realign_to_prior_after_solve) {
    LOG(INFO) << "Skipping realignment to the prior frame: position priors "
                 "already fix the gauge and keep the model metric.";
  }
  if (!windowed && !prior_positions_active &&
      mapper_opts.realign_to_prior_after_solve) {
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

  // 10b. Per-image position-prior residuals (plan-6 item 2).
  //
  //      Computed after the solve and after any realignment, so the numbers
  //      describe the model as it is written out. Everything here is already
  //      known to bundle adjustment; the point is that it stops being
  //      implicit. A displaced frame is identifiable from this one file --
  //      residual, sigma, robust weight -- rather than from a Sim3 and six
  //      rounds of re-deriving per-frame rotations.
  if (prior_positions_active) {
    // Recompute the frame fit against the final poses: the solve has moved
    // them, so the priors captured before registration are one solve stale.
    WorldPositionPriorOptions prior_opts;
    prior_opts.fallback_stddev = mapper_opts.prior_position_fallback_stddev;
    prior_opts.alignment_ransac_options.random_seed = mapper_opts.random_seed;
    const WorldPositionPriors final_priors = CollectWorldPositionPriors(
        *database_cache_, *reconstruction, prior_opts);

    prior_residuals_ = ComputePriorPositionResiduals(
        *reconstruction,
        final_priors.Empty() ? mapper.PositionPriors() : final_priors,
        mapper.PriorConstrainedImageIds(),
        mapper_opts.prior_position_fallback_stddev,
        mapper_opts.use_robust_loss_on_prior_position,
        mapper_opts.prior_position_loss_scale);

    const PriorPositionResidualSummary summary =
        SummarizePriorPositionResiduals(prior_residuals_);
    ledger_.prior_residuals = summary.num_residuals;
    ledger_.prior_residual_rms = summary.rms;
    ledger_.prior_residual_worst = summary.worst_norm;
    ledger_.prior_residual_worst_sigma = summary.worst_mahalanobis;
    ledger_.prior_residual_worst_image = summary.worst_image_id;

    LOG(INFO) << "Position-prior residuals: " << summary.num_residuals
              << " image(s), rms " << summary.rms << " m, worst "
              << summary.worst_norm << " m (" << summary.worst_mahalanobis
              << " sigma) at image " << summary.worst_image_id << ".";
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

  // 12. Drift monitor and intrinsics-convergence bookkeeping.
  //
  //     The drift check runs first: if this add's reprojection error has
  //     risen well above the best we have ever seen, the frozen intrinsics
  //     are the prime suspect, so give that global parameter its own global
  //     pass and recover the structure that was filtered away while it was
  //     wrong. Convergence is then evaluated against the *post-refresh*
  //     intrinsics.
  state_.num_adds += 1;

  double& reproj_ema =
      windowed ? state_.reproj_ema_windowed : state_.reproj_ema_full;

  if (mapper_opts.intrinsics_drift_factor > 0.0 && reproj_ema > 0.0 &&
      ledger_.mean_reproj_error >
          reproj_ema * mapper_opts.intrinsics_drift_factor) {
    LOG(WARNING) << "Reprojection error drift: " << ledger_.mean_reproj_error
                 << " exceeds the running " << (windowed ? "windowed" : "full")
                 << "-path average " << reproj_ema << " by more than "
                 << mapper_opts.intrinsics_drift_factor
                 << "x. Running global refresh + structure recovery.";
    mapper.RefreshIntrinsicsGlobally(mapper_opts, &ledger_);
    state_.last_refresh_add = state_.num_adds;
    // Deliberately NOT resetting the convergence state here. Forcing the
    // next add back onto the full path would rebuild the whole model --
    // exactly the O(N) cost this is all trying to remove -- on the strength
    // of an indirect signal. If the intrinsics genuinely moved, the
    // stationarity test below detects it directly and un-converges on its
    // own; if they did not, the refresh was a cheap no-op plus some
    // recovered structure, which is pure gain.
  }

  // Convergence test: every camera's mean focal length must sit within the
  // relative tolerance of its running average, for intrinsics_min_stable_adds
  // consecutive adds.
  //
  // This is deliberately a stationarity test rather than a decay test. The
  // focal estimate never stops moving -- each full solve rebuilds all
  // structure and re-lands slightly differently -- so comparing consecutive
  // adds just samples that noise and a strict threshold never fires. Testing
  // against the running average asks the question we actually care about:
  // is the estimate sitting in a tight band (fine to freeze), or is it
  // walking somewhere (not fine to freeze)?
  double max_rel_dev = -1.0;
  {
    // Exponential running average; 0.5 keeps it responsive to a genuine
    // move while still averaging out single-add jitter.
    constexpr double kFocalEmaAlpha = 0.5;

    std::unordered_map<camera_t, double> mean_focal;
    for (const auto& [camera_id, camera] : reconstruction->Cameras()) {
      mean_focal[camera_id] = camera.MeanFocalLength();
    }

    bool stable = !mean_focal.empty() && !state_.camera_mean_focal.empty();
    std::unordered_map<camera_t, double> updated_ema;
    for (const auto& [camera_id, focal] : mean_focal) {
      const auto it = state_.camera_mean_focal.find(camera_id);
      if (it == state_.camera_mean_focal.end() || it->second <= 0.0) {
        // A camera we have never seen before cannot be called stable, and
        // seeds its average with the current value.
        stable = false;
        updated_ema[camera_id] = focal;
        continue;
      }
      const double ema = it->second;
      const double rel_dev = std::abs(focal - ema) / std::abs(ema);
      max_rel_dev = std::max(max_rel_dev, rel_dev);
      if (rel_dev > mapper_opts.intrinsics_convergence_rel_tol) {
        stable = false;
      }
      updated_ema[camera_id] = kFocalEmaAlpha * focal + (1.0 - kFocalEmaAlpha) * ema;
    }

    state_.stable_adds = stable ? state_.stable_adds + 1 : 0;
    state_.camera_mean_focal = std::move(updated_ema);

    const bool converged =
        state_.stable_adds >= mapper_opts.intrinsics_min_stable_adds &&
        state_.num_adds >= mapper_opts.intrinsics_min_adds;
    if (converged && !state_.intrinsics_converged) {
      LOG(INFO) << "Intrinsics converged after " << state_.num_adds
                << " add(s) (max relative focal deviation " << max_rel_dev
                << "); subsequent adds may run windowed.";
    }
    state_.intrinsics_converged = converged;
  }

  if (ledger_.mean_reproj_error > 0.0) {
    constexpr double kReprojEmaAlpha = 0.5;
    reproj_ema = (reproj_ema <= 0.0)
                     ? ledger_.mean_reproj_error
                     : kReprojEmaAlpha * ledger_.mean_reproj_error +
                           (1.0 - kReprojEmaAlpha) * reproj_ema;
  }

  LOG(INFO) << "Add " << state_.num_adds << " (" << (windowed ? "windowed"
                                                              : "full")
            << "): " << ledger_.points_out << " point(s), mean track length "
            << ledger_.mean_track_length << ", mean reproj error "
            << ledger_.mean_reproj_error << "; intrinsics "
            << (state_.intrinsics_converged ? "converged" : "not converged")
            << " (" << state_.stable_adds
            << " stable add(s), max relative focal deviation " << max_rel_dev
            << " vs tolerance " << mapper_opts.intrinsics_convergence_rel_tol
            << ").";

  // 13. Write output.
  Reconstruction& out =
      *reconstruction_manager_->Get(reconstruction_manager_->Add());
  out = *reconstruction;

  if (!options_.image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    out.ExtractColorsForAllImages(options_.image_path);
  }
}

}  // namespace colmap
