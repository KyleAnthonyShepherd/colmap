// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/exe/sfm.h"

#include "colmap/controllers/automatic_reconstruction.h"
#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/incremental_global_pipeline.h"
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/rotation_averaging.h"
#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/estimators/view_graph_calibration.h"
#include "colmap/exe/gui.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/sfm/prior_positions.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {

std::pair<std::vector<image_t>, std::vector<Eigen::Vector3d>>
ExtractExistingImages(const Reconstruction& reconstruction) {
  std::vector<image_t> fixed_image_ids = reconstruction.RegImageIds();
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  orig_fixed_image_positions.reserve(fixed_image_ids.size());
  for (const image_t image_id : fixed_image_ids) {
    orig_fixed_image_positions.push_back(
        reconstruction.Image(image_id).ProjectionCenter());
  }
  return {std::move(fixed_image_ids), std::move(orig_fixed_image_positions)};
}

void UpdateDatabasePosePriorsCovariance(
    const std::filesystem::path& database_path,
    const Eigen::Matrix3d& covariance) {
  auto database = Database::Open(database_path);
  DatabaseTransaction database_transaction(database.get());

  LOG(INFO)
      << "Setting up database pose priors with the same covariance matrix: \n"
      << covariance << '\n';

  for (auto& pose_prior : database->ReadAllPosePriors()) {
    pose_prior.position_covariance = covariance;
    database->UpdatePosePrior(pose_prior);
  }
}

}  // namespace

int RunAutomaticReconstructor(int argc, char** argv) {
  AutomaticReconstructionController::Options reconstruction_options;
  std::filesystem::path image_list_path;
  std::string data_type = "individual";
  std::string quality = "high";
  std::string feature = "sift";
  std::string mapper = "incremental";
  std::string mesher = "poisson";

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("mask_path", &reconstruction_options.mask_path);
  options.AddDefaultOption("vocab_tree_path",
                           &reconstruction_options.vocab_tree_path);
  options.AddDefaultOption(
      "data_type", &data_type, "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
  options.AddDefaultOption("single_camera_per_folder",
                           &reconstruction_options.single_camera_per_folder);
  options.AddDefaultOption("camera_params",
                           &reconstruction_options.camera_params);
  options.AddDefaultOption("extraction", &reconstruction_options.extraction);
  options.AddDefaultOption("matching", &reconstruction_options.matching);
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("feature", &feature, "{sift, aliked}");
  options.AddDefaultOption(
      "mapper", &mapper, "{incremental, hierarchical, global}");
  options.AddDefaultOption("mesher", &mesher, "{poisson, delaunay}");
  options.AddDefaultOption("num_threads", &reconstruction_options.num_threads);
  options.AddDefaultOption("random_seed", &reconstruction_options.random_seed);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("gpu_index", &reconstruction_options.gpu_index);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    reconstruction_options.image_names = ReadTextFileLines(image_list_path);
  }

  StringToUpper(&data_type);
  reconstruction_options.data_type =
      AutomaticReconstructionController::DataTypeFromString(data_type);

  StringToUpper(&quality);
  reconstruction_options.quality =
      AutomaticReconstructionController::QualityFromString(quality);

  StringToUpper(&feature);
  reconstruction_options.feature =
      AutomaticReconstructionController::FeatureFromString(feature);

  StringToUpper(&mapper);
  reconstruction_options.mapper =
      AutomaticReconstructionController::MapperFromString(mapper);

  StringToUpper(&mesher);
  reconstruction_options.mesher =
      AutomaticReconstructionController::MesherFromString(mesher);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  AutomaticReconstructionController controller(reconstruction_options,
                                               reconstruction_manager);
  if (controller.RequiresOpenGL()) {
    QApplication app(argc, argv);
    controller.Setup();
    RunThreadWithOpenGLContext(&controller);
  } else {
    controller.Setup();
    controller.Start();
    controller.Wait();
  }

  return EXIT_SUCCESS;
}

int RunBundleAdjuster(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddBundleAdjustmentOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  BundleAdjustmentController ba_controller(options, reconstruction);
  ba_controller.Run();

  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunColorExtractor(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  int num_threads = -1;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_threads", &num_threads);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.ExtractColorsForAllImages(*options.image_path, num_threads);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

bool RunIncrementalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<IncrementalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager,
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback) {
  // If fix_existing_frames is enabled, we store the initial positions of
  // existing images in order to transform them back to the original coordinate
  // frame, as the reconstruction is normalized multiple times for numerical
  // stability.
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  std::vector<image_t> fixed_image_ids;
  const bool exists_input_reconstruction = reconstruction_manager->Size() > 0;
  if (mapper_options->fix_existing_frames && exists_input_reconstruction) {
    std::tie(fixed_image_ids, orig_fixed_image_positions) =
        ExtractExistingImages(*reconstruction_manager->Get(0));
  }

  mapper_options->image_path = image_path;

  auto database = Database::Open(database_path);

  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (!exists_input_reconstruction) {
    mapper.AddCallback(IncrementalPipeline::LAST_IMAGE_REG_CALLBACK, [&]() {
      // If the number of reconstructions has not changed, the last model
      // was discarded for some reason.
      if (reconstruction_manager->Size() > prev_num_reconstructions) {
        const auto reconstruction_path =
            output_path / std::to_string(prev_num_reconstructions);
        CreateDirIfNotExists(reconstruction_path);
        reconstruction_manager->Get(prev_num_reconstructions)
            ->Write(reconstruction_path);
        prev_num_reconstructions = reconstruction_manager->Size();
      }
    });
  }

  if (initial_image_pair_callback) {
    mapper.AddCallback(IncrementalPipeline::INITIAL_IMAGE_PAIR_REG_CALLBACK,
                       std::move(initial_image_pair_callback));
  }

  if (next_image_callback) {
    mapper.AddCallback(IncrementalPipeline::NEXT_IMAGE_REG_CALLBACK,
                       std::move(next_image_callback));
  }

  mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "Failed to create any sparse model";
    return false;
  }

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (exists_input_reconstruction) {
    const auto& reconstruction = reconstruction_manager->Get(0);

    // Transform the final reconstruction back to the original coordinate frame.
    if (mapper_options->fix_existing_frames) {
      if (fixed_image_ids.size() < 3) {
        LOG(WARNING) << "Too few images to transform the reconstruction.";
      } else {
        std::vector<Eigen::Vector3d> new_fixed_image_positions;
        new_fixed_image_positions.reserve(fixed_image_ids.size());
        for (const image_t image_id : fixed_image_ids) {
          new_fixed_image_positions.push_back(
              reconstruction->Image(image_id).ProjectionCenter());
        }
        Sim3d orig_from_new;
        if (EstimateSim3d(new_fixed_image_positions,
                          orig_fixed_image_positions,
                          orig_from_new)) {
          reconstruction->Transform(orig_from_new);
        } else {
          LOG(WARNING) << "Failed to transform the reconstruction back "
                          "to the input coordinate frame.";
        }
      }
    }

    reconstruction->Write(output_path);
  }

  return true;
}

int RunMapper(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (!input_path.empty()) {
    if (!ExistsDir(input_path)) {
      LOG(ERROR) << "`input_path` is not a directory.";
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  if (!RunIncrementalMapperImpl(*options.database_path,
                                *options.image_path,
                                output_path,
                                options.mapper,
                                reconstruction_manager)) {
    return EXIT_FAILURE;
  }

  if (input_path.empty()) {
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
      const auto reconstruction_path = output_path / std::to_string(i);
      options.Write(reconstruction_path / "project.ini");
    }
  }

  return EXIT_SUCCESS;
}

bool RunGlobalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<GlobalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager) {
  GlobalPipelineOptions options = *mapper_options;
  options.image_path = image_path;

  GlobalPipeline global_mapper(std::move(options),
                               Database::Open(database_path),
                               reconstruction_manager);
  global_mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "Failed to create sparse model";
    return false;
  }

  reconstruction_manager->Write(output_path);
  return true;
}

int RunGlobalMapper(int argc, char** argv) {
  std::filesystem::path output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddGlobalMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (!RunGlobalMapperImpl(*options.database_path,
                           *options.image_path,
                           output_path,
                           options.global_mapper,
                           reconstruction_manager)) {
    return EXIT_FAILURE;
  }

  options.Write(output_path / "project.ini");
  return EXIT_SUCCESS;
}

int RunIncrementalGlobalMapper(int argc, char** argv) {
  std::filesystem::path prior_reconstruction_path;
  std::filesystem::path output_path;

  // Position-prior options. Named exactly as in `pose_prior_mapper` and, like
  // there, un-prefixed -- the `Mapper.`-prefixed options belong to the
  // reconstruction algorithm, these to the prior handling around it.
  bool overwrite_priors_covariance = false;
  double prior_position_std_x = 1.;
  double prior_position_std_y = 1.;
  double prior_position_std_z = 1.;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("prior_reconstruction_path",
                             &prior_reconstruction_path,
                             "Path to the existing solved reconstruction "
                             "(directory with cameras/images/points3D).");
  options.AddRequiredOption("output_path", &output_path);
  options.AddGlobalMapperOptions();  // Reuses GlobalMapper option registration.

  // Expose the key incremental-add options directly on the command line.
  options.AddDefaultOption(
      "bootstrap_min_inliers",
      &options.global_mapper->mapper.bootstrap_min_inliers,
      "Min inliers on a prior↔new edge for rotation bootstrapping.");
  options.AddDefaultOption(
      "bootstrap_max_candidate_deg",
      &options.global_mapper->mapper.bootstrap_max_candidate_deg,
      "Max rotation error (deg) for outlier rejection in bootstrapping.");
  options.AddDefaultOption(
      "realign_to_prior",
      &options.global_mapper->mapper.realign_to_prior_after_solve,
      "Re-align output to the prior coordinate frame via Sim3 after solve.");
  options.AddDefaultOption(
      "bootstrap_max_gravity_error_deg",
      &options.global_mapper->mapper.bootstrap_max_gravity_error_deg,
      "Max angle (deg) between measured and candidate-implied gravity "
      "before a bootstrap candidate is rejected; <= 0 disables.");
  options.AddDefaultOption(
      "optimize_window_size",
      &options.global_mapper->mapper.optimize_window_size,
      "When > 0, run a windowed local solve around the new image(s) with "
      "this many covisible images instead of a full global solve. Windowed "
      "adds only begin once intrinsics have converged; until then full "
      "solves run so intrinsics can be refined globally.");
  options.AddDefaultOption(
      "intrinsics_convergence_rel_tol",
      &options.global_mapper->mapper.intrinsics_convergence_rel_tol,
      "Max relative change in a camera's mean focal length between adds for "
      "that add to count as stable.");
  options.AddDefaultOption(
      "intrinsics_min_stable_adds",
      &options.global_mapper->mapper.intrinsics_min_stable_adds,
      "Consecutive stable adds required to declare intrinsics converged.");
  options.AddDefaultOption(
      "intrinsics_min_adds",
      &options.global_mapper->mapper.intrinsics_min_adds,
      "Minimum number of adds before convergence may be declared.");
  options.AddDefaultOption(
      "intrinsics_drift_factor",
      &options.global_mapper->mapper.intrinsics_drift_factor,
      "Run an intrinsics-only global refresh plus structure recovery when "
      "mean reprojection error exceeds the best recorded times this factor; "
      "<= 0 disables the drift monitor.");
  options.AddDefaultOption(
      "pnp_min_num_inliers",
      &options.global_mapper->mapper.pnp_min_num_inliers,
      "Minimum inlier 2D-3D correspondences to register a new image by PnP.");
  options.AddDefaultOption(
      "gravity_uncertainty_deg",
      &options.global_mapper->mapper.gravity_uncertainty_deg,
      "Expected worst-case gravity prior error (deg); widens the PnP inlier "
      "threshold by focal*tan(this) while the upright solver is in use.");
  options.AddDefaultOption(
      "allow_lossy_prior_import",
      &options.global_mapper->mapper.allow_lossy_prior_import,
      "Continue a windowed add even when prior 3D structure could not be "
      "imported faithfully. Off by default: a lossy import means the prior "
      "and the database disagree, and the loss compounds across adds.");

  // ── Position priors ──────────────────────────────────────────────────
  options.AddDefaultOption(
      "use_prior_position",
      &options.global_mapper->mapper.use_prior_position,
      "Constrain the solve with the database's position priors (the "
      "`pose_priors` table). Off by default. With priors on, the model is "
      "metric before georeferencing runs and a pose that contradicts its "
      "GNSS fix is declined at registration rather than optimised "
      "afterwards.");
  options.AddDefaultOption(
      "use_robust_loss_on_prior_position",
      &options.global_mapper->mapper.use_robust_loss_on_prior_position,
      "Down-weight prior-position residuals with a Cauchy loss.");
  options.AddDefaultOption(
      "prior_position_loss_scale",
      &options.global_mapper->mapper.prior_position_loss_scale,
      "Threshold on the covariance-whitened residual for the robust loss "
      "(chi2, 3 DOF, 95% = 7.815).");
  options.AddDefaultOption(
      "prior_position_fallback_stddev",
      &options.global_mapper->mapper.prior_position_fallback_stddev,
      "Standard deviation (m) assumed for a prior that carries no covariance "
      "of its own. Covariance is read from the database by default; this is "
      "a last resort, and a 1 m default would throw away a centimetre-level "
      "measurement.");
  options.AddDefaultOption(
      "prior_position_max_error_sigma",
      &options.global_mapper->mapper.prior_position_max_error_sigma,
      "Decline an absolute-pose registration whose camera centre disagrees "
      "with its prior by more than this many sigmas AND more than "
      "--prior_position_max_error_m metres. <= 0 disables the gate.");
  options.AddDefaultOption(
      "prior_position_max_error_m",
      &options.global_mapper->mapper.prior_position_max_error_m,
      "Absolute floor (m) for the registration gate above, so a very tight "
      "prior sigma cannot reject on ordinary SfM noise.");
  options.AddDefaultOption(
      "overwrite_priors_covariance",
      &overwrite_priors_covariance,
      "Priors covariance is read from the database. If true, overwrite it in "
      "the database using the prior_position_std_... options below.");
  options.AddDefaultOption("prior_position_std_x", &prior_position_std_x);
  options.AddDefaultOption("prior_position_std_y", &prior_position_std_y);
  options.AddDefaultOption("prior_position_std_z", &prior_position_std_z);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (overwrite_priors_covariance) {
    const Eigen::Matrix3d covariance =
        Eigen::Vector3d(
            prior_position_std_x, prior_position_std_y, prior_position_std_z)
            .cwiseAbs2()
            .asDiagonal();
    UpdateDatabasePosePriorsCovariance(*options.database_path, covariance);
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(prior_reconstruction_path)) {
    LOG(ERROR) << "`prior_reconstruction_path` is not a directory.";
    return EXIT_FAILURE;
  }

  // Build IncrementalGlobalPipelineOptions from parsed GlobalPipelineOptions.
  IncrementalGlobalPipelineOptions pipeline_opts;
  pipeline_opts.prior_reconstruction_path = prior_reconstruction_path;
  pipeline_opts.min_num_matches = options.global_mapper->min_num_matches;
  pipeline_opts.ignore_watermarks = options.global_mapper->ignore_watermarks;
  pipeline_opts.image_path = *options.image_path;
  pipeline_opts.num_threads = options.global_mapper->num_threads;
  pipeline_opts.random_seed = options.global_mapper->random_seed;
  pipeline_opts.decompose_relative_pose =
      options.global_mapper->decompose_relative_pose;
  pipeline_opts.mapper = options.global_mapper->mapper;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  IncrementalGlobalPipeline pipeline(std::move(pipeline_opts),
                                     Database::Open(*options.database_path),
                                     reconstruction_manager);
  pipeline.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "Failed to create sparse model.";
    return EXIT_FAILURE;
  }

  reconstruction_manager->Write(output_path);
  options.Write(output_path / "project.ini");

  // Persist the gauge anchors next to the reconstruction files so they
  // survive the caller's promotion of the output directory and act as the
  // fixed realignment reference for all subsequent incremental adds.
  const std::filesystem::path numbered_dir = output_path / "0";
  const std::filesystem::path model_dir =
      ExistsDir(numbered_dir) ? numbered_dir : output_path;

  if (!pipeline.Anchors().Empty()) {
    WriteAnchors(model_dir / "anchors.txt", pipeline.Anchors());
  }

  // Persist the cross-add state alongside the anchors: the server runs this
  // binary once per added image, so intrinsics-convergence and drift
  // tracking only survive on disk. The ledger carries the whole session's
  // structure accounting forward so decay across a drip-fed session is a
  // query over a file rather than a guess.
  // Per-image position-prior residuals, beside the model: prior position,
  // solved position, residual, the sigma actually used, and the weight the
  // robust loss gave it. Written every add, so a displaced frame is visible
  // during the walk rather than only at finalize. See plan-6 item 2.
  if (!pipeline.PriorResiduals().empty()) {
    WritePriorPositionResiduals(model_dir / "prior_residuals.tsv",
                                pipeline.PriorResiduals());
  }

  WriteIncrementalState(model_dir / "incremental_state.txt", pipeline.State());
  AppendIncrementalLedger(
      model_dir / "incremental_ledger.tsv",
      prior_reconstruction_path / "incremental_ledger.tsv",
      pipeline.State().num_adds,
      pipeline.PathLabel(),
      pipeline.Ledger());

  return EXIT_SUCCESS;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalPipeline::Options mapper_options;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("image_path", &mapper_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_threads", &mapper_options.num_threads);
  options.AddDefaultOption("num_workers", &mapper_options.num_workers);
  options.AddDefaultOption("image_overlap",
                           &mapper_options.clustering_options.image_overlap);
  options.AddDefaultOption(
      "leaf_max_num_images",
      &mapper_options.clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  mapper_options.incremental_options = *options.mapper;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipeline hierarchical_mapper(
      mapper_options,
      Database::Open(*options.database_path),
      reconstruction_manager);
  hierarchical_mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  reconstruction_manager->Write(output_path);
  options.Write(output_path / "project.ini");

  return EXIT_SUCCESS;
}

int RunPosePriorMapper(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  bool overwrite_priors_covariance = false;
  double prior_position_std_x = 1.;
  double prior_position_std_y = 1.;
  double prior_position_std_z = 1.;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();

  options.mapper->use_prior_position = true;

  // NOTE: these prior options are deliberately un-prefixed, while this
  // binary's reconstruction options carry the `Mapper.` prefix
  // (--Mapper.ba_refine_focal_length and friends). The same three settings
  // are also reachable as --Mapper.use_prior_position,
  // --Mapper.use_robust_loss_on_prior_position and
  // --Mapper.prior_position_loss_scale; both spellings set the same values.
  // The covariance options below exist only here and in
  // `incremental_global_mapper`, because they write to the database.
  options.AddDefaultOption(
      "overwrite_priors_covariance",
      &overwrite_priors_covariance,
      "Priors covariance is read from the database. If true, overwrite it "
      "using the prior_position_std_... options below. Un-prefixed, unlike "
      "the Mapper.* reconstruction options.");
  options.AddDefaultOption("prior_position_std_x",
                           &prior_position_std_x,
                           "Prior position std dev (m) on x, used only with "
                           "--overwrite_priors_covariance.");
  options.AddDefaultOption("prior_position_std_y",
                           &prior_position_std_y,
                           "Prior position std dev (m) on y, used only with "
                           "--overwrite_priors_covariance.");
  options.AddDefaultOption("prior_position_std_z",
                           &prior_position_std_z,
                           "Prior position std dev (m) on z, used only with "
                           "--overwrite_priors_covariance.");
  options.AddDefaultOption(
      "use_robust_loss_on_prior_position",
      &options.mapper->use_robust_loss_on_prior_position,
      "Down-weight prior-position residuals with a Cauchy loss. Un-prefixed; "
      "same as --Mapper.use_robust_loss_on_prior_position.");
  options.AddDefaultOption(
      "prior_position_loss_scale",
      &options.mapper->prior_position_loss_scale,
      "Threshold on the covariance-whitened residual for the robust loss "
      "(chi2, 3 DOF, 95% = 7.815). Un-prefixed; same as "
      "--Mapper.prior_position_loss_scale.");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  if (overwrite_priors_covariance) {
    const Eigen::Matrix3d covariance =
        Eigen::Vector3d(
            prior_position_std_x, prior_position_std_y, prior_position_std_z)
            .cwiseAbs2()
            .asDiagonal();
    UpdateDatabasePosePriorsCovariance(*options.database_path, covariance);
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      LOG(ERROR) << "`input_path` is not a directory.";
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  if (!RunIncrementalMapperImpl(*options.database_path,
                                *options.image_path,
                                output_path,
                                options.mapper,
                                reconstruction_manager)) {
    return EXIT_FAILURE;
  }

  if (input_path.empty()) {
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
      const auto reconstruction_path = output_path / std::to_string(i);
      options.Write(reconstruction_path / "project.ini");
    }
  }

  return EXIT_SUCCESS;
}

int RunPointFiltering(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  int min_track_len = 2;
  double max_reproj_error = 4.0;
  double min_tri_angle = 1.5;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_track_len", &min_track_len);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("min_tri_angle", &min_tri_angle);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  ObservationManager obs_manager(reconstruction);
  size_t num_filtered =
      obs_manager.FilterAllPoints3D(max_reproj_error, min_tri_angle);
  num_filtered += obs_manager.FilterPoints3DWithShortTracks(min_track_len);

  LOG(INFO) << "Filtered observations: " << num_filtered;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunPointTriangulator(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  bool clear_points = true;
  bool refine_intrinsics = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points",
      &clear_points,
      "Whether to clear all existing points and observations and recompute "
      "the image_ids based on matching filenames between the model and the "
      "database");
  options.AddDefaultOption("refine_intrinsics",
                           &refine_intrinsics,
                           "Whether to refine the intrinsics of the cameras "
                           "(fixing the principal point)");
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  LOG_HEADING1("Loading model");

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  RunPointTriangulatorImpl(reconstruction,
                           *options.database_path,
                           *options.image_path,
                           output_path,
                           *options.mapper,
                           clear_points,
                           refine_intrinsics);
  return EXIT_SUCCESS;
}

void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const IncrementalPipelineOptions& options,
    const bool clear_points,
    const bool refine_intrinsics) {
  THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
      << "Need at least two images for triangulation";
  if (clear_points) {
    reconstruction->DeleteAllPoints2DAndPoints3D();
    reconstruction->TranscribeImageIdsToDatabase(
        *Database::Open(database_path));
  }

  auto custom_options = std::make_shared<IncrementalPipelineOptions>(options);
  custom_options->image_path = image_path;
  custom_options->load_all_images = true;
  custom_options->fix_existing_frames = true;
  custom_options->ba_refine_focal_length = refine_intrinsics;
  custom_options->ba_refine_principal_point = false;
  custom_options->ba_refine_extra_params = refine_intrinsics;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(
      custom_options, Database::Open(database_path), reconstruction_manager);
  mapper.TriangulateReconstruction(reconstruction);
  reconstruction->Write(output_path);
}

int RunRotationAverager(int argc, char** argv) {
  std::filesystem::path output_path;
  std::filesystem::path image_list_path;

  RotationAveragingPipelineOptions controller_options;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("min_num_matches",
                           &controller_options.min_num_matches);
  options.AddDefaultOption("ignore_watermarks",
                           &controller_options.ignore_watermarks);
  options.AddDefaultOption("num_threads", &controller_options.num_threads);
  options.AddDefaultOption("random_seed", &controller_options.random_seed);
  options.AddDefaultOption("use_gravity",
                           &controller_options.rotation_estimation.use_gravity);
  options.AddDefaultOption(
      "use_stratified", &controller_options.rotation_estimation.use_stratified);
  options.AddDefaultOption("refine_gravity",
                           &controller_options.refine_gravity);
  options.AddGravityRefinerOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  controller_options.gravity_refiner = *options.gravity_refiner;

  if (!image_list_path.empty()) {
    controller_options.image_names = ReadTextFileLines(image_list_path);
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto database = Database::Open(*options.database_path);
  auto reconstruction = std::make_shared<Reconstruction>();

  RotationAveragingPipeline controller(
      controller_options, std::move(database), reconstruction);
  controller.Run();

  if (reconstruction->NumRegFrames() == 0) {
    LOG(ERROR) << "No frames registered";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Writing reconstruction to " << output_path;
  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunViewGraphCalibrator(int argc, char** argv) {
  ViewGraphCalibrationOptions calibration_options;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddDefaultOption(
      "cross_validate_prior_focal_lengths",
      &calibration_options.cross_validate_prior_focal_lengths,
      "Cross-validate prior focal lengths");
  options.AddDefaultOption(
      "min_calibrated_pair_ratio",
      &calibration_options.min_calibrated_pair_ratio,
      "Minimum ratio of calibrated pairs for cross-validation");
  options.AddDefaultOption("reestimate_relative_pose",
                           &calibration_options.reestimate_relative_pose,
                           "Re-estimate relative poses after calibration");
  options.AddDefaultOption("min_focal_length_ratio",
                           &calibration_options.min_focal_length_ratio,
                           "Minimum ratio of estimated to prior focal length");
  options.AddDefaultOption("max_focal_length_ratio",
                           &calibration_options.max_focal_length_ratio,
                           "Maximum ratio of estimated to prior focal length");
  options.AddDefaultOption("max_calibration_error",
                           &calibration_options.max_calibration_error,
                           "Maximum calibration error for an image pair");
  options.AddDefaultOption("relpose_max_error",
                           &calibration_options.relpose_max_error,
                           "Maximum error for relative pose re-estimation");
  options.AddDefaultOption("relpose_min_num_inliers",
                           &calibration_options.relpose_min_num_inliers,
                           "Minimum inliers for relative pose re-estimation");
  options.AddDefaultOption(
      "relpose_min_inlier_ratio",
      &calibration_options.relpose_min_inlier_ratio,
      "Minimum inlier ratio for relative pose re-estimation");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  auto database = Database::Open(*options.database_path);

  if (!CalibrateViewGraph(calibration_options, database.get())) {
    LOG(ERROR) << "View graph calibration failed";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "View graph calibration completed successfully";
  return EXIT_SUCCESS;
}

}  // namespace colmap
