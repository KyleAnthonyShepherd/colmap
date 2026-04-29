// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// (Same BSD licence header as other COLMAP files.)

#pragma once

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/base_controller.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace colmap {

// ── IncrementalGlobalPipelineOptions ──────────────────────────────────────
//
// Extends GlobalPipelineOptions with a `prior_reconstruction_path` that
// points to an already-solved COLMAP sparse reconstruction.  When this path
// is non-empty, IncrementalGlobalPipeline:
//
//   1. Loads the prior reconstruction.
//   2. Builds a DatabaseCache from the current database (which must already
//      contain the features and verified matches for the new image).
//   3. Calls GlobalMapper::BeginReconstruction on a fresh Reconstruction.
//   4. Seeds all prior camera poses with GlobalMapper::LoadPriorPoses.
//   5. Bootstraps the new camera's rotation with
//      GlobalMapper::BootstrapNewImagePoses.
//   6. Calls GlobalMapper::Solve with skip_rotation_averaging = true so the
//      bootstrapped rotations are used directly.
//   7. Optionally realigns the output to the prior coordinate frame with a
//      Sim3 fit over the prior camera centres.
//
// All options that govern the global SfM stages after bootstrapping (track
// establishment, global positioning, bundle adjustment, retriangulation) are
// inherited from GlobalPipelineOptions::mapper.
//
struct IncrementalGlobalPipelineOptions {
  // Path to a previously solved COLMAP sparse reconstruction directory
  // (contains cameras.bin/txt, images.bin/txt, points3D.bin/txt).
  // If empty, the pipeline falls back to a standard GlobalPipeline run.
  std::filesystem::path prior_reconstruction_path;

  // ── Forwarded GlobalPipelineOptions fields ──────────────────────────────
  int min_num_matches = 15;
  bool ignore_watermarks = false;
  std::vector<std::string> image_names;
  std::filesystem::path image_path;
  int num_threads = -1;
  int random_seed = -1;
  bool decompose_relative_pose = true;

  // Core solver options (skip_rotation_averaging is forced to true when
  // prior_reconstruction_path is set).
  GlobalMapperOptions mapper;
};

// ── IncrementalGlobalPipeline ──────────────────────────────────────────────
//
// Controller that adds a single new image to an already-solved global
// reconstruction.  Designed to be a drop-in replacement for GlobalPipeline
// in the incremental-add use case.
//
class IncrementalGlobalPipeline : public BaseController {
 public:
  IncrementalGlobalPipeline(
      IncrementalGlobalPipelineOptions options,
      std::shared_ptr<Database> database,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Run() override;

 private:
  const IncrementalGlobalPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
