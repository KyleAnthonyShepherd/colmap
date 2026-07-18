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
#include <unordered_set>
#include <utility>
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

// ── Anchor-based gauge handling ────────────────────────────────────────────
//
// Rolling Sim3 realignment (output → previous output) compounds scale error
// multiplicatively over long sessions. Instead, the first successful
// reconstruction persists a small set of well-conditioned anchor images
// (id + camera centre at anchor-creation time) in an `anchors.txt` file next
// to the reconstruction files. All later full solves realign to the FIXED
// anchor centres; windowed solves keep the anchor poses constant so the
// gauge is fixed by construction and no realignment is needed at all.

struct AnchorSet {
  // (image_id, world camera centre at anchor-creation time).
  std::vector<std::pair<image_t, Eigen::Vector3d>> anchors;

  bool Empty() const { return anchors.empty(); }
  std::unordered_set<image_t> ImageIds() const {
    std::unordered_set<image_t> ids;
    for (const auto& [id, _] : anchors) ids.insert(id);
    return ids;
  }
};

// Selects up to `num_anchors` well-conditioned anchor images: many 3D
// points, mutually wide baselines. Requires a reconstruction with poses.
std::vector<image_t> SelectAnchorImages(const Reconstruction& reconstruction,
                                        int num_anchors = 4);

// Reads/writes `anchors.txt` (one "image_id cx cy cz" line per anchor).
// ReadAnchors returns false if the file does not exist or is unparsable.
bool ReadAnchors(const std::filesystem::path& path, AnchorSet* anchor_set);
void WriteAnchors(const std::filesystem::path& path,
                  const AnchorSet& anchor_set);

// Realigns `reconstruction` with a Sim3 fitted from its current anchor
// camera centres to the stored anchor centres. Returns false (and leaves
// the reconstruction untouched) when fewer than 3 anchors are registered in
// the reconstruction or the fit fails.
bool RealignToAnchors(const AnchorSet& anchor_set,
                      Reconstruction* reconstruction);

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

  // The anchor set to persist next to the output reconstruction: the prior
  // anchors when they exist (fixed reference), otherwise freshly selected
  // from the output. Empty until Run() has produced a reconstruction.
  const AnchorSet& Anchors() const { return anchors_; }

 private:
  const IncrementalGlobalPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
  AnchorSet anchors_;
};

}  // namespace colmap
