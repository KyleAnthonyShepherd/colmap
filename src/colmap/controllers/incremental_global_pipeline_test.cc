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

#include "colmap/controllers/incremental_global_pipeline.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <filesystem>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Synthesizes a scene, writes a prior reconstruction that excludes one
// image, and runs the incremental pipeline to add that image back.
TEST(IncrementalGlobalPipeline, AddOneImageToPrior) {
  SetPRNGSeed(1);
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto prior_path = test_dir / "prior";
  std::filesystem::create_directories(prior_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  const image_t held_out = reg_ids.back();

  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());
  prior.Write(prior_path);

  IncrementalGlobalPipelineOptions options;
  options.prior_reconstruction_path = prior_path;
  options.mapper.random_seed = 1;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalGlobalPipeline pipeline(
      std::move(options), database, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const Reconstruction& output = *reconstruction_manager->Get(0);
  ASSERT_TRUE(output.ExistsImage(held_out));
  EXPECT_TRUE(output.Image(held_out).HasPose());

  // The output is realigned to the prior frame, which is the ground-truth
  // frame, so poses must match the ground truth without extra alignment.
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(output,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-2));
}

// Windowed mode: same add-one-image flow, but through the O(window) local
// solve. The output must stay in the prior frame without realignment.
TEST(IncrementalGlobalPipeline, AddOneImageWindowed) {
  SetPRNGSeed(1);
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto prior_path = test_dir / "prior";
  std::filesystem::create_directories(prior_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  const image_t held_out = reg_ids.back();

  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());
  prior.Write(prior_path);

  IncrementalGlobalPipelineOptions options;
  options.prior_reconstruction_path = prior_path;
  options.mapper.random_seed = 1;
  options.mapper.optimize_window_size = 4;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalGlobalPipeline pipeline(
      std::move(options), database, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const Reconstruction& output = *reconstruction_manager->Get(0);
  ASSERT_TRUE(output.ExistsImage(held_out));
  EXPECT_TRUE(output.Image(held_out).HasPose());
  EXPECT_GT(output.NumPoints3D(), 50);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(output,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-2));
}

// ── Anchor-based gauge handling ────────────────────────────────────────────

Reconstruction MakeSyntheticReconstruction(const std::string& db_name,
                                           std::shared_ptr<Database>* database,
                                           const int num_frames = 7) {
  const auto database_path = CreateTestDir() / db_name;
  *database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_frames;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database->get());
  return gt_reconstruction;
}

TEST(Anchors, SelectReadWriteRoundTrip) {
  SetPRNGSeed(1);
  std::shared_ptr<Database> database;
  const Reconstruction reconstruction =
      MakeSyntheticReconstruction("database_anchors.db", &database);

  const std::vector<image_t> anchor_ids = SelectAnchorImages(reconstruction);
  ASSERT_GE(anchor_ids.size(), 3);
  ASSERT_LE(anchor_ids.size(), 4);
  // Distinct ids, all registered.
  const std::unordered_set<image_t> unique_ids(anchor_ids.begin(),
                                               anchor_ids.end());
  EXPECT_EQ(unique_ids.size(), anchor_ids.size());
  for (const image_t id : anchor_ids) {
    EXPECT_TRUE(reconstruction.Image(id).HasPose());
  }

  AnchorSet anchor_set;
  for (const image_t id : anchor_ids) {
    anchor_set.anchors.emplace_back(
        id, reconstruction.Image(id).ProjectionCenter());
  }
  const auto anchors_path = CreateTestDir() / "anchors.txt";
  WriteAnchors(anchors_path, anchor_set);

  AnchorSet read_back;
  ASSERT_TRUE(ReadAnchors(anchors_path, &read_back));
  ASSERT_EQ(read_back.anchors.size(), anchor_set.anchors.size());
  for (size_t i = 0; i < anchor_set.anchors.size(); ++i) {
    EXPECT_EQ(read_back.anchors[i].first, anchor_set.anchors[i].first);
    EXPECT_LT(
        (read_back.anchors[i].second - anchor_set.anchors[i].second).norm(),
        1e-12);
  }

  AnchorSet missing;
  EXPECT_FALSE(ReadAnchors(CreateTestDir() / "nonexistent.txt", &missing));
}

TEST(Anchors, RealignToAnchorsUndoesKnownTransform) {
  SetPRNGSeed(1);
  std::shared_ptr<Database> database;
  const Reconstruction gt =
      MakeSyntheticReconstruction("database_realign.db", &database);

  AnchorSet anchor_set;
  for (const image_t id : SelectAnchorImages(gt)) {
    anchor_set.anchors.emplace_back(id, gt.Image(id).ProjectionCenter());
  }

  Reconstruction drifted = gt;
  const Sim3d drift(1.4,
                    Eigen::Quaterniond(Eigen::AngleAxisd(
                        0.3, Eigen::Vector3d(1, -1, 1).normalized())),
                    Eigen::Vector3d(0.5, 0.2, -0.7));
  drifted.Transform(drift);

  ASSERT_TRUE(RealignToAnchors(anchor_set, &drifted));
  for (const image_t id : gt.RegImageIds()) {
    EXPECT_LT((drifted.Image(id).ProjectionCenter() -
               gt.Image(id).ProjectionCenter())
                  .norm(),
              1e-6);
  }

  // Fewer than 3 registered anchors: refuse and leave untouched.
  AnchorSet tiny;
  tiny.anchors = {anchor_set.anchors[0], anchor_set.anchors[1]};
  Reconstruction copy = gt;
  copy.Transform(drift);
  const Eigen::Vector3d before =
      copy.Image(anchor_set.anchors[0].first).ProjectionCenter();
  EXPECT_FALSE(RealignToAnchors(tiny, &copy));
  EXPECT_LT((copy.Image(anchor_set.anchors[0].first).ProjectionCenter() -
             before)
                .norm(),
            1e-12);
}

// Anchor realignment error stays bounded while rolling realignment drifts:
// simulate a long session where each step's solve adds per-camera noise, and
// compare fitting against the fixed anchors vs the previous (noisy) output.
TEST(Anchors, FixedAnchorsBeatRollingRealignment) {
  SetPRNGSeed(11);
  constexpr int kNumCameras = 12;
  constexpr int kNumSteps = 100;
  constexpr double kNoise = 0.01;

  std::vector<Eigen::Vector3d> gt_centers;
  for (int i = 0; i < kNumCameras; ++i) {
    gt_centers.emplace_back(RandomUniformReal(-2.0, 2.0),
                            RandomUniformReal(-2.0, 2.0),
                            RandomUniformReal(-2.0, 2.0));
  }

  auto noisy = [&](const std::vector<Eigen::Vector3d>& centers) {
    std::vector<Eigen::Vector3d> out = centers;
    for (auto& c : out) {
      c += kNoise * Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                                    RandomUniformReal(-1.0, 1.0),
                                    RandomUniformReal(-1.0, 1.0));
    }
    return out;
  };

  // Each step, the "solver" reproduces the true geometry with fresh noise
  // in an arbitrary drifted gauge (random Sim3). The rolling arm realigns
  // it onto the PREVIOUS aligned output — a noisy, random-walking reference
  // — while the anchored arm realigns onto the fixed ground-truth anchor
  // centres (first 4 cameras).
  auto random_gauge = [&]() {
    return Sim3d(RandomUniformReal(0.8, 1.25),
                 Eigen::Quaterniond(Eigen::AngleAxisd(
                     RandomUniformReal(-0.3, 0.3),
                     Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                                     RandomUniformReal(-1.0, 1.0),
                                     RandomUniformReal(-1.0, 1.0))
                         .normalized())),
                 Eigen::Vector3d(RandomUniformReal(-0.5, 0.5),
                                 RandomUniformReal(-0.5, 0.5),
                                 RandomUniformReal(-0.5, 0.5)));
  };
  auto make_output = [&]() {
    const Sim3d gauge = random_gauge();
    std::vector<Eigen::Vector3d> output = noisy(gt_centers);
    for (auto& c : output) c = gauge * c;
    return output;
  };

  const std::vector<Eigen::Vector3d> anchor_tgt(gt_centers.begin(),
                                                gt_centers.begin() + 4);
  std::vector<Eigen::Vector3d> rolling_ref = gt_centers;
  std::vector<Eigen::Vector3d> anchored_latest = gt_centers;

  for (int step = 0; step < kNumSteps; ++step) {
    // Rolling arm: fit all cameras onto the previous aligned output.
    {
      const std::vector<Eigen::Vector3d> output = make_output();
      Sim3d fit;
      ASSERT_TRUE(EstimateSim3d(output, rolling_ref, fit));
      std::vector<Eigen::Vector3d> aligned;
      for (const auto& c : output) aligned.push_back(fit * c);
      rolling_ref = std::move(aligned);
    }
    // Anchored arm: fit the anchor cameras onto their fixed references.
    {
      const std::vector<Eigen::Vector3d> output = make_output();
      const std::vector<Eigen::Vector3d> anchor_src(output.begin(),
                                                    output.begin() + 4);
      Sim3d fit;
      ASSERT_TRUE(EstimateSim3d(anchor_src, anchor_tgt, fit));
      std::vector<Eigen::Vector3d> aligned;
      for (const auto& c : output) aligned.push_back(fit * c);
      anchored_latest = std::move(aligned);
    }
  }

  // Cumulative geometry error vs ground truth.
  auto mean_error = [&](const std::vector<Eigen::Vector3d>& centers) {
    double sum = 0;
    for (int i = 0; i < kNumCameras; ++i) {
      sum += (centers[i] - gt_centers[i]).norm();
    }
    return sum / kNumCameras;
  };

  // The anchored session must stay near the ground truth; the rolling
  // session is free to random-walk away. Bound the anchored error at a few
  // noise multiples and require it not to exceed the rolling error.
  EXPECT_LT(mean_error(anchored_latest), 10 * kNoise);
  EXPECT_LE(mean_error(anchored_latest), mean_error(rolling_ref));
}

// Full-solve pipeline run with anchors present: realignment targets the
// anchors and the anchor set propagates unchanged.
TEST(IncrementalGlobalPipeline, FullSolveUsesAnchors) {
  SetPRNGSeed(1);
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto prior_path = test_dir / "prior";
  std::filesystem::create_directories(prior_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  const image_t held_out = reg_ids.back();

  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());
  prior.Write(prior_path);

  AnchorSet prior_anchors;
  for (const image_t id : SelectAnchorImages(prior)) {
    prior_anchors.anchors.emplace_back(id,
                                       prior.Image(id).ProjectionCenter());
  }
  WriteAnchors(prior_path / "anchors.txt", prior_anchors);

  IncrementalGlobalPipelineOptions options;
  options.prior_reconstruction_path = prior_path;
  options.mapper.random_seed = 1;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalGlobalPipeline pipeline(
      std::move(options), database, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const Reconstruction& output = *reconstruction_manager->Get(0);
  EXPECT_TRUE(output.Image(held_out).HasPose());
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(output,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-2));

  // Prior anchors propagate unchanged as the fixed reference.
  ASSERT_EQ(pipeline.Anchors().anchors.size(), prior_anchors.anchors.size());
  for (size_t i = 0; i < prior_anchors.anchors.size(); ++i) {
    EXPECT_EQ(pipeline.Anchors().anchors[i].first,
              prior_anchors.anchors[i].first);
  }
}

// Without a prior path the pipeline must behave like a standard global
// reconstruction run.
TEST(IncrementalGlobalPipeline, FallsBackToGlobalWithoutPrior) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  IncrementalGlobalPipelineOptions options;
  options.mapper.random_seed = 1;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalGlobalPipeline pipeline(
      std::move(options), database, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt));
}

}  // namespace
}  // namespace colmap
