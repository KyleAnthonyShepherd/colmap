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
