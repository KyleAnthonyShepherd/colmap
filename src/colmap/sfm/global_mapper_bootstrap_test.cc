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

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/testing.h"

#include <algorithm>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

double RotationErrorDeg(const Rigid3d& pose1, const Rigid3d& pose2) {
  return Eigen::Quaterniond(pose1.rotation())
             .angularDistance(pose2.rotation()) *
         180.0 / M_PI;
}

Eigen::Vector3d Center(const Rigid3d& cam_from_world) {
  return Inverse(cam_from_world).translation();
}

// ── SolvePoseFromPriorEdges (pure solver, fabricated evidence) ─────────────

// Ground-truth setup: a new camera and several prior cameras at known poses.
struct SolverFixture {
  Rigid3d new_pose;                // gt cam_from_world of the new image
  std::vector<Rigid3d> prior_poses;

  SolverFixture() {
    new_pose.rotation() =
        Eigen::Quaterniond(Eigen::AngleAxisd(
            0.4, Eigen::Vector3d(1, 2, -1).normalized()));
    new_pose.translation() = Eigen::Vector3d(0.3, -0.2, 1.5);

    const std::vector<Eigen::Vector3d> centers = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.2}, {-1.0, 0.5, 0.1}, {0.5, -1.0, 0.4}};
    for (size_t i = 0; i < centers.size(); ++i) {
      Rigid3d pose;
      pose.rotation() = Eigen::Quaterniond(Eigen::AngleAxisd(
          0.1 * (i + 1), Eigen::Vector3d(1, i + 1.0, 0.5).normalized()));
      pose.translation() = -(pose.rotation() * centers[i]);
      prior_poses.push_back(pose);
    }
  }

  // Builds an exact observation for prior i with the given edge ordering.
  BootstrapEdgeObservation MakeObservation(const size_t i,
                                           const bool prior_is_cam1,
                                           const double weight = 100.0) const {
    BootstrapEdgeObservation obs;
    obs.prior_is_cam1 = prior_is_cam1;
    obs.prior_cam_from_world = prior_poses[i];
    obs.weight = weight;
    if (prior_is_cam1) {
      // cam1 = prior, cam2 = new.
      obs.cam2_from_cam1 = new_pose * Inverse(prior_poses[i]);
    } else {
      // cam1 = new, cam2 = prior.
      obs.cam2_from_cam1 = prior_poses[i] * Inverse(new_pose);
    }
    return obs;
  }
};

TEST(SolvePoseFromPriorEdges, ExactRecoveryPriorIsCam1) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/true));
  }
  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 1e-6);
}

TEST(SolvePoseFromPriorEdges, ExactRecoveryPriorIsCam2) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/false));
  }
  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 1e-6);
}

TEST(SolvePoseFromPriorEdges, ExactRecoveryMixedOrderings) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/(i % 2 == 0)));
  }
  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 1e-6);
}

TEST(SolvePoseFromPriorEdges, NoisyObservations) {
  SetPRNGSeed(7);
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    BootstrapEdgeObservation o = f.MakeObservation(i, i % 2 == 0);
    // ~1 degree of rotation noise on each edge.
    const Eigen::Vector3d axis = Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                                                 RandomUniformReal(-1.0, 1.0),
                                                 RandomUniformReal(-1.0, 1.0))
                                     .normalized();
    o.cam2_from_cam1.rotation() =
        (Eigen::Quaterniond(Eigen::AngleAxisd(1.0 * M_PI / 180.0, axis)) *
         o.cam2_from_cam1.rotation())
            .normalized();
    obs.push_back(o);
  }
  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 2.0);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 0.2);
}

TEST(SolvePoseFromPriorEdges, GrossOutlierCandidateRejected) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/true,
                                    /*weight=*/100.0));
  }
  // A gross outlier: rotation off by 40 degrees, moderate weight.
  BootstrapEdgeObservation outlier =
      f.MakeObservation(0, /*prior_is_cam1=*/true, /*weight=*/50.0);
  outlier.cam2_from_cam1.rotation() =
      (Eigen::Quaterniond(Eigen::AngleAxisd(
           40.0 * M_PI / 180.0, Eigen::Vector3d::UnitY())) *
       outlier.cam2_from_cam1.rotation())
          .normalized();
  obs.push_back(outlier);

  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  // The outlier is discarded by the re-pass, so recovery is exact.
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
}

TEST(SolvePoseFromPriorEdges, DegenerateTranslationKeepsRotation) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < 3; ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/true));
  }
  // A pure-rotation edge: contributes a rotation candidate but no ray.
  BootstrapEdgeObservation degenerate =
      f.MakeObservation(3, /*prior_is_cam1=*/true);
  degenerate.cam2_from_cam1.translation() = Eigen::Vector3d::Zero();
  obs.push_back(degenerate);

  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 1e-6);
}

TEST(SolvePoseFromPriorEdges, SingleRayFallsBackToCentroid) {
  SolverFixture f;
  const std::vector<BootstrapEdgeObservation> obs = {
      f.MakeObservation(0, /*prior_is_cam1=*/true)};
  const auto pose = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(pose.has_value());
  // Rotation is exact; the centre cannot be resolved from one ray and falls
  // back to the (single) prior centre.
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.prior_poses[0])).norm(), 1e-9);
}

TEST(SolvePoseFromPriorEdges, EmptyObservationsReturnsNullopt) {
  EXPECT_FALSE(
      SolvePoseFromPriorEdges({}, /*max_candidate_deg=*/10.0).has_value());
}

// ── Gravity gate ───────────────────────────────────────────────────────────

BootstrapGravityGate MakeGravityGate(const Rigid3d& new_pose,
                                     const Eigen::Vector3d& gravity_in_world,
                                     const double max_error_deg = 15.0) {
  BootstrapGravityGate gate;
  gate.gravity_in_world = gravity_in_world;
  // Measured gravity = ground-truth gravity in the new camera's frame.
  gate.gravity_in_new_cam = new_pose.rotation() * gravity_in_world;
  gate.max_error_deg = max_error_deg;
  return gate;
}

TEST(SolvePoseFromPriorEdges, GravityGateOverridesInlierWeight) {
  SolverFixture f;
  const Eigen::Vector3d gravity_in_world = Eigen::Vector3d::UnitY();

  // Two candidates only: the HIGHER-weight one has a rotation 30 degrees off
  // (tilting the implied gravity), the lower-weight one is correct. With so
  // few candidates the weighted mean + distance-from-mean re-pass follows
  // the heavy outlier; the gravity gate must reject it instead.
  std::vector<BootstrapEdgeObservation> obs;

  const BootstrapGravityGate gate = MakeGravityGate(f.new_pose,
                                                    gravity_in_world);

  BootstrapEdgeObservation outlier =
      f.MakeObservation(0, /*prior_is_cam1=*/true, /*weight=*/500.0);
  // Perturb about an axis orthogonal to the measured gravity so the implied
  // gravity moves by the full perturbation angle.
  const Eigen::Vector3d perturb_axis = gate.gravity_in_new_cam.unitOrthogonal();
  outlier.cam2_from_cam1.rotation() =
      (Eigen::Quaterniond(Eigen::AngleAxisd(30.0 * M_PI / 180.0,
                                            perturb_axis)) *
       outlier.cam2_from_cam1.rotation())
          .normalized();
  obs.push_back(outlier);
  obs.push_back(f.MakeObservation(1, /*prior_is_cam1=*/true, /*weight=*/50.0));
  obs.push_back(f.MakeObservation(2, /*prior_is_cam1=*/true, /*weight=*/40.0));

  // Without the gate the heavy outlier drags the mean far off.
  const auto ungated = SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0);
  ASSERT_TRUE(ungated.has_value());
  EXPECT_GT(RotationErrorDeg(*ungated, f.new_pose), 5.0);

  // With the gate, the outlier candidate is discarded and recovery is exact.
  const auto gated =
      SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0, gate);
  ASSERT_TRUE(gated.has_value());
  EXPECT_LT(RotationErrorDeg(*gated, f.new_pose), 1e-6);
}

TEST(SolvePoseFromPriorEdges, AllCandidatesViolatingGravityFallsBack) {
  SolverFixture f;
  std::vector<BootstrapEdgeObservation> obs;
  for (size_t i = 0; i < f.prior_poses.size(); ++i) {
    obs.push_back(f.MakeObservation(i, /*prior_is_cam1=*/true));
  }

  // A gate whose measured gravity disagrees with every candidate: solving
  // must proceed ungated rather than fail, and the observations are exact so
  // recovery stays exact.
  BootstrapGravityGate bogus_gate;
  bogus_gate.gravity_in_world = Eigen::Vector3d::UnitY();
  bogus_gate.gravity_in_new_cam =
      -(f.new_pose.rotation() * Eigen::Vector3d::UnitY());
  bogus_gate.max_error_deg = 15.0;

  const auto pose =
      SolvePoseFromPriorEdges(obs, /*max_candidate_deg=*/10.0, bogus_gate);
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(RotationErrorDeg(*pose, f.new_pose), 1e-6);
  EXPECT_LT((Center(*pose) - Center(f.new_pose)).norm(), 1e-6);
}

// ── BootstrapNewImagePoses (graph walk, synthetic database) ────────────────

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

// Synthesizes a dataset, holds out one registered image, and verifies that
// BootstrapNewImagePoses recovers its ground-truth pose from the pose graph.
void TestHoldOneOut(const size_t hold_out_rank) {
  SetPRNGSeed(1);
  const auto database_path =
      CreateTestDir() / ("database_" + std::to_string(hold_out_rank) + ".db");

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

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  ASSERT_GT(reg_ids.size(), hold_out_rank);
  const image_t held_out = reg_ids[hold_out_rank];

  // The prior reconstruction is the ground truth minus the held-out image.
  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  mapper.LoadPriorPoses(prior);
  EXPECT_EQ(mapper.PriorImageIds().size(), reg_ids.size() - 1);
  EXPECT_EQ(mapper.PriorImageIds().count(held_out), 0);

  const GlobalMapperOptions options;
  const std::unordered_set<image_t> bootstrapped =
      mapper.BootstrapNewImagePoses(options);

  ASSERT_EQ(bootstrapped.size(), 1);
  EXPECT_EQ(bootstrapped.count(held_out), 1);
  ASSERT_TRUE(reconstruction->Image(held_out).HasPose());

  const Rigid3d& recovered = reconstruction->Image(held_out).CamFromWorld();
  const Rigid3d& gt_pose = gt_reconstruction.Image(held_out).CamFromWorld();
  EXPECT_LT(RotationErrorDeg(recovered, gt_pose), 1e-2);
  EXPECT_LT((Center(recovered) - Center(gt_pose)).norm(), 1e-3);
}

// Holding out the lowest id exercises the (new=cam1, prior=cam2) ordering
// for every edge; the highest id exercises (prior=cam1, new=cam2); a middle
// id exercises both in one solve.
TEST(BootstrapNewImagePoses, HoldOutFirstImage) { TestHoldOneOut(0); }
TEST(BootstrapNewImagePoses, HoldOutMiddleImage) { TestHoldOneOut(3); }
TEST(BootstrapNewImagePoses, HoldOutLastImage) { TestHoldOneOut(5); }

TEST(BootstrapNewImagePoses, WithGravityPriorsInDatabase) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database_gravity.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  synthetic_dataset_options.prior_gravity = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  const image_t held_out = reg_ids[reg_ids.size() / 2];

  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  mapper.LoadPriorPoses(prior);

  // The gravity gate is active (priors exist for every image) and all
  // observations are consistent, so recovery must remain exact.
  const GlobalMapperOptions options;
  ASSERT_GT(options.bootstrap_max_gravity_error_deg, 0);
  const std::unordered_set<image_t> bootstrapped =
      mapper.BootstrapNewImagePoses(options);

  ASSERT_EQ(bootstrapped.count(held_out), 1);
  const Rigid3d& recovered = reconstruction->Image(held_out).CamFromWorld();
  const Rigid3d& gt_pose = gt_reconstruction.Image(held_out).CamFromWorld();
  EXPECT_LT(RotationErrorDeg(recovered, gt_pose), 1e-2);
  EXPECT_LT((Center(recovered) - Center(gt_pose)).norm(), 1e-3);
}

TEST(BootstrapNewImagePoses, MinInliersTooHighBootstrapsNothing) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());
  const image_t held_out = reg_ids.back();

  Reconstruction prior = gt_reconstruction;
  prior.DeRegisterFrame(prior.Image(held_out).FrameId());

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  mapper.LoadPriorPoses(prior);

  GlobalMapperOptions options;
  options.bootstrap_min_inliers = 1000000;
  EXPECT_TRUE(mapper.BootstrapNewImagePoses(options).empty());
  EXPECT_FALSE(reconstruction->Image(held_out).HasPose());
}

// ── SolveIncrementalWindowed ───────────────────────────────────────────────

TEST(SolveIncrementalWindowed, AddOneImage) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database_windowed.db";

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

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  mapper.LoadPriorPoses(prior);

  GlobalMapperOptions options;
  options.random_seed = 1;
  options.optimize_window_size = 4;
  const std::unordered_set<image_t> bootstrapped =
      mapper.BootstrapNewImagePoses(options);
  ASSERT_EQ(bootstrapped.count(held_out), 1);

  ASSERT_TRUE(
      mapper.SolveIncrementalWindowed(options, prior, bootstrapped));

  // The new image's pose is refined near ground truth and the output stays
  // in the prior (= ground-truth) frame without any realignment.
  const Rigid3d& recovered = reconstruction->Image(held_out).CamFromWorld();
  const Rigid3d& gt_pose = gt_reconstruction.Image(held_out).CamFromWorld();
  EXPECT_LT(RotationErrorDeg(recovered, gt_pose), 1e-1);
  EXPECT_LT((Center(recovered) - Center(gt_pose)).norm(), 1e-2);

  // Prior structure was imported and the new image observes 3D points.
  EXPECT_GT(reconstruction->NumPoints3D(), 50);
  EXPECT_GT(reconstruction->Image(held_out).NumPoints3D(), 10);

  // Prior poses stayed in the prior frame.
  for (const image_t id : mapper.PriorImageIds()) {
    EXPECT_LT((Center(reconstruction->Image(id).CamFromWorld()) -
               Center(gt_reconstruction.Image(id).CamFromWorld()))
                  .norm(),
              1e-2);
  }
}

// Drip-feeds images one at a time through the windowed path, mirroring the
// production server loop (each add starts from the previous output).
TEST(SolveIncrementalWindowed, DripFeedStaysNearGroundTruth) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database_dripfeed.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 12;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::vector<image_t> reg_ids = gt_reconstruction.RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end());

  // Start from a prior containing the first 5 images.
  Reconstruction prior = gt_reconstruction;
  for (size_t i = 5; i < reg_ids.size(); ++i) {
    prior.DeRegisterFrame(prior.Image(reg_ids[i]).FrameId());
  }

  GlobalMapperOptions options;
  options.random_seed = 1;
  options.optimize_window_size = 4;

  for (size_t i = 5; i < reg_ids.size(); ++i) {
    const image_t new_id = reg_ids[i];

    // Restrict the database cache to the images "uploaded" so far, as in
    // production where the database grows by one image per solve. This
    // leaves exactly one unregistered image: the new one.
    DatabaseCache::Options cache_options;
    std::vector<std::string> visible_names;
    for (size_t j = 0; j <= i; ++j) {
      visible_names.push_back(gt_reconstruction.Image(reg_ids[j]).Name());
    }
    cache_options.image_names = {visible_names.begin(), visible_names.end()};
    const auto database_cache =
        DatabaseCache::Create(*database, cache_options);

    auto reconstruction = std::make_shared<Reconstruction>();
    GlobalMapper mapper(database_cache);
    mapper.BeginReconstruction(reconstruction);
    mapper.LoadPriorPoses(prior);

    const std::unordered_set<image_t> bootstrapped =
        mapper.BootstrapNewImagePoses(options);
    ASSERT_EQ(bootstrapped.size(), 1) << "at step " << i;
    ASSERT_EQ(bootstrapped.count(new_id), 1)
        << "image " << new_id << " not bootstrapped at step " << i;

    ASSERT_TRUE(
        mapper.SolveIncrementalWindowed(options, prior, bootstrapped));

    prior = *reconstruction;
  }

  // After drip-feeding all remaining images, every pose must be near ground
  // truth in the ground-truth frame (no realignment applied anywhere).
  for (const image_t id : reg_ids) {
    ASSERT_TRUE(prior.Image(id).HasPose());
    EXPECT_LT(RotationErrorDeg(prior.Image(id).CamFromWorld(),
                               gt_reconstruction.Image(id).CamFromWorld()),
              0.5)
        << "image " << id;
    EXPECT_LT((Center(prior.Image(id).CamFromWorld()) -
               Center(gt_reconstruction.Image(id).CamFromWorld()))
                  .norm(),
              0.05)
        << "image " << id;
  }
}

// ── Sim3 realignment building block ────────────────────────────────────────

TEST(Sim3Realignment, RecoversKnownTransform) {
  SetPRNGSeed(3);
  // A set of well-spread camera centres.
  std::vector<Eigen::Vector3d> centers;
  for (int i = 0; i < 10; ++i) {
    centers.emplace_back(RandomUniformReal(-2.0, 2.0),
                         RandomUniformReal(-2.0, 2.0),
                         RandomUniformReal(-2.0, 2.0));
  }

  // Apply a known Sim3.
  Sim3d gt_transform(1.7,
                     Eigen::Quaterniond(Eigen::AngleAxisd(
                         0.5, Eigen::Vector3d(1, 1, -2).normalized())),
                     Eigen::Vector3d(0.4, -1.0, 2.0));
  std::vector<Eigen::Vector3d> transformed;
  for (const auto& c : centers) {
    transformed.push_back(gt_transform * c);
  }

  // Fitting transformed → original must recover the inverse.
  Sim3d recovered;
  ASSERT_TRUE(EstimateSim3d(transformed, centers, recovered));
  const Sim3d expected = Inverse(gt_transform);
  EXPECT_NEAR(recovered.scale(), expected.scale(), 1e-6);
  EXPECT_LT(Eigen::Quaterniond(recovered.rotation())
                .angularDistance(expected.rotation()),
            1e-6);
  EXPECT_LT((recovered.translation() - expected.translation()).norm(), 1e-6);

  // Fewer than 3 points throws — callers (the pipeline's realignment step)
  // must guard the point count before calling.
  Sim3d degenerate;
  EXPECT_ANY_THROW(EstimateSim3d({transformed[0], transformed[1]},
                                 {centers[0], centers[1]},
                                 degenerate));
}

}  // namespace
}  // namespace colmap
