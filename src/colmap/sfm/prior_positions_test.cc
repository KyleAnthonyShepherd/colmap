// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// (Same BSD licence header as other COLMAP files.)

#include "colmap/sfm/prior_positions.h"

#include "colmap/math/math.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::shared_ptr<DatabaseCache> CreateCache(const Database& database) {
  DatabaseCache::Options cache_options;
  cache_options.min_num_matches = 0;
  return DatabaseCache::Create(database, cache_options);
}

// Synthesizes a small dataset whose pose priors are exact, so that any
// residual the code under test reports is its own doing rather than noise.
struct PriorFixture {
  Reconstruction reconstruction;
  std::shared_ptr<Database> database;
  std::shared_ptr<DatabaseCache> cache;
};

PriorFixture MakeFixture(const std::filesystem::path& database_path) {
  PriorFixture fixture;
  fixture.database = Database::Open(database_path);
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 8;
  options.num_points3D = 50;
  options.prior_position = true;
  options.prior_position_coordinate_system =
      PosePrior::CoordinateSystem::CARTESIAN;
  SynthesizeDataset(options, &fixture.reconstruction, fixture.database.get());
  fixture.cache = CreateCache(*fixture.database);
  return fixture;
}

TEST(PriorPositions, CollectOnAlignedModelIsIdentity) {
  const auto database_path = CreateTestDir() / "database.db";
  PriorFixture fixture = MakeFixture(database_path);

  const WorldPositionPriors priors = CollectWorldPositionPriors(
      *fixture.cache, fixture.reconstruction, WorldPositionPriorOptions());

  ASSERT_FALSE(priors.Empty());
  EXPECT_EQ(priors.priors.size(), fixture.reconstruction.NumRegImages());
  EXPECT_NEAR(priors.world_from_prior.scale(), 1.0, 1e-6);
  EXPECT_LT(priors.alignment_rmse, 1e-6);

  // The model is already in the priors' frame, so every prior must land on
  // its camera centre.
  for (const image_t image_id : fixture.reconstruction.RegImageIds()) {
    const PosePrior* prior = priors.Find(image_id);
    ASSERT_NE(prior, nullptr);
    EXPECT_LT((prior->position -
               fixture.reconstruction.Image(image_id).ProjectionCenter())
                  .norm(),
              1e-6);
  }
}

TEST(PriorPositions, CollectUndoesAnArbitraryGauge) {
  const auto database_path = CreateTestDir() / "database.db";
  PriorFixture fixture = MakeFixture(database_path);

  // Move the reconstruction into a gauge of its own, exactly what a priorless
  // solve hands us. The priors must follow it, so the residuals stay zero.
  const Sim3d world_from_prior(
      2.5,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(10.0, -4.0, 3.0));
  fixture.reconstruction.Transform(world_from_prior);

  const WorldPositionPriors priors = CollectWorldPositionPriors(
      *fixture.cache, fixture.reconstruction, WorldPositionPriorOptions());

  ASSERT_FALSE(priors.Empty());
  EXPECT_NEAR(priors.world_from_prior.scale(), 2.5, 1e-4);
  for (const image_t image_id : fixture.reconstruction.RegImageIds()) {
    const PosePrior* prior = priors.Find(image_id);
    ASSERT_NE(prior, nullptr);
    EXPECT_LT((prior->position -
               fixture.reconstruction.Image(image_id).ProjectionCenter())
                  .norm(),
              1e-4);
  }
}

TEST(PriorPositions, CollectRejectsNonCartesianPriors) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_frames_per_rig = 8;
  options.num_points3D = 50;
  options.prior_position = true;
  options.prior_position_coordinate_system =
      PosePrior::CoordinateSystem::UNDEFINED;
  SynthesizeDataset(options, &reconstruction, database.get());

  const WorldPositionPriors priors = CollectWorldPositionPriors(
      *CreateCache(*database), reconstruction, WorldPositionPriorOptions());

  // A prior in an unknown frame has no metric meaning, so it must not be
  // silently mixed in.
  EXPECT_TRUE(priors.Empty());
}

TEST(PriorPositions, MahalanobisUsesTheCovariance) {
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  covariance.diagonal() << 0.12 * 0.12, 0.12 * 0.12, 0.016 * 0.016;

  // 1 sigma along each axis.
  EXPECT_NEAR(SquaredMahalanobisDistance(
                  covariance, Eigen::Vector3d(0.12, 0, 0), 1.0),
              1.0,
              1e-9);
  EXPECT_NEAR(SquaredMahalanobisDistance(
                  covariance, Eigen::Vector3d(0, 0, 0.016), 1.0),
              1.0,
              1e-9);

  // The vertical axis is far tighter, so the same metric error is a much
  // larger disagreement there. This is the whole reason the default 1 m
  // fallback must not be applied to a real GNSS fix.
  EXPECT_GT(
      SquaredMahalanobisDistance(covariance, Eigen::Vector3d(0, 0, 0.12), 1.0),
      SquaredMahalanobisDistance(covariance, Eigen::Vector3d(0.12, 0, 0), 1.0));

  // A non-finite covariance falls back to the isotropic sigma.
  const Eigen::Matrix3d nan_cov =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_NEAR(
      SquaredMahalanobisDistance(nan_cov, Eigen::Vector3d(2.0, 0, 0), 2.0),
      1.0,
      1e-9);
}

TEST(PriorPositions, RobustWeight) {
  EXPECT_EQ(PriorPositionRobustWeight(100.0, 7.815, /*use_robust_loss=*/false),
            1.0);
  // Cauchy: 1 / (1 + s / c^2).
  EXPECT_NEAR(PriorPositionRobustWeight(0.0, 2.0, true), 1.0, 1e-12);
  EXPECT_NEAR(PriorPositionRobustWeight(4.0, 2.0, true), 0.5, 1e-12);
  EXPECT_LT(PriorPositionRobustWeight(400.0, 2.0, true),
            PriorPositionRobustWeight(4.0, 2.0, true));
}

TEST(PriorPositions, ResidualsLocateADisplacedFrame) {
  const auto database_path = CreateTestDir() / "database.db";
  PriorFixture fixture = MakeFixture(database_path);

  const WorldPositionPriors priors = CollectWorldPositionPriors(
      *fixture.cache, fixture.reconstruction, WorldPositionPriorOptions());
  ASSERT_FALSE(priors.Empty());

  // Displace exactly one camera, the way an admitted blunder does.
  const image_t displaced_id = fixture.reconstruction.RegImageIds().front();
  {
    Image& image = fixture.reconstruction.Image(displaced_id);
    Rigid3d cam_from_world = image.CamFromWorld();
    // Move the camera centre by 1 m along +x, keeping the rotation.
    const Eigen::Vector3d new_center =
        image.ProjectionCenter() + Eigen::Vector3d(1.0, 0.0, 0.0);
    cam_from_world.translation() = -(cam_from_world.rotation() * new_center);
    image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  }

  const std::vector<PriorPositionResidual> residuals =
      ComputePriorPositionResiduals(fixture.reconstruction,
                                    priors,
                                    /*used_in_ba=*/{displaced_id},
                                    /*fallback_stddev=*/1.0,
                                    /*use_robust_loss=*/true,
                                    /*loss_scale=*/7.815);

  ASSERT_FALSE(residuals.empty());
  const PriorPositionResidualSummary summary =
      SummarizePriorPositionResiduals(residuals);
  EXPECT_EQ(summary.worst_image_id, displaced_id);
  EXPECT_NEAR(summary.worst_norm, 1.0, 1e-4);

  for (const PriorPositionResidual& residual : residuals) {
    if (residual.image_id == displaced_id) {
      EXPECT_TRUE(residual.used_in_ba);
      EXPECT_NEAR(residual.residual.x(), 1.0, 1e-4);
      EXPECT_LT(residual.robust_weight, 1.0);
    } else {
      EXPECT_FALSE(residual.used_in_ba);
      EXPECT_LT(residual.residual_norm, 1e-4);
    }
  }
}

TEST(PriorPositions, WriteReportIsParsable) {
  PriorPositionResidual residual;
  residual.image_id = 7;
  residual.name = "img_0059.jpg";
  residual.prior_position = Eigen::Vector3d(1, 2, 3);
  residual.solved_position = Eigen::Vector3d(1.5, 2, 3);
  residual.residual = Eigen::Vector3d(0.5, 0, 0);
  residual.residual_norm = 0.5;
  residual.sigma = Eigen::Vector3d(0.12, 0.12, 0.016);
  residual.mahalanobis = 4.1667;
  residual.robust_weight = 0.5;
  residual.used_in_ba = true;

  const auto path = CreateTestDir() / "prior_residuals.tsv";
  WritePriorPositionResiduals(path, {residual});

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::string header;
  std::string row;
  ASSERT_TRUE(std::getline(file, header));
  ASSERT_TRUE(std::getline(file, row));
  EXPECT_EQ(std::count(header.begin(), header.end(), '\t'),
            std::count(row.begin(), row.end(), '\t'));
  EXPECT_NE(header.find("residual_norm"), std::string::npos);
  EXPECT_NE(row.find("img_0059.jpg"), std::string::npos);
}

}  // namespace
}  // namespace colmap
