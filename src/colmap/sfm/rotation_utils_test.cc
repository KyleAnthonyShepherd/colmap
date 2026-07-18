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

#include "colmap/sfm/rotation_utils.h"

#include "colmap/math/random.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

double AngularErrorDeg(const Eigen::Quaterniond& q1,
                       const Eigen::Quaterniond& q2) {
  return q1.angularDistance(q2) * 180.0 / M_PI;
}

TEST(BestCandidateIdxByWeight, PicksHighestWeight) {
  EXPECT_EQ(BestCandidateIdxByWeight({1.0}), 0);
  EXPECT_EQ(BestCandidateIdxByWeight({1.0, 3.0, 2.0}), 1);
  EXPECT_EQ(BestCandidateIdxByWeight({5.0, 3.0, 2.0}), 0);
  EXPECT_EQ(BestCandidateIdxByWeight({1.0, 3.0, 7.0}), 2);
  // Ties resolve to the first occurrence (std::max_element semantics).
  EXPECT_EQ(BestCandidateIdxByWeight({2.0, 2.0}), 0);
}

TEST(KarcherMeanSO3, SingleQuaternion) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Quaterniond mean = KarcherMeanSO3({q}, {1.0}, q);
  EXPECT_LT(AngularErrorDeg(mean, q), 1e-9);
}

TEST(KarcherMeanSO3, IdenticalQuaternions) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(1.1, Eigen::Vector3d(0, 1, 1).normalized()));
  const Eigen::Quaterniond mean =
      KarcherMeanSO3({q, q, q}, {1.0, 2.0, 3.0}, q);
  EXPECT_LT(AngularErrorDeg(mean, q), 1e-9);
}

TEST(KarcherMeanSO3, WeightedTwoQuaternionGeodesic) {
  // The weighted Karcher mean of two rotations lies on the geodesic between
  // them at parameter w2 / (w1 + w2), i.e. it matches slerp.
  const Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond q2(
      Eigen::AngleAxisd(0.8, Eigen::Vector3d::UnitZ()));

  const double w1 = 3.0;
  const double w2 = 1.0;
  const Eigen::Quaterniond expected = q1.slerp(w2 / (w1 + w2), q2);

  const Eigen::Quaterniond mean = KarcherMeanSO3({q1, q2}, {w1, w2}, q1);
  EXPECT_LT(AngularErrorDeg(mean, expected), 1e-6);
}

TEST(KarcherMeanSO3, AntipodalRepresentationInvariance) {
  // q and -q represent the same rotation; the mean must not depend on which
  // representative is passed in.
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX()));
  const Eigen::Quaterniond q2(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond q2_neg = q2;
  q2_neg.coeffs() = -q2_neg.coeffs();

  const Eigen::Quaterniond mean_pos =
      KarcherMeanSO3({q1, q2}, {1.0, 1.0}, q1);
  const Eigen::Quaterniond mean_neg =
      KarcherMeanSO3({q1, q2_neg}, {1.0, 1.0}, q1);

  EXPECT_LT(AngularErrorDeg(mean_pos, mean_neg), 1e-6);
}

TEST(KarcherMeanSO3, ConvergesForRandomCluster) {
  SetPRNGSeed(42);
  const Eigen::Quaterniond nominal(
      Eigen::AngleAxisd(0.6, Eigen::Vector3d(1, -1, 2).normalized()));

  // 50 random rotations within 30 degrees of the nominal.
  std::vector<Eigen::Quaterniond> qs;
  std::vector<double> weights;
  for (int i = 0; i < 50; ++i) {
    const double angle = RandomUniformReal(0.0, 30.0 * M_PI / 180.0);
    const Eigen::Vector3d axis = Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                                                 RandomUniformReal(-1.0, 1.0),
                                                 RandomUniformReal(-1.0, 1.0))
                                     .normalized();
    qs.push_back(nominal * Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis)));
    weights.push_back(RandomUniformReal(1.0, 100.0));
  }

  const Eigen::Quaterniond mean =
      KarcherMeanSO3(qs, weights, qs[BestCandidateIdxByWeight(weights)]);

  // The mean of a 30-degree cluster must lie within the cluster.
  EXPECT_LT(AngularErrorDeg(mean, nominal), 30.0);

  // The mean must be a fixed point: re-running seeded from the mean itself
  // converges immediately to the same rotation.
  const Eigen::Quaterniond mean2 = KarcherMeanSO3(qs, weights, mean);
  EXPECT_LT(AngularErrorDeg(mean, mean2), 1e-6);
}

TEST(KarcherMeanSO3, ZeroTotalWeightReturnsInit) {
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX()));
  const Eigen::Quaterniond q_init(
      Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond mean = KarcherMeanSO3({q1}, {0.0}, q_init);
  EXPECT_LT(AngularErrorDeg(mean, q_init), 1e-9);
}

}  // namespace
}  // namespace colmap
