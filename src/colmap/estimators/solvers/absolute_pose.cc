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

#include "colmap/estimators/solvers/absolute_pose.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <PoseLib/solvers/p3p.h>
#include <PoseLib/solvers/p4pf.h>

namespace colmap {

P3PEstimator::P3PEstimator(ImgFromCamFunc img_from_cam_func)
    : img_from_cam_func_(std::move(img_from_cam_func)) {}

void P3PEstimator::Estimate(const std::vector<X_t>& points2D,
                            const std::vector<Y_t>& points3D,
                            std::vector<M_t>* cams_from_world) const {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(cams_from_world);

  cams_from_world->clear();

  std::vector<Eigen::Vector3d> rays(3);
  for (int i = 0; i < 3; ++i) {
    rays[i] = points2D[i].camera_ray;
  }

  std::vector<poselib::CameraPose> poses;
  const int num_poses = poselib::p3p(rays, points3D, &poses);

  cams_from_world->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*cams_from_world)[i] = poses[i].Rt();
  }
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& cam_from_world,
                             std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(
      points2D, points3D, cam_from_world, img_from_cam_func_, residuals);
}

namespace {

// Minimal rotation R with R * a = b, for unit vectors a and b.
Eigen::Matrix3d RotationBetweenVectors(const Eigen::Vector3d& a,
                                       const Eigen::Vector3d& b) {
  const Eigen::Vector3d an = a.normalized();
  const Eigen::Vector3d bn = b.normalized();
  const Eigen::Vector3d v = an.cross(bn);
  const double sin_angle = v.norm();
  const double cos_angle = an.dot(bn);
  if (sin_angle < 1e-12) {
    if (cos_angle > 0) return Eigen::Matrix3d::Identity();
    // Anti-parallel: rotate by pi about any axis perpendicular to a.
    Eigen::Vector3d perp(1, 0, 0);
    if (std::abs(an.x()) > 0.9) perp = Eigen::Vector3d(0, 1, 0);
    const Eigen::Vector3d axis = an.cross(perp).normalized();
    const Eigen::Matrix3d K = CrossProductMatrix(axis);
    return Eigen::Matrix3d::Identity() + 2 * K * K;
  }
  const Eigen::Matrix3d K = CrossProductMatrix(v);
  return Eigen::Matrix3d::Identity() + K +
         K * K * ((1 - cos_angle) / (sin_angle * sin_angle));
}

}  // namespace

Up2PEstimator::Up2PEstimator(ImgFromCamFunc img_from_cam_func,
                             const Eigen::Vector3d& gravity_in_world,
                             const Eigen::Vector3d& gravity_in_cam)
    : img_from_cam_func_(std::move(THROW_CHECK_NOTNULL(img_from_cam_func))),
      gravity_in_world_(gravity_in_world.normalized()),
      align_from_world_(RotationBetweenVectors(gravity_in_world.normalized(),
                                               gravity_in_cam.normalized())) {}

void Up2PEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* cams_from_world) const {
  THROW_CHECK_EQ(points2D.size(), 2);
  THROW_CHECK_EQ(points3D.size(), 2);
  THROW_CHECK_NOTNULL(cams_from_world);
  cams_from_world->clear();

  // Re-express each world point in the gravity-aligned basis so that, for the
  // remaining yaw theta about the vertical,
  //     R(theta) * X_i = A_i * cos(theta) + B_i * sin(theta) + C_i
  // which makes the projection equations linear in (cos, sin, t).
  Eigen::Matrix<double, 3, 2> A;
  Eigen::Matrix<double, 3, 2> B;
  Eigen::Matrix<double, 3, 2> C;
  for (int i = 0; i < 2; ++i) {
    const Eigen::Vector3d& X = points3D[i];
    const Eigen::Vector3d c = gravity_in_world_ * gravity_in_world_.dot(X);
    A.col(i) = align_from_world_ * (X - c);
    B.col(i) = align_from_world_ * gravity_in_world_.cross(X);
    C.col(i) = align_from_world_ * c;
  }

  // Each correspondence contributes [ray]_x * (A c + B s + t + C) = 0.
  Eigen::Matrix<double, 6, 5> M;
  Eigen::Matrix<double, 6, 1> rhs;
  for (int i = 0; i < 2; ++i) {
    const Eigen::Matrix3d skew = CrossProductMatrix(points2D[i].camera_ray);
    M.block<3, 1>(3 * i, 0) = skew * A.col(i);
    M.block<3, 1>(3 * i, 1) = skew * B.col(i);
    M.block<3, 3>(3 * i, 2) = skew;
    rhs.segment<3>(3 * i) = -skew * C.col(i);
  }

  // M has rank 4 for two correspondences, so the solutions form a line:
  // a particular least-squares solution plus the one-dimensional null space.
  const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(
      M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix<double, 5, 1> p0 = svd.solve(rhs);
  const Eigen::Matrix<double, 5, 1> singular_values =
      Eigen::Matrix<double, 5, 1>::Zero().cwiseMax(
          svd.singularValues().head<5>());
  if (singular_values(4) > 1e-9 * std::max(1.0, singular_values(0))) {
    // Full rank: no free direction to intersect with the unit circle.
    return;
  }
  const Eigen::Matrix<double, 5, 1> p1 = svd.matrixV().col(4);

  // Intersect the line p0 + lambda * p1 with cos^2 + sin^2 = 1.
  const double qa = p1(0) * p1(0) + p1(1) * p1(1);
  const double qb = 2 * (p0(0) * p1(0) + p0(1) * p1(1));
  const double qc = p0(0) * p0(0) + p0(1) * p0(1) - 1.0;

  std::array<double, 2> lambdas;
  int num_lambdas = 0;
  if (std::abs(qa) < 1e-14) {
    if (std::abs(qb) < 1e-14) return;
    lambdas[num_lambdas++] = -qc / qb;
  } else {
    double disc = qb * qb - 4 * qa * qc;
    if (disc < 0) {
      if (disc > -1e-9) {
        disc = 0.0;
      } else {
        return;
      }
    }
    const double sq = std::sqrt(disc);
    lambdas[num_lambdas++] = (-qb + sq) / (2 * qa);
    lambdas[num_lambdas++] = (-qb - sq) / (2 * qa);
  }

  cams_from_world->reserve(num_lambdas);
  for (int i = 0; i < num_lambdas; ++i) {
    const Eigen::Matrix<double, 5, 1> sol = p0 + lambdas[i] * p1;
    const double norm = std::hypot(sol(0), sol(1));
    if (norm < 1e-9) continue;
    const double cos_theta = sol(0) / norm;
    const double sin_theta = sol(1) / norm;
    const Eigen::Matrix3d yaw(
        Eigen::AngleAxisd(std::atan2(sin_theta, cos_theta), gravity_in_world_));
    M_t cam_from_world;
    cam_from_world.leftCols<3>() = align_from_world_ * yaw;
    cam_from_world.col(3) = sol.tail<3>();
    cams_from_world->push_back(cam_from_world);
  }
}

void Up2PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& cam_from_world,
                              std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(
      points2D, points3D, cam_from_world, img_from_cam_func_, residuals);
}

void P4PFEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* models) {
  THROW_CHECK_EQ(points2D.size(), 4);
  THROW_CHECK_EQ(points3D.size(), 4);
  THROW_CHECK_NOTNULL(models);

  models->clear();

  std::vector<poselib::CameraPose> poses;
  std::vector<double> focals;
  const int num_poses = poselib::p4pf(
      points2D, points3D, &poses, &focals, /*filter_solutions=*/true);

  models->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*models)[i].cam_from_world = poses[i].Rt();
    (*models)[i].focal_length = focals[i];
  }
}

void P4PFEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& model,
                              std::vector<double>* residuals) {
  const size_t num_points2D = points2D.size();
  CHECK_EQ(num_points2D, points3D.size());
  residuals->resize(num_points2D);
  for (size_t i = 0; i < num_points2D; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        model.cam_from_world * points3D[i].homogeneous();
    // Check if 3D point is in front of camera.
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      (*residuals)[i] =
          (model.focal_length * point3D_in_cam.hnormalized() - points2D[i])
              .squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

EPNPEstimator::EPNPEstimator(ImgFromCamFunc img_from_cam_func)
    : img_from_cam_func_(std::move(img_from_cam_func)) {}

void EPNPEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* cams_from_world) {
  THROW_CHECK_GE(points2D.size(), 4);
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(cams_from_world);

  cams_from_world->clear();

  M_t cam_from_world;
  if (!ComputePose(points2D, points3D, &cam_from_world)) {
    return;
  }

  cams_from_world->resize(1);
  (*cams_from_world)[0] = cam_from_world;
}

void EPNPEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& cam_from_world,
                              std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(
      points2D, points3D, cam_from_world, img_from_cam_func_, residuals);
}

bool EPNPEstimator::ComputePose(const std::vector<X_t>& points2D,
                                const std::vector<Y_t>& points3D,
                                Eigen::Matrix3x4d* cam_from_world) {
  points2D_ = &points2D;
  points3D_ = &points3D;

  ChooseControlPoints();

  if (!ComputeBarycentricCoordinates()) {
    return false;
  }

  const Eigen::Matrix<double, Eigen::Dynamic, 12> M = ComputeM();
  const Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;

  Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(
      MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix<double, 12, 12> Ut = svd.matrixU().transpose();

  const Eigen::Matrix<double, 6, 10> L6x10 = ComputeL6x10(Ut);
  const Eigen::Matrix<double, 6, 1> rho = ComputeRho();

  Eigen::Vector4d betas[4];
  std::array<double, 4> reproj_errors;
  std::array<Eigen::Matrix3d, 4> Rs;
  std::array<Eigen::Vector3d, 4> ts;

  FindBetasApprox1(L6x10, rho, &betas[1]);
  RunGaussNewton(L6x10, rho, &betas[1]);
  reproj_errors[1] = ComputeRT(Ut, betas[1], &Rs[1], &ts[1]);

  FindBetasApprox2(L6x10, rho, &betas[2]);
  RunGaussNewton(L6x10, rho, &betas[2]);
  reproj_errors[2] = ComputeRT(Ut, betas[2], &Rs[2], &ts[2]);

  FindBetasApprox3(L6x10, rho, &betas[3]);
  RunGaussNewton(L6x10, rho, &betas[3]);
  reproj_errors[3] = ComputeRT(Ut, betas[3], &Rs[3], &ts[3]);

  int best_idx = 1;
  if (reproj_errors[2] < reproj_errors[1]) {
    best_idx = 2;
  }
  if (reproj_errors[3] < reproj_errors[best_idx]) {
    best_idx = 3;
  }

  cam_from_world->leftCols<3>() = Rs[best_idx];
  cam_from_world->rightCols<1>() = ts[best_idx];

  return true;
}

void EPNPEstimator::ChooseControlPoints() {
  // Take C0 as the reference points centroid:
  cws_[0].setZero();
  for (size_t i = 0; i < points3D_->size(); ++i) {
    cws_[0] += (*points3D_)[i];
  }
  cws_[0] /= points3D_->size();

  Eigen::Matrix<double, Eigen::Dynamic, 3> PW0(points3D_->size(), 3);
  for (size_t i = 0; i < points3D_->size(); ++i) {
    PW0.row(i) = (*points3D_)[i] - cws_[0];
  }

  const Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Vector3d& D = svd.singularValues();
  const Eigen::Matrix3d Ut = svd.matrixU().transpose();

  for (int i = 1; i < 4; ++i) {
    const double k = std::sqrt(D(i - 1) / points3D_->size());
    cws_[i] = cws_[0] + k * Ut.row(i - 1).transpose();
  }
}

bool EPNPEstimator::ComputeBarycentricCoordinates() {
  Eigen::Matrix3d CC;
  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 4; ++j) {
      CC(i, j - 1) = cws_[j][i] - cws_[0][i];
    }
  }

  if (CC.colPivHouseholderQr().rank() < 3) {
    return false;
  }

  const Eigen::Matrix3d CC_inv = CC.inverse();

  alphas_.resize(points2D_->size());
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      alphas_[i][1 + j] = CC_inv(j, 0) * ((*points3D_)[i][0] - cws_[0][0]) +
                          CC_inv(j, 1) * ((*points3D_)[i][1] - cws_[0][1]) +
                          CC_inv(j, 2) * ((*points3D_)[i][2] - cws_[0][2]);
    }
    alphas_[i][0] = 1.0 - alphas_[i][1] - alphas_[i][2] - alphas_[i][3];
  }

  return true;
}

Eigen::Matrix<double, Eigen::Dynamic, 12> EPNPEstimator::ComputeM() {
  Eigen::Matrix<double, Eigen::Dynamic, 12> M(3 * points2D_->size(), 12);
  for (size_t i = 0; i < points3D_->size(); ++i) {
    const Eigen::Vector3d& ray = (*points2D_)[i].camera_ray;
    for (size_t j = 0; j < 4; ++j) {
      M(3 * i, 3 * j) = 0.0;
      M(3 * i, 3 * j + 1) = -alphas_[i][j] * ray.z();
      M(3 * i, 3 * j + 2) = alphas_[i][j] * ray.y();

      M(3 * i + 1, 3 * j) = alphas_[i][j] * ray.z();
      M(3 * i + 1, 3 * j + 1) = 0.0;
      M(3 * i + 1, 3 * j + 2) = -alphas_[i][j] * ray.x();

      M(3 * i + 2, 3 * j) = -alphas_[i][j] * ray.y();
      M(3 * i + 2, 3 * j + 1) = alphas_[i][j] * ray.x();
      M(3 * i + 2, 3 * j + 2) = 0;
    }
  }
  return M;
}

Eigen::Matrix<double, 6, 10> EPNPEstimator::ComputeL6x10(
    const Eigen::Matrix<double, 12, 12>& Ut) {
  Eigen::Matrix<double, 6, 10> L6x10;

  std::array<std::array<Eigen::Vector3d, 6>, 4> dv;
  for (int i = 0; i < 4; ++i) {
    int a = 0, b = 1;
    for (int j = 0; j < 6; ++j) {
      dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
      dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
      dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);

      b += 1;
      if (b > 3) {
        a += 1;
        b = a + 1;
      }
    }
  }

  for (int i = 0; i < 6; ++i) {
    L6x10(i, 0) = dv[0][i].transpose() * dv[0][i];
    L6x10(i, 1) = 2.0 * dv[0][i].transpose() * dv[1][i];
    L6x10(i, 2) = dv[1][i].transpose() * dv[1][i];
    L6x10(i, 3) = 2.0 * dv[0][i].transpose() * dv[2][i];
    L6x10(i, 4) = 2.0 * dv[1][i].transpose() * dv[2][i];
    L6x10(i, 5) = dv[2][i].transpose() * dv[2][i];
    L6x10(i, 6) = 2.0 * dv[0][i].transpose() * dv[3][i];
    L6x10(i, 7) = 2.0 * dv[1][i].transpose() * dv[3][i];
    L6x10(i, 8) = 2.0 * dv[2][i].transpose() * dv[3][i];
    L6x10(i, 9) = dv[3][i].transpose() * dv[3][i];
  }

  return L6x10;
}

Eigen::Matrix<double, 6, 1> EPNPEstimator::ComputeRho() {
  Eigen::Matrix<double, 6, 1> rho;
  rho[0] = (cws_[0] - cws_[1]).squaredNorm();
  rho[1] = (cws_[0] - cws_[2]).squaredNorm();
  rho[2] = (cws_[0] - cws_[3]).squaredNorm();
  rho[3] = (cws_[1] - cws_[2]).squaredNorm();
  rho[4] = (cws_[1] - cws_[3]).squaredNorm();
  rho[5] = (cws_[2] - cws_[3]).squaredNorm();
  return rho;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void EPNPEstimator::FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 4> L_6x4;
  for (int i = 0; i < 6; ++i) {
    L_6x4(i, 0) = L6x10(i, 0);
    L_6x4(i, 1) = L6x10(i, 1);
    L_6x4(i, 2) = L6x10(i, 3);
    L_6x4(i, 3) = L6x10(i, 6);
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(
      L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix<double, 4, 1> b4 = svd.solve(rho);

  if (b4[0] < 0) {
    (*betas)[0] = std::sqrt(-b4[0]);
    (*betas)[1] = -b4[1] / (*betas)[0];
    (*betas)[2] = -b4[2] / (*betas)[0];
    (*betas)[3] = -b4[3] / (*betas)[0];
  } else {
    (*betas)[0] = std::sqrt(b4[0]);
    (*betas)[1] = b4[1] / (*betas)[0];
    (*betas)[2] = b4[2] / (*betas)[0];
    (*betas)[3] = b4[3] / (*betas)[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void EPNPEstimator::FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 3> L_6x3(6, 3);

  for (int i = 0; i < 6; ++i) {
    L_6x3(i, 0) = L6x10(i, 0);
    L_6x3(i, 1) = L6x10(i, 1);
    L_6x3(i, 2) = L6x10(i, 2);
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd(
      L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix<double, 3, 1> b3 = svd.solve(rho);

  if (b3[0] < 0) {
    (*betas)[0] = std::sqrt(-b3[0]);
    (*betas)[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
  } else {
    (*betas)[0] = std::sqrt(b3[0]);
    (*betas)[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) {
    (*betas)[0] = -(*betas)[0];
  }

  (*betas)[2] = 0.0;
  (*betas)[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void EPNPEstimator::FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(
      L6x10.leftCols<5>(), Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix<double, 5, 1> b5 = svd.solve(rho);

  if (b5[0] < 0) {
    (*betas)[0] = std::sqrt(-b5[0]);
    (*betas)[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
  } else {
    (*betas)[0] = std::sqrt(b5[0]);
    (*betas)[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) {
    (*betas)[0] = -(*betas)[0];
  }
  (*betas)[2] = b5[3] / (*betas)[0];
  (*betas)[3] = 0.0;
}

void EPNPEstimator::RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L6x10,
                                   const Eigen::Matrix<double, 6, 1>& rho,
                                   Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 4> A;
  Eigen::Matrix<double, 6, 1> b;

  const int kNumIterations = 5;
  for (int k = 0; k < kNumIterations; ++k) {
    for (int i = 0; i < 6; ++i) {
      A(i, 0) = 2 * L6x10(i, 0) * (*betas)[0] + L6x10(i, 1) * (*betas)[1] +
                L6x10(i, 3) * (*betas)[2] + L6x10(i, 6) * (*betas)[3];
      A(i, 1) = L6x10(i, 1) * (*betas)[0] + 2 * L6x10(i, 2) * (*betas)[1] +
                L6x10(i, 4) * (*betas)[2] + L6x10(i, 7) * (*betas)[3];
      A(i, 2) = L6x10(i, 3) * (*betas)[0] + L6x10(i, 4) * (*betas)[1] +
                2 * L6x10(i, 5) * (*betas)[2] + L6x10(i, 8) * (*betas)[3];
      A(i, 3) = L6x10(i, 6) * (*betas)[0] + L6x10(i, 7) * (*betas)[1] +
                L6x10(i, 8) * (*betas)[2] + 2 * L6x10(i, 9) * (*betas)[3];

      b(i) = rho[i] - (L6x10(i, 0) * (*betas)[0] * (*betas)[0] +
                       L6x10(i, 1) * (*betas)[0] * (*betas)[1] +
                       L6x10(i, 2) * (*betas)[1] * (*betas)[1] +
                       L6x10(i, 3) * (*betas)[0] * (*betas)[2] +
                       L6x10(i, 4) * (*betas)[1] * (*betas)[2] +
                       L6x10(i, 5) * (*betas)[2] * (*betas)[2] +
                       L6x10(i, 6) * (*betas)[0] * (*betas)[3] +
                       L6x10(i, 7) * (*betas)[1] * (*betas)[3] +
                       L6x10(i, 8) * (*betas)[2] * (*betas)[3] +
                       L6x10(i, 9) * (*betas)[3] * (*betas)[3]);
    }

    const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

    (*betas) += x;
  }
}

double EPNPEstimator::ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                                const Eigen::Vector4d& betas,
                                Eigen::Matrix3d* R,
                                Eigen::Vector3d* t) {
  ComputeCcs(betas, Ut);
  ComputePcs();

  SolveForSign();

  EstimateRT(R, t);

  return ComputeTotalError(*R, *t);
}

void EPNPEstimator::ComputeCcs(const Eigen::Vector4d& betas,
                               const Eigen::Matrix<double, 12, 12>& Ut) {
  for (int i = 0; i < 4; ++i) {
    ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0;
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; ++k) {
        ccs_[j][k] += betas[i] * Ut(11 - i, 3 * j + k);
      }
    }
  }
}

void EPNPEstimator::ComputePcs() {
  pcs_.resize(points2D_->size());
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      pcs_[i][j] = alphas_[i][0] * ccs_[0][j] + alphas_[i][1] * ccs_[1][j] +
                   alphas_[i][2] * ccs_[2][j] + alphas_[i][3] * ccs_[3][j];
    }
  }
}

void EPNPEstimator::SolveForSign() {
  if (pcs_[0][2] < 0.0) {
    for (int i = 0; i < 4; ++i) {
      ccs_[i] = -ccs_[i];
    }
    for (size_t i = 0; i < points3D_->size(); ++i) {
      pcs_[i] = -pcs_[i];
    }
  }
}

void EPNPEstimator::EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t) {
  Eigen::Vector3d pc0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < points3D_->size(); ++i) {
    pc0 += pcs_[i];
    pw0 += (*points3D_)[i];
  }
  pc0 /= points3D_->size();
  pw0 /= points3D_->size();

  Eigen::Matrix3d abt = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      abt(j, 0) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][0] - pw0[0]);
      abt(j, 1) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][1] - pw0[1]);
      abt(j, 2) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][2] - pw0[2]);
    }
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix3d& abt_U = svd.matrixU();
  const Eigen::Matrix3d& abt_V = svd.matrixV();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      (*R)(i, j) = abt_U.row(i) * abt_V.row(j).transpose();
    }
  }

  if (R->determinant() < 0) {
    Eigen::Matrix3d Abt_v_prime = abt_V;
    Abt_v_prime.col(2) = -abt_V.col(2);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        (*R)(i, j) = abt_U.row(i) * Abt_v_prime.row(j).transpose();
      }
    }
  }

  *t = pc0 - *R * pw0;
}

double EPNPEstimator::ComputeTotalError(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
  Eigen::Matrix3x4d cam_from_world;
  cam_from_world.leftCols<3>() = R;
  cam_from_world.rightCols<1>() = t;

  std::vector<double> residuals;
  ComputeSquaredReprojectionError(
      *points2D_, *points3D_, cam_from_world, img_from_cam_func_, &residuals);

  double error = 0.0;
  for (const double residual : residuals) {
    error += std::sqrt(residual);
  }

  return error;
}

void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals) {
  const size_t num_points = points2D.size();
  THROW_CHECK_EQ(num_points, points3D.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        cam_from_world * points3D[i].homogeneous();
    const std::optional<Eigen::Vector2d> proj_image_point =
        img_from_cam_func(point3D_in_cam);
    if (proj_image_point) {
      (*residuals)[i] =
          (*proj_image_point - points2D[i].image_point).squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
