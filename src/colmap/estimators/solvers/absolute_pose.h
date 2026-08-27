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

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Function mapping 3D point in the camera frame to 2D point in the image.
// Returns null if the point projection is invalid (e.g., behind the camera).
using ImgFromCamFunc =
    std::function<std::optional<Eigen::Vector2d>(const Eigen::Vector3d&)>;

struct Point2DWithRay {
  // The 2D image point in pixels.
  Eigen::Vector2d image_point;
  // The normalized 3D ray direction in the camera frame.
  Eigen::Vector3d camera_ray;
};

class P3PEstimator {
 public:
  // The 2D image feature observations.
  using X_t = Point2DWithRay;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  // The transformation from the world to the camera frame.
  using M_t = Eigen::Matrix3x4d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  explicit P3PEstimator(ImgFromCamFunc img_from_cam_func);

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D         2D image observations with rays.
  // @param points3D         3D world points.
  // @param cams_from_world  Output vector of 3x4 transformation matrices.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world) const;

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D        2D image observations with rays.
  // @param points3D        3D world points.
  // @param cam_from_world  3x4 projection matrix.
  // @param residuals       Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

 private:
  const ImgFromCamFunc img_from_cam_func_;
};

// Minimal solver for absolute pose with a KNOWN VERTICAL (gravity) direction.
//
// A known "up" fixes 2 of the 3 rotation DOF, leaving one yaw angle plus the
// 3-DOF translation -- 4 unknowns, solvable from 2 correspondences instead of
// P3P's 3. Two properties matter for incremental mapping:
//
//   * RANSAC needs log(1-conf)/log(1-w^s) draws to see one clean minimal
//     sample, so dropping s from 3 to 2 shrinks the budget sharply at low
//     inlier ratios (at w=0.1, ~1.1k draws instead of ~11.5k -- and COLMAP's
//     default max_num_trials is 10k, i.e. P3P can silently run out).
//   * With 4 unknowns instead of 6, a given number of inliers constrains the
//     pose better. This is the property that matters for a weakly-connected
//     image, which may only have a few dozen correspondences in total.
//
// Based on:
//
//    Kukelova, Bujnak, Pajdla. "Closed-Form Solutions to Minimal Absolute
//    Pose Problems with Known Vertical Direction." ACCV 2010.
//    Sweeney et al. "Efficient Computation of Absolute Pose for
//    Gravity-Aware Augmented Reality." ISMAR 2015.
//
// The gravity direction is expected to be imperfect (a phone IMU reading is
// good to a degree or so). This solver therefore only *constrains the search*;
// the caller is expected to polish the winning model with an unconstrained
// refinement (RefineAbsolutePose) so IMU error is never baked into the output.
// A hypothesis that exactly satisfies a slightly-wrong gravity direction is
// systematically biased by roughly focal_length * tan(gravity_error), which is
// tens of pixels for a degree or two, so the RANSAC inlier threshold must be
// widened to match or correct correspondences get rejected.
class Up2PEstimator {
 public:
  // The 2D image feature observations.
  using X_t = Point2DWithRay;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  // The transformation from the world to the camera frame.
  using M_t = Eigen::Matrix3x4d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // `gravity_in_world` and `gravity_in_cam` are the same physical direction
  // expressed in the world frame and in the camera frame respectively. They
  // need not be unit norm; they are normalized internally.
  Up2PEstimator(ImgFromCamFunc img_from_cam_func,
                const Eigen::Vector3d& gravity_in_world,
                const Eigen::Vector3d& gravity_in_cam);

  // Estimate up to two poses from two 2D-3D correspondences.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world) const;

  // Squared reprojection error, in pixels, as for P3PEstimator.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

 private:
  const ImgFromCamFunc img_from_cam_func_;
  // Unit gravity in the world frame; the yaw axis.
  Eigen::Vector3d gravity_in_world_;
  // Minimal rotation taking gravity_in_world_ to gravity_in_cam_.
  Eigen::Matrix3d align_from_world_;
};

// Minimal solver for 6-DOF pose and focal length.
class P4PFEstimator {
 public:
  // The 2D image feature observations.
  // Expected to be normalized by the principal point.
  using X_t = Eigen::Vector2d;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  struct M_t {
    // The transformation from the world to the camera frame.
    Eigen::Matrix3x4d cam_from_world;
    // The focal length of the camera.
    double focal_length = 0.;
  };

  static const int kMinNumSamples = 4;

  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* models);

  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& model,
                        std::vector<double>* residuals);
};

// EPNP solver for the PNP (Perspective-N-Point) problem. The solver needs a
// minimum of 4 2D-3D correspondences.
//
// The algorithm is based on the following paper:
//
//    Lepetit, Vincent, Francesc Moreno-Noguer, and Pascal Fua.
//    "Epnp: An accurate o (n) solution to the pnp problem."
//    International journal of computer vision 81.2 (2009): 155-166.
//
// The implementation is based on their original open-source release, but is
// ported to Eigen and contains several improvements over the original code.
class EPNPEstimator {
 public:
  // The 2D image feature observations.
  using X_t = Point2DWithRay;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  // The transformation from the world to the camera frame.
  using M_t = Eigen::Matrix3x4d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;

  explicit EPNPEstimator(ImgFromCamFunc img_from_cam_func);

  // Estimate the most probable solution of the EPNP problem from a set of
  // four or more 2D-3D point correspondences.
  //
  // @param points2D         2D image observations with rays.
  // @param points3D         3D world points.
  // @param cams_from_world  Output vector of 3x4 transformation matrices.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D        2D image observations with rays.
  // @param points3D        3D world points.
  // @param cam_from_world  3x4 projection matrix.
  // @param residuals       Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

 private:
  bool ComputePose(const std::vector<X_t>& points2D,
                   const std::vector<Y_t>& points3D,
                   Eigen::Matrix3x4d* cam_from_world);

  void ChooseControlPoints();
  bool ComputeBarycentricCoordinates();

  Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM();
  Eigen::Matrix<double, 6, 10> ComputeL6x10(
      const Eigen::Matrix<double, 12, 12>& Ut);
  Eigen::Matrix<double, 6, 1> ComputeRho();

  void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);

  void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L_6x10,
                      const Eigen::Matrix<double, 6, 1>& rho,
                      Eigen::Vector4d* betas);

  double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                   const Eigen::Vector4d& betas,
                   Eigen::Matrix3d* R,
                   Eigen::Vector3d* t);

  void ComputeCcs(const Eigen::Vector4d& betas,
                  const Eigen::Matrix<double, 12, 12>& Ut);
  void ComputePcs();

  void SolveForSign();

  void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t);

  double ComputeTotalError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

  const ImgFromCamFunc img_from_cam_func_;
  const std::vector<X_t>* points2D_ = nullptr;
  const std::vector<Y_t>* points3D_ = nullptr;
  std::vector<Eigen::Vector3d> pcs_;
  std::vector<Eigen::Vector4d> alphas_;
  std::array<Eigen::Vector3d, 4> cws_;
  std::array<Eigen::Vector3d, 4> ccs_;
};

// Compute squared reprojection error in pixels.
void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals);

}  // namespace colmap
