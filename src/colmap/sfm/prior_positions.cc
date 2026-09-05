// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// (Same BSD licence header as other COLMAP files.)

#include "colmap/sfm/prior_positions.h"

#include "colmap/estimators/alignment.h"
#include "colmap/math/math.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <Eigen/Cholesky>

namespace colmap {
namespace {

// Rotates and scales a covariance through a Sim3: cov' = (sR) cov (sR)^T.
Eigen::Matrix3d TransformCovariance(const Sim3d& b_from_a,
                                    const Eigen::Matrix3d& cov_in_a) {
  const Eigen::Matrix3d scaled_rotation =
      b_from_a.scale() * b_from_a.rotation().toRotationMatrix();
  return scaled_rotation * cov_in_a * scaled_rotation.transpose();
}

}  // namespace

const PosePrior* WorldPositionPriors::Find(const image_t image_id) const {
  const auto it = priors.find(image_id);
  return (it == priors.end()) ? nullptr : &it->second;
}

WorldPositionPriors CollectWorldPositionPriors(
    const DatabaseCache& database_cache,
    const Reconstruction& reconstruction,
    const WorldPositionPriorOptions& options) {
  WorldPositionPriors result;

  // 1. Filter the database's priors down to the ones this reconstruction can
  //    actually use: camera priors, with a position, in a metric frame, for
  //    an image the reconstruction knows about.
  std::vector<PosePrior> usable;
  size_t num_non_cartesian = 0;
  size_t num_without_position = 0;
  for (const PosePrior& pose_prior : database_cache.PosePriors()) {
    if (pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    if (!reconstruction.ExistsImage(pose_prior.corr_data_id.id)) {
      continue;
    }
    if (!pose_prior.HasPosition()) {
      ++num_without_position;
      continue;
    }
    if (pose_prior.coordinate_system !=
        PosePrior::CoordinateSystem::CARTESIAN) {
      ++num_non_cartesian;
      continue;
    }
    usable.push_back(pose_prior);
  }

  if (num_non_cartesian > 0) {
    LOG(WARNING)
        << "CollectWorldPositionPriors: ignoring " << num_non_cartesian
        << " position prior(s) that are not in a CARTESIAN coordinate "
           "system. Write ENU metres with coordinate_system=1, or enable "
           "pose-prior ENU conversion so WGS84 priors are converted first; a "
           "prior in an unknown frame cannot be compared with a camera "
           "centre.";
  }
  if (num_without_position > 0) {
    VLOG(2) << "CollectWorldPositionPriors: " << num_without_position
            << " prior(s) carry no position (gravity only).";
  }

  if (usable.empty()) {
    LOG(WARNING) << "CollectWorldPositionPriors: no usable position priors.";
    return result;
  }

  // 2. Fit the prior<-world Sim3 over the registered images, exactly the way
  //    PosePriorBundleAdjuster does, then invert it. The fit is over the
  //    already-registered images only, so a new image being tested against
  //    the priors cannot influence the frame it is tested in.
  size_t num_registered_with_prior = 0;
  {
    std::unordered_set<image_t> prior_image_ids;
    for (const PosePrior& pose_prior : usable) {
      prior_image_ids.insert(pose_prior.corr_data_id.id);
    }
    for (const image_t image_id : reconstruction.RegImageIds()) {
      if (prior_image_ids.count(image_id) > 0) ++num_registered_with_prior;
    }
  }

  const size_t min_alignment_images =
      static_cast<size_t>(std::max(3, options.min_alignment_images));
  if (num_registered_with_prior < min_alignment_images) {
    LOG(WARNING) << "CollectWorldPositionPriors: only "
                 << num_registered_with_prior
                 << " registered image(s) carry a position prior; at least "
                 << min_alignment_images
                 << " are needed to place the priors in the reconstruction's "
                    "frame. Position priors are not used for this solve.";
    return result;
  }

  RANSACOptions ransac_options = options.alignment_ransac_options;
  if (ransac_options.max_error <= 0) {
    std::vector<double> rms_vars;
    rms_vars.reserve(usable.size());
    for (const PosePrior& pose_prior : usable) {
      if (!pose_prior.HasPositionCov()) continue;
      const double trace = pose_prior.position_covariance.trace();
      if (trace <= 0.0) continue;
      rms_vars.push_back(trace / 3.0);
    }
    if (rms_vars.empty()) {
      rms_vars.push_back(options.fallback_stddev * options.fallback_stddev);
    }
    ransac_options.max_error =
        std::sqrt(kChiSquare95ThreeDof * Median(rms_vars));
  }

  Sim3d prior_from_world;
  if (!AlignReconstructionToPosePriors(
          reconstruction, usable, ransac_options, &prior_from_world)) {
    LOG(WARNING) << "CollectWorldPositionPriors: could not align the "
                    "reconstruction to its position priors; position priors "
                    "are not used for this solve.";
    return result;
  }

  // 3. Report how well the fit landed, in the priors' own metric frame. A
  //    large RMSE here is the signal that the model and the priors disagree
  //    globally, which the per-image residual report then localises.
  {
    std::vector<double> sq_errors;
    for (const PosePrior& pose_prior : usable) {
      const Image& image = reconstruction.Image(pose_prior.corr_data_id.id);
      if (!image.HasPose()) continue;
      sq_errors.push_back(
          ((prior_from_world * image.ProjectionCenter()) - pose_prior.position)
              .squaredNorm());
    }
    if (!sq_errors.empty()) {
      result.alignment_rmse = std::sqrt(Mean(sq_errors));
    }
  }

  result.world_from_prior = Inverse(prior_from_world);
  result.num_alignment_images = num_registered_with_prior;

  // 4. Move every prior into the world frame, covariance included.
  const Eigen::Matrix3d fallback_cov =
      (options.fallback_stddev * options.fallback_stddev) *
      Eigen::Matrix3d::Identity();
  for (PosePrior pose_prior : usable) {
    const Eigen::Matrix3d cov_in_prior = pose_prior.HasPositionCov()
                                             ? pose_prior.position_covariance
                                             : fallback_cov;
    const image_t image_id = pose_prior.corr_data_id.id;
    pose_prior.position = result.world_from_prior * pose_prior.position;
    pose_prior.position_covariance =
        TransformCovariance(result.world_from_prior, cov_in_prior);
    result.priors[image_id] = std::move(pose_prior);
  }

  LOG(INFO) << "CollectWorldPositionPriors: " << result.priors.size()
            << " position prior(s) placed in the reconstruction frame from "
            << num_registered_with_prior << " registered image(s); scale "
            << result.world_from_prior.scale() << ", alignment rmse "
            << result.alignment_rmse << " m.";

  return result;
}

double SquaredMahalanobisDistance(const Eigen::Matrix3d& covariance,
                                  const Eigen::Vector3d& residual,
                                  const double fallback_stddev) {
  const double sigma = std::max(std::abs(fallback_stddev), 1e-12);
  Eigen::Matrix3d cov = covariance;
  if (!cov.allFinite()) {
    cov = (sigma * sigma) * Eigen::Matrix3d::Identity();
  }
  const Eigen::LLT<Eigen::Matrix3d> llt(cov);
  if (llt.info() != Eigen::Success) {
    // Not positive definite: fall back to an isotropic sigma rather than
    // returning a meaningless (or negative) distance.
    return residual.squaredNorm() / (sigma * sigma);
  }
  const Eigen::Vector3d whitened = llt.matrixL().solve(residual);
  return whitened.squaredNorm();
}

double PriorPositionRobustWeight(const double squared_mahalanobis,
                                 const double loss_scale,
                                 const bool use_robust_loss) {
  if (!use_robust_loss || loss_scale <= 0.0) return 1.0;
  // Cauchy: rho(s) = c^2 log(1 + s / c^2), rho'(s) = 1 / (1 + s / c^2).
  return 1.0 / (1.0 + squared_mahalanobis / (loss_scale * loss_scale));
}

std::vector<PriorPositionResidual> ComputePriorPositionResiduals(
    const Reconstruction& reconstruction,
    const WorldPositionPriors& world_priors,
    const std::unordered_set<image_t>& used_in_ba,
    const double fallback_stddev,
    const bool use_robust_loss,
    const double loss_scale) {
  std::vector<PriorPositionResidual> residuals;
  const Eigen::Matrix3d fallback_cov =
      (fallback_stddev * fallback_stddev) * Eigen::Matrix3d::Identity();

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const PosePrior* prior = world_priors.Find(image_id);
    if (prior == nullptr) continue;
    const Image& image = reconstruction.Image(image_id);

    PriorPositionResidual record;
    record.image_id = image_id;
    record.name = image.Name();
    record.prior_position = prior->position;
    record.solved_position = image.ProjectionCenter();
    record.residual = record.solved_position - record.prior_position;
    record.residual_norm = record.residual.norm();

    const Eigen::Matrix3d cov =
        prior->HasPositionCov() ? prior->position_covariance : fallback_cov;
    record.sigma = cov.diagonal().cwiseMax(0.0).cwiseSqrt();
    const double squared_mahalanobis =
        SquaredMahalanobisDistance(cov, record.residual, fallback_stddev);
    record.mahalanobis = std::sqrt(std::max(0.0, squared_mahalanobis));
    record.robust_weight = PriorPositionRobustWeight(
        squared_mahalanobis, loss_scale, use_robust_loss);
    record.used_in_ba = used_in_ba.count(image_id) > 0;

    residuals.push_back(std::move(record));
  }

  std::sort(residuals.begin(),
            residuals.end(),
            [](const PriorPositionResidual& lhs,
               const PriorPositionResidual& rhs) {
              return lhs.image_id < rhs.image_id;
            });
  return residuals;
}

void WritePriorPositionResiduals(
    const std::filesystem::path& path,
    const std::vector<PriorPositionResidual>& residuals) {
  std::ofstream file(path);
  if (!file.is_open()) {
    LOG(WARNING) << "WritePriorPositionResiduals: could not open " << path;
    return;
  }
  file << "image_id\tname\tprior_x\tprior_y\tprior_z\tsolved_x\tsolved_y\t"
          "solved_z\tresidual_x\tresidual_y\tresidual_z\tresidual_norm\t"
          "sigma_x\tsigma_y\tsigma_z\tmahalanobis\trobust_weight\tused_in_ba"
       << '\n';
  file << std::setprecision(9);
  for (const PriorPositionResidual& r : residuals) {
    file << r.image_id << '\t' << r.name << '\t' << r.prior_position.x() << '\t'
         << r.prior_position.y() << '\t' << r.prior_position.z() << '\t'
         << r.solved_position.x() << '\t' << r.solved_position.y() << '\t'
         << r.solved_position.z() << '\t' << r.residual.x() << '\t'
         << r.residual.y() << '\t' << r.residual.z() << '\t' << r.residual_norm
         << '\t' << r.sigma.x() << '\t' << r.sigma.y() << '\t' << r.sigma.z()
         << '\t' << r.mahalanobis << '\t' << r.robust_weight << '\t'
         << (r.used_in_ba ? 1 : 0) << '\n';
  }
}

PriorPositionResidualSummary SummarizePriorPositionResiduals(
    const std::vector<PriorPositionResidual>& residuals) {
  PriorPositionResidualSummary summary;
  if (residuals.empty()) return summary;

  double sum_squared = 0.0;
  for (const PriorPositionResidual& r : residuals) {
    sum_squared += r.residual_norm * r.residual_norm;
    if (r.mahalanobis > summary.worst_mahalanobis) {
      summary.worst_mahalanobis = r.mahalanobis;
      summary.worst_image_id = r.image_id;
    }
    summary.worst_norm = std::max(summary.worst_norm, r.residual_norm);
  }
  summary.num_residuals = residuals.size();
  summary.rms = std::sqrt(sum_squared / residuals.size());
  return summary;
}

}  // namespace colmap
