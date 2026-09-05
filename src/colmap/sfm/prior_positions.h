// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// (Same BSD licence header as other COLMAP files.)

#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class Reconstruction;
class DatabaseCache;

// ── Position priors in a reconstruction's own world frame ──────────────────
//
// The database stores position priors in a metric frame (ENU metres for the
// site pipeline; see `db_priors.pack_doubles_le`). A reconstruction produced
// by `pose_prior_mapper` is already expressed in that frame, but one produced
// by a priorless solve -- or an incremental add seeded from it -- is not: it
// carries an arbitrary gauge. Comparing a solved camera centre with its prior
// is only meaningful once both sit in the same frame.
//
// So the priors are brought to the reconstruction rather than the other way
// round: a Sim3 is fitted robustly from the registered images' solved centres
// to their priors, inverted, and applied to every prior (covariances
// included). When the model is already metric the fit is the identity and
// nothing changes; when it is not, the residuals are still the quantity we
// care about -- the per-image disagreement -- rather than the gauge.
//
// Transforming the priors, not the model, is deliberate: an incremental add
// must not move the reconstruction out of the prior model's gauge, which is
// exactly what PosePriorBundleAdjuster's own alignment step would do.
// How the prior-position term behaves, carried alongside the priors so that
// solve stages which never see GlobalMapperOptions still apply exactly the
// same constraint.
struct PriorPositionSettings {
  bool use_robust_loss = false;
  double loss_scale = 7.815;
  double fallback_stddev = 1.0;
  double max_error_sigma = 5.0;
  double max_error_m = 0.1;
};

struct WorldPositionPriors {
  // image_id -> prior with `position` and `position_covariance` expressed in
  // the reconstruction's world frame.
  std::unordered_map<image_t, PosePrior> priors;

  // See PriorPositionSettings.
  PriorPositionSettings settings;

  // The fitted transform taking a prior-frame point to the world frame.
  Sim3d world_from_prior;

  // Number of registered images the fit was computed from.
  size_t num_alignment_images = 0;

  // RMS distance, in the priors' own (metric) frame, between the fitted
  // camera centres and the priors. Negative when no fit was performed.
  double alignment_rmse = -1.0;

  bool Empty() const { return priors.empty(); }

  // Returns nullptr when the image has no usable prior.
  const PosePrior* Find(image_t image_id) const;
};

struct WorldPositionPriorOptions {
  // Standard deviation, in metres, assumed for priors that carry no
  // covariance of their own. The site pipeline writes real per-frame
  // covariance, so this should never be reached there.
  double fallback_stddev = 1.0;

  // RANSAC options for the world<-prior fit. A max_error <= 0 derives the
  // threshold from the priors' own covariances (95% confidence radius),
  // matching PosePriorBundleAdjuster::AlignReconstruction.
  RANSACOptions alignment_ransac_options;

  // Minimum number of registered images with priors required to fit.
  int min_alignment_images = 3;
};

// Collects the database's position priors for the images of `reconstruction`
// and expresses them in that reconstruction's world frame.
//
// Only CARTESIAN priors are used. WGS84 priors are converted to Cartesian ENU
// by DatabaseCache when `convert_pose_priors_to_enu` is set, so anything that
// is still WGS84 (or UNDEFINED) at this point has an unknown metric meaning
// and is dropped with a warning rather than silently mixed in.
//
// Returns an empty result when the frame fit is impossible or fails; that is
// the safe direction, since an identity fallback on a non-metric model would
// make every residual meaningless and reject every image.
WorldPositionPriors CollectWorldPositionPriors(
    const DatabaseCache& database_cache,
    const Reconstruction& reconstruction,
    const WorldPositionPriorOptions& options);

// Squared Mahalanobis distance of `residual` under `covariance`.
// Falls back to an isotropic `fallback_stddev` when the covariance is not
// usable (non-finite, or not positive definite).
double SquaredMahalanobisDistance(const Eigen::Matrix3d& covariance,
                                  const Eigen::Vector3d& residual,
                                  double fallback_stddev);

// Weight the robust loss applies to a residual with the given squared
// Mahalanobis distance. Returns 1 for the trivial (non-robust) loss, and the
// Cauchy weight 1 / (1 + s / c^2) otherwise, matching the loss
// PosePriorBundleAdjuster installs.
double PriorPositionRobustWeight(double squared_mahalanobis,
                                 double loss_scale,
                                 bool use_robust_loss);

// ── Per-image prior residual report (plan-6 item 2) ────────────────────────
//
// Everything bundle adjustment already knows about the gap between a camera
// and its position prior, written out so that a displaced frame is
// identifiable from one file instead of six rounds of re-deriving the
// geometry from `residuals.csv` and a Sim3.
struct PriorPositionResidual {
  image_t image_id = kInvalidImageId;
  std::string name;
  // All positions in the reconstruction's world frame.
  Eigen::Vector3d prior_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d solved_position = Eigen::Vector3d::Zero();
  // solved - prior.
  Eigen::Vector3d residual = Eigen::Vector3d::Zero();
  double residual_norm = 0.0;
  // Per-axis standard deviation actually used (sqrt of the covariance
  // diagonal in world frame), i.e. what the cost function was weighted by.
  Eigen::Vector3d sigma = Eigen::Vector3d::Zero();
  // Mahalanobis distance of the residual, in sigmas.
  double mahalanobis = 0.0;
  // Weight the robust loss gave this residual (1 when not down-weighted).
  double robust_weight = 1.0;
  // Whether this image's pose was a free parameter constrained by its prior
  // in this solve's bundle adjustment.
  bool used_in_ba = false;
};

// Computes one record per registered image that has a usable prior.
// `used_in_ba` marks the images whose poses the solve actually left free.
std::vector<PriorPositionResidual> ComputePriorPositionResiduals(
    const Reconstruction& reconstruction,
    const WorldPositionPriors& world_priors,
    const std::unordered_set<image_t>& used_in_ba,
    double fallback_stddev,
    bool use_robust_loss,
    double loss_scale);

// Writes the records as a TSV with a header line, sorted by image id.
void WritePriorPositionResiduals(
    const std::filesystem::path& path,
    const std::vector<PriorPositionResidual>& residuals);

// Summary statistics over a residual report, for the incremental ledger.
struct PriorPositionResidualSummary {
  size_t num_residuals = 0;
  double rms = 0.0;
  double worst_norm = 0.0;
  double worst_mahalanobis = 0.0;
  image_t worst_image_id = kInvalidImageId;
};

PriorPositionResidualSummary SummarizePriorPositionResiduals(
    const std::vector<PriorPositionResidual>& residuals);

}  // namespace colmap
