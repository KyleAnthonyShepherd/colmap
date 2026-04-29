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

#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace colmap {

// Returns the index of the quaternion in `qs` that has the highest weight.
inline int BestCandidateIdxByWeight(const std::vector<double>& weights) {
  DCHECK(!weights.empty());
  return static_cast<int>(
      std::distance(weights.begin(),
                    std::max_element(weights.begin(), weights.end())));
}

// Computes the weighted Fréchet / Karcher mean of unit quaternions on SO(3).
//
// Uses iterative Riemannian gradient descent:
//   q_{k+1} = q_k * Exp( sum_i (w_i/W) * Log(q_k^{-1} * q_i) )
// where W = sum of all weights.
//
// The algorithm converges quadratically near the mean and is exact in the
// limit (within `tol` of the tangent-space step norm).
//
// Args:
//   qs        - input unit quaternions
//   weights   - non-negative weights; normalised internally
//   q_init    - starting estimate (use BestCandidateIdxByWeight for safety)
//   max_iters - hard iteration limit
//   tol       - convergence threshold on the tangent-space update ‖·‖
//
// Returns the weighted mean rotation on SO(3).
inline Eigen::Quaterniond KarcherMeanSO3(
    const std::vector<Eigen::Quaterniond>& qs,
    const std::vector<double>& weights,
    Eigen::Quaterniond q_init,
    int max_iters = 30,
    double tol = 1e-8) {
  DCHECK_EQ(qs.size(), weights.size());
  DCHECK(!qs.empty());

  // Normalise weights.
  double w_sum = 0.0;
  for (const double w : weights) {
    DCHECK_GE(w, 0.0);
    w_sum += w;
  }
  if (w_sum < 1e-12) return q_init;

  Eigen::Quaterniond q_mean = q_init.normalized();

  for (int iter = 0; iter < max_iters; ++iter) {
    Eigen::Vector3d tangent_sum = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < qs.size(); ++i) {
      // Relative rotation expressed in the tangent space at q_mean.
      Eigen::Quaterniond dq = (q_mean.inverse() * qs[i]).normalized();

      // Ensure shortest-arc path on S^3.
      if (dq.w() < 0.0) dq.coeffs() = -dq.coeffs();

      // SO(3) Log map: tangent vector = axis * angle.
      const double angle = 2.0 * std::acos(std::clamp(dq.w(), -1.0, 1.0));
      Eigen::Vector3d axis = dq.vec();
      const double axis_norm = axis.norm();
      if (axis_norm > 1e-12) axis /= axis_norm;

      tangent_sum += (weights[i] / w_sum) * angle * axis;
    }

    const double update_norm = tangent_sum.norm();
    if (update_norm < tol) break;  // Converged.

    // SO(3) Exp map: update the mean.
    q_mean =
        (q_mean * Eigen::Quaterniond(
                      Eigen::AngleAxisd(update_norm, tangent_sum / update_norm)))
            .normalized();
  }

  return q_mean;
}

}  // namespace colmap
