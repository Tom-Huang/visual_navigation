/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  Eigen::Vector3d t_normalized = t_0_1.normalized();
  Eigen::Matrix3d skew;
  skew << 0, -t_normalized(2), t_normalized(1), t_normalized(2), 0,
      -t_normalized(0), -t_normalized(1), t_normalized(0), 0;
  E = -skew * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    if (abs(cam1->unproject(p0_2d).transpose() * E * cam2->unproject(p1_2d)) <
        epipolar_error_threshold) {
      md.inliers.push_back(
          std::make_pair(md.matches[j].first, md.matches[j].second));
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  // TODO SHEET 3: run RANSAC with using opengv's CentralRelativePose and store
  // in md.inliers. If the number if inliers is smaller than ransac_min_inliers,
  // leave md.inliers empty.
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<opengv::bearingVector_t>>
      KeyPoint3d_1;
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<opengv::bearingVector_t>>
      KeyPoint3d_2;
  for (int j = 0; j < md.matches.size(); ++j) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d).normalized();
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d).normalized();
    KeyPoint3d_1.push_back(p0_3d);
    KeyPoint3d_2.push_back(p1_3d);
  }
  const std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<opengv::bearingVector_t>>
      kp3d1 = KeyPoint3d_1;
  const std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<opengv::bearingVector_t>>
      kp3d2 = KeyPoint3d_2;

  opengv::relative_pose::CentralRelativeAdapter adapter(kp3d1, kp3d2);
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 200;
  ransac.computeModel();
  opengv::transformation_t best_transformation = ransac.model_coefficients_;
  Eigen::Matrix4d b_tf_4_4;
  b_tf_4_4.setZero();
  b_tf_4_4.block<3, 4>(0, 0) = best_transformation;
  b_tf_4_4(3, 3) = 1;
  std::cout << b_tf_4_4 << std::endl;
  const Sophus::SE3d T_0_1(b_tf_4_4);
  md.T_i_j = T_0_1;
  std::cout << "Matches size: " << md.matches.size()
            << ", inliers size: " << ransac.inliers_.size() << std::endl;
  if (ransac.inliers_.size() < ransac_min_inliers) {
    return;
  } else {
    for (int i = 0; i < ransac.inliers_.size(); i++) {
      md.inliers.push_back(md.matches[ransac.inliers_[i]]);
    }
  }
}
}  // namespace visnav
