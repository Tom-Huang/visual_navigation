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

  // create bearingVectors to store all the matched 3D points
  opengv::bearingVectors_t kp3d1;
  opengv::bearingVectors_t kp3d2;
  for (int j = 0; j < md.matches.size(); ++j) {
    Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d).normalized();
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d).normalized();
    kp3d1.push_back(p0_3d);
    kp3d2.push_back(p1_3d);
  }

  // create adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(kp3d1, kp3d2);

  // create RANSAC
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;

  // create relative pose problem which uses NISTER five points algorithm
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));

  // initialize some parameters in ransac object
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;

  // compute inliers and transformation with ransac
  ransac.computeModel();

  // store all inliers acquired from ransac
  opengv::bearingVectors_t inliers_1;
  opengv::bearingVectors_t inliers_2;
  for (int j = 0; j < ransac.inliers_.size(); ++j) {
    int inliers_ind = ransac.inliers_[j];
    inliers_1.push_back(kp3d1[inliers_ind]);
    inliers_2.push_back(kp3d2[inliers_ind]);
  }

  // create a new adapter to store inliers from ransac
  opengv::relative_pose::CentralRelativeAdapter new_adapter(inliers_1,
                                                            inliers_2);

  // initialize rotation matrix and translation vector acquired from ransac
  new_adapter.setR12(ransac.model_coefficients_.block<3, 3>(0, 0));
  new_adapter.sett12(ransac.model_coefficients_.col(3));

  // use nonlinear optimization to refine transformation matrix
  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(new_adapter);

  // calculate new inliers with new transformation
  std::vector<int> new_inliers;
  relposeproblem_ptr->selectWithinDistance(nonlinear_transformation, 1e-3,
                                           new_inliers);

  // create T_0_1 to hold the transformation matrix
  Sophus::SE3d T_0_1(
      nonlinear_transformation.block(0, 0, 3, 3),   // rotation matrix
      nonlinear_transformation.block(0, 3, 3, 1));  // translation vector

  // store optimized transformation in kd
  md.T_i_j = T_0_1;

  for (int i = 0; i < new_inliers.size(); i++) {
    md.inliers.push_back(std::make_pair(md.matches[new_inliers[i]].first,
                                        md.matches[new_inliers[i]].second));
  }
  std::cout << "Matches size: " << md.matches.size()
            << ", ransac inliers size: " << ransac.inliers_.size()
            << ", final inliers size: " << md.inliers.size() << std::endl;
}
}  // namespace visnav
