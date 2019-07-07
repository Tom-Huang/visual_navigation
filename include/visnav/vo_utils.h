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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

// TODO PROJECT: include sophus sim3
#include <sophus/sim3.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  for (const auto landmark : landmarks) {
    Landmark lm = landmark.second;
    Eigen::Vector3d p_3d_c = current_pose.inverse() * lm.p;
    if (p_3d_c[2] < cam_z_threshold) continue;

    Eigen::Vector2d p_2d_c = cam->project(p_3d_c);
    if (p_2d_c[0] < 0 || p_2d_c[0] >= 752 || p_2d_c[1] < 0 || p_2d_c[1] >= 480)
      continue;

    projected_track_ids.push_back(landmark.first);
    projected_points.push_back(p_2d_c);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_max_dist and feature_match_test_next_best
  // should be used to filter outliers the same way as in exercise 3.

  for (int i = 0; i < kdl.corners.size(); i++) {
    std::bitset<256> kp_des = kdl.corner_descriptors[i];
    Eigen::Vector2d kp_2d = kdl.corners[i];
    int best_dist_for_kpl = 500;
    int second_best_dist_for_kpl = 500;
    FeatureId best_trackid_for_kpl = -1;
    for (int j = 0; j < projected_points.size(); j++) {
      Eigen::Vector2d pp_2d = projected_points[j];
      TrackId ptrack_id = projected_track_ids[j];
      int best_dist_in_obs = 500;
      if ((pp_2d - kp_2d).norm() < match_max_dist_2d) {
        for (const auto tcid_featid_pair : landmarks.at(ptrack_id).obs) {
          TimeCamId tcid = tcid_featid_pair.first;
          FeatureId featid = tcid_featid_pair.second;
          std::bitset<256> obs_des =
              feature_corners.at(tcid).corner_descriptors[featid];

          int dist = (obs_des ^ kp_des).count();
          if (dist <= best_dist_in_obs) {
            best_dist_in_obs = dist;
          }
        }
      }
      if (best_dist_in_obs < best_dist_for_kpl) {
        second_best_dist_for_kpl = best_dist_for_kpl;
        best_dist_for_kpl = best_dist_in_obs;
        best_trackid_for_kpl = projected_track_ids[j];
      } else if (best_dist_in_obs < second_best_dist_for_kpl) {
        second_best_dist_for_kpl = best_dist_in_obs;
      }
    }
    if (best_dist_for_kpl < feature_match_max_dist &&
        best_dist_for_kpl * feature_match_test_next_best <
            second_best_dist_for_kpl)
      md.matches.push_back(std::make_pair(i, best_trackid_for_kpl));
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  // TODO SHEET 5: Find the pose (T_w_c) and the inliers using the landmark to
  // keypoints matches and PnP. This should be similar to the localize_camera in
  // exercise 4 but in this execise we don't explicitelly have tracks.
  opengv::bearingVectors_t bearingvec1;
  opengv::points_t points_w;
  for (const auto kpid_trackid_pair : md.matches) {
    bearingvec1.push_back(
        cam->unproject(kdl.corners[kpid_trackid_pair.first]).normalized());
    points_w.push_back(landmarks.at(kpid_trackid_pair.second).p);
  }

  // create central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingvec1, points_w);

  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;

  // create an AbsolutePoseSacProblem
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();

  // construct a new adapter with inliers computed from ransac
  opengv::bearingVectors_t bearingvec1_inliers;
  opengv::points_t points_w_inliers;
  for (const auto in_ind : ransac.inliers_) {
    bearingvec1_inliers.push_back(bearingvec1[in_ind]);
    points_w_inliers.push_back(points_w[in_ind]);
  }
  opengv::absolute_pose::CentralAbsoluteAdapter adapter_inliers(
      bearingvec1_inliers, points_w_inliers);
  adapter_inliers.sett(ransac.model_coefficients_.block(0, 3, 3, 1));
  adapter_inliers.setR(ransac.model_coefficients_.block(0, 0, 3, 3));

  // nonliner optimization
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter_inliers);

  // recompute the inliers
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, inliers);

  T_w_c = Sophus::SE3d(nonlinear_transformation.block(0, 0, 3, 3),
                       nonlinear_transformation.block(0, 3, 3, 1));
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains landmark to map
  // matches for the left camera (camera 0). The inliers vector contains all
  // inliers in md that were used to compute the pose T_w_c0. For all inliers
  // add the observations to the existing ladmarks (if the left point is in
  // md_stereo.inliers then add both observations). For all stereo observations
  // that were not added to the existing landmarks triangulate and add new
  // landmarks. Here next_landmark_id is a running index of the landmarks, so
  // after adding a new landmark you should always increase next_landmark_id
  // by 1.

  // create bearing vectors for triangulation
  opengv::bearingVectors_t bv1;
  opengv::bearingVectors_t bv2;
  for (int i = 0; i < md_stereo.inliers.size(); i++) {
    FeatureId featidl = md_stereo.inliers[i].first;
    FeatureId featidr = md_stereo.inliers[i].second;
    bv1.push_back(
        calib_cam.intrinsics[0]->unproject(kdl.corners[featidl]).normalized());
    bv2.push_back(
        calib_cam.intrinsics[1]->unproject(kdr.corners[featidr]).normalized());
  }

  for (const auto inliers_ind : inliers) {
    FeatureId featidl = md.matches[inliers_ind].first;
    TrackId trackid = md.matches[inliers_ind].second;

    // add observations for matches in inliers
    landmarks.at(trackid).obs.emplace(tcidl, featidl);

    for (int stereo_i = 0; stereo_i < md_stereo.inliers.size(); stereo_i++) {
      FeatureId featidl_stereo = md_stereo.inliers[stereo_i].first;
      FeatureId featidr = md_stereo.inliers[stereo_i].second;

      // add obervations for stereo matches
      if (featidl_stereo == featidl) {
        landmarks.at(trackid).obs.emplace(tcidr, featidr);
      }
    }
  }

  for (int i = 0; i < md_stereo.inliers.size(); i++) {
    FeatureId featidl = md_stereo.inliers[i].first;
    FeatureId featidr = md_stereo.inliers[i].second;
    int find_flag = 0;
    for (int j = 0; j < inliers.size(); j++) {
      if (md.matches[inliers[j]].first == featidl) {
        find_flag = 1;
        break;
      }
    }
    if (find_flag == 0) {
      // create new landmark by triangulation
      opengv::relative_pose::CentralRelativeAdapter adapter(bv1, bv2, t_0_1,
                                                            R_0_1);
      opengv::point_t p_3d_c = opengv::triangulation::triangulate(adapter, i);
      landmarks[next_landmark_id].p = T_w_c0 * p_3d_c;
      landmarks[next_landmark_id].obs.emplace(tcidl, featidl);
      landmarks[next_landmark_id].obs.emplace(tcidr, featidr);
      next_landmark_id++;
    }
  }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  if (kf_frames.size() > max_num_kfs) {
    int kf_size = kf_frames.size();

    for (int i = 0; i < int(kf_size - max_num_kfs); i++) {
      FrameId removed_frame = *kf_frames.begin();

      TimeCamId frame_cam_l = std::make_pair(removed_frame, 0);
      TimeCamId frame_cam_r = std::make_pair(removed_frame, 1);
      kf_frames.erase(kf_frames.begin());
      cameras.erase(frame_cam_l);
      cameras.erase(frame_cam_r);
      std::vector<TrackId> move_to_old_id;
      move_to_old_id.clear();

      for (const auto trackid_landmark_pair : landmarks) {
        FeatureTrack obs = trackid_landmark_pair.second.obs;
        TrackId trackid = trackid_landmark_pair.first;
        if (obs.find(frame_cam_l) != obs.end()) {
          landmarks.at(trackid).obs.erase(frame_cam_l);
        }
        if (obs.find(frame_cam_r) != obs.end()) {
          landmarks.at(trackid).obs.erase(frame_cam_r);
        }
        int no_left_obs_flag = 1;
        for (const auto timecamid_featid_pair : obs) {
          if (timecamid_featid_pair.first.second == 0) {
            no_left_obs_flag = 0;
            break;
          }
        }
        if (no_left_obs_flag == 1) {
          move_to_old_id.push_back(trackid);
        }
      }
      // move track without left obs to old
      for (const auto no_left_obs_trackid : move_to_old_id) {
        old_landmarks[no_left_obs_trackid] = landmarks.at(no_left_obs_trackid);
        landmarks.erase(no_left_obs_trackid);
      }
    }
  }
}

// TODO PROJECT: function for alignment of camera trajectory and ground truth
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using ArrX = Eigen::Array<double, 1, Eigen::Dynamic>;
using Poses = std::vector<Sophus::SE3d>;

struct ErrorMetricValue {
  double rmse = 0;
  double mean = 0;
  double min = 0;
  double max = 0;
  double count = 0;  //!< number of elements involved in the evaluation
};

/// Compute Sim(3) transformation that aligns the 3D points in model to 3D
/// points in data in the least squares sense using Horn's algorithm. I.e. the
/// Sim(3) transformation T is computed that minimizes the sum over all i of
/// ||T*m_i - d_i||^2, m_i are the column's of model and d_i are the column's of
/// data. Both data and model need to be of the same dimension and have at least
/// 3 columns.
///
/// Optionally computes the translational rmse.
///
/// Note that for the orientation we don't actually use Horn's algorithm, but
/// the one published by Arun
/// (http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf) based on SVD with
/// later extension to cover degenerate cases.
///
/// See the illustrative comparison by Eggert:
/// http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
Sophus::Sim3d align_points_sim3(const Mat3X& data, const Mat3X& model,
                                Mat3X& model_transformed,
                                ErrorMetricValue& ate) {
  // 0. Centroids
  const Vec3 centroid_data = data.rowwise().mean();
  const Vec3 centroid_model = model.rowwise().mean();

  // center both clouds to 0 centroid
  const Mat3X data_centered = data.colwise() - centroid_data;
  const Mat3X model_centered = model.colwise() - centroid_model;
  std::cout << "centralization succeeds." << std::endl;

  // 1. Rotation

  // sum of outer products of columns
  const Mat3 W = data_centered * model_centered.transpose();

  const auto svd = W.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  // last entry to ensure we don't get a reflection, only rotations
  const Mat3 S = Eigen::DiagonalMatrix<double, 3, 3>(
      1, 1,
      svd.matrixU().determinant() * svd.matrixV().determinant() < 0 ? -1 : 1);

  const Mat3 R = svd.matrixU() * S * svd.matrixV().transpose();

  const Mat3X model_rotated = R * model_centered;

  std::cout << "rotation succeeds." << std::endl;

  //  // 2. Scale (regular, non-symmetric variant)

  //  // sum of column-wise dot products
  //  const double dots = (data_centered.cwiseProduct(model_rotated)).sum();

  //  // sum of column-wise norms
  //  const double norms = model_centered.colwise().squaredNorm().sum();

  //  // scale
  //  const double s = dots / norms;

  // 3. Translation
  const Vec3 t = centroid_data - R * centroid_model;

  std::cout << "translation succeeds." << std::endl;

  model_transformed = (R * model).colwise() + t;

  std::cout << "model transformation succeeds." << std::endl;
  // 4. Translational error
  if (1) {  // ate.count == 0) {
    // static_assert(ArrX::ColsAtCompileTime == 1);

    //    const Mat3X diff = data - ((s * R * model).colwise() + t);
    const Mat3X diff = data - ((R * model).colwise() + t);
    const ArrX errors = diff.colwise().norm().transpose();

    //  auto& ref = *ate;
    ate.rmse = std::sqrt(errors.square().sum() / errors.rows());
    ate.mean = errors.mean();
    ate.min = errors.minCoeff();
    ate.max = errors.maxCoeff();
    ate.count = errors.rows();
  }
  std::cout << "ate calculation succeeds." << std::endl;
  return Sophus::Sim3d(Sophus::RxSO3d(1, R), t);
}

}  // namespace visnav
