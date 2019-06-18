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

// TODO PROJECT: include pangolin and opencv
#include <pangolin/image/managed_image.h>
#include <opencv2/opencv.hpp>

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

// TODO PROJECT: use optical flow to calculate feature points in next left frame
// based on feature points in the last left frame
void OpticalFlowBetweenFrame_opencv_version(
    FrameId current_frame, const pangolin::ManagedImage<uint8_t>& imglt0,
    const pangolin::ManagedImage<uint8_t>& imglt1, const KeypointsData& kdlt0,
    KeypointsData& kdlt1, const Landmarks& landmarks,
    MatchData& md_feat2track) {
  cv::Mat imglt0_cv(imglt0.h, imglt0.w, CV_8U, imglt0.ptr);
  cv::Mat imglt1_cv(imglt1.h, imglt1.w, CV_8U, imglt1.ptr);

  std::vector<Eigen::Vector2f> points0;
  std::vector<Eigen::Vector2f> points1;

  std::vector<unsigned char> status;
  std::vector<float> errors;
  cv::Size winSize(31, 31);

  // convert double vector to float vector in eigen
  for (const auto p_2dl : kdlt0.corners) {
    Eigen::Vector2f p_2d = p_2dl;
    points0.push_back(p_2d);
  }
  cv::calcOpticalFlowPyrLK(imglt0_cv, imglt1_cv, points0, points1, status,
                           errors, winSize, 4);
  kdlt1.corners.clear();
  int j = 0;
  for (int i = 0; i < points1.size(); i++) {
    if (status[i]) {
      Eigen::Vector2d p_2d = points1[i];
      kdlt1.corners.push_back(p_2d);
      TrackId trackid = kdlt0.trackids[i];
      kdlt1.trackids.push_back(trackid);
      md_feat2track.matches.push_back(std::make_pair(j, trackid));
      j++;
    }
  }
}

// TODO PROJECT: make grid in the image and store the top left corner and bottom
// right corner of each cell in a Cell object. The rnum and cnum should be
// devidable by h and w respectively.
void makeCells(int h, int w, int rnum, int cnum, std::vector<Cell>& cells) {
  int cellh = h / rnum;
  int cellw = w / cnum;
  for (int i = 0; i < rnum; i++) {
    for (int j = 0; j < cnum; j++) {
      int rmin = i * cellh;
      int rmax = (i + 1) * cellh;
      int cmin = j * cellw;
      int cmax = (j + 1) * cellw;
      Cell cell;
      cell.topleft = std::make_pair(rmin, cmin);
      cell.bottomright = std::make_pair(rmax, cmax);
      cell.kpnum = 0;
      cells.push_back(cell);
    }
  }
}

// TODO PROJECT: add keypoints in left image to existing landmarks' observations
void add_points_to_landmark_obs_left(const MatchData& md_feat2track,
                                     const KeypointsData& kdl,
                                     Landmarks& landmarks,
                                     FrameId current_frame) {
  TimeCamId tcid = std::make_pair(current_frame, 0);
  for (const auto feat_track_pair : md_feat2track.matches) {
    FeatureId featid = feat_track_pair.first;
    TrackId trackid = feat_track_pair.second;
    landmarks.at(trackid).obs.emplace(tcid, featid);
  }
}

// TODO PROJECT: add keypoints in right image to existing landmarks's
// observations
void add_points_to_landmarks_obs_right(const MatchData& md_stereo,
                                       const MatchData& md_feat2track,
                                       Landmarks& landmarks,
                                       FrameId current_frame) {
  TimeCamId tcid = std::make_pair(current_frame, 1);
  for (const auto left_right_feat_pair : md_stereo.matches) {
    FeatureId featidl = left_right_feat_pair.first;
    FeatureId featidr = left_right_feat_pair.second;
    for (const auto featl_track_pair : md_feat2track.matches) {
      if (featidl == featl_track_pair.first) {
        TrackId trackid = featl_track_pair.second;
        landmarks.at(trackid).obs.emplace(tcid, featidr);
        break;
      }
    }
  }
}

// TODO PROJECT: check the number of keypoints in each grid and fulfill
// the variable kpnum of each cell.
// TODO: double check whether there is an overlapping of the grid!!!!!!!
void check_num_points_in_cells(const KeypointsData& kdlt1, std::vector<Cell>& cells){
  for (auto & kp : kdlt1.corners){
    for (auto & cell : cells){
      if (kp[0]>=cell.topleft.first && kp[1]>=cell.topleft.second&&kp[0]<cell.bottomright.first&&kp[1]<cell.bottomright.second){
        cell.kpnum++;
      }
    }
  }
}

// TODO PROJECT: calculate the number of empty cells. empty cells indexes saved
// in empty_indexes, and returns the number of empty cells.  
int sparsity(std::vector<Cell>& cells, std::vector<int> & empty_indexes){
  int num_of_empty_cells = 0;
  for (int i =0; i<cells.size();i++){
    if (cells[i].kpnum==0){
      num_of_empty_cells = num_of_empty_cells+1;
      empty_indexes.push_back(i);
    }
  }
  return num_of_empty_cells;
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
}  // namespace visnav
