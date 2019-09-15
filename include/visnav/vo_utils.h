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
#include <visnav/keypoints.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

// TODO PROJECT: include pangolin
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

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

// TODO PROJECT: Manual implementation of bilinear interpolation of image
/*
 * Input:
 *  img    : image to interpolate on
 *  x      : column position(float)
 *  y      : row position(float)
 * Output:
 *  result : interpolation result
 * */
float bilinear_interpolate(cv::Mat img, float x, float y) {
  if (x < 0 || x >= img.size().width || y < 0 || y >= img.size().height) {
    // std::cout << "x, y out of range!" << std::endl;
    return 0;
  }
  float result;
  float topleft, topright, bottomleft, bottomright;
  int x_floor = int(x);
  int y_floor = int(y);
  topleft = img.at<float>(y_floor, x_floor);
  topright = img.at<float>(y_floor, x_floor + 1);
  bottomleft = img.at<float>(y_floor + 1, x_floor);
  bottomright = img.at<float>(y_floor + 1, x_floor + 1);
  result = (y - y_floor) *
               ((x_floor + 1 - x) * bottomleft + (x - x_floor) * bottomright) +
           (y_floor + 1 - y) *
               ((x_floor + 1 - x) * topleft + (x - x_floor) * topright);
  return result;
}

// TODO PROJECT: Find gradient of image in x direction
/*
 * ---->x
 * /
 * /
 * y
 * Input:
 *  img        : input image
 *  gradient_x : output Mat storing gradient
 * */
void cv_gradient_x(const cv::Mat& img, cv::Mat& gradient_x) {
  float k1[3] = {-0.5, 0, 0.5};
  cv::Mat ker = cv::Mat(1, 3, CV_32FC1, k1);
  cv::filter2D(img, gradient_x, img.depth(), ker);
}

// TODO PROJECT: Find gradient of image in y direction
/*
 * ---->x
 * /
 * /
 * y
 * Input:
 *  img        : input image
 *  gradient_y : output Mat storing gradient
 * */
void cv_gradient_y(const cv::Mat& img, cv::Mat& gradient_y) {
  float k1[3][1] = {-0.5, 0, 0.5};
  cv::Mat ker = cv::Mat(3, 1, CV_32FC1, k1);
  cv::filter2D(img, gradient_y, img.depth(), ker);
}

// TODO PROJECT: Manual implementation of intensity invariant optical flow
// method
/*
 * Input:
 *  img0_uchar        : source image with format CV_8U.
 *  img1_uchar        : target image with format CV_8U.
 *  points0           : vector of source feature points.
 *  points1           : vector of output target feature points.
 *  status            : vector of status storing 1 or 0. 1 represents output
 *                      feature point is inside the image boundary.
 *  errors            : vector of errors of every output feature point.
 *  maxLevel          : maximum pyramid level in optical flow.
 * */
void OpticalFLowLK(const cv::Mat& img0_uchar, const cv::Mat& img1_uchar,
                   std::vector<cv::Point2f>& points0,
                   std::vector<cv::Point2f>& points1,
                   std::vector<unsigned char>& status,
                   std::vector<float>& errors, int maxLevel = 5) {
  const int pattern[][2] = {
      {-3, 7},  {-1, 7},  {1, 7},   {3, 7},

      {-5, 5},  {-3, 5},  {-1, 5},  {1, 5},   {3, 5},  {5, 5},

      {-7, 3},  {-5, 3},  {-3, 3},  {-1, 3},  {1, 3},  {3, 3},
      {5, 3},   {7, 3},

      {-7, 1},  {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},
      {5, 1},   {7, 1},

      {-7, -1}, {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1},
      {5, -1},  {7, -1},

      {-7, -3}, {-5, -3}, {-3, -3}, {-1, -3}, {1, -3}, {3, -3},
      {5, -3},  {7, -3},

      {-5, -5}, {-3, -5}, {-1, -5}, {1, -5},  {3, -5}, {5, -5},

      {-3, -7}, {-1, -7}, {1, -7},  {3, -7}

  };

  cv::Mat img00, img10;
  img0_uchar.convertTo(img00, CV_32FC1, 1.0 / 255);
  img1_uchar.convertTo(img10, CV_32FC1, 1.0 / 255);

  cv::Mat gradient_x, gradient_y;

  // create different layers
  std::vector<cv::Mat> layers0;
  std::vector<cv::Mat> layers1;
  layers0.push_back(img00);
  layers1.push_back(img10);

  for (int l = 0; l < maxLevel - 1; l++) {
    cv::Mat tmp0, tmp1;
    cv::pyrDown(layers0[l], tmp0);
    cv::pyrDown(layers1[l], tmp1);
    layers0.push_back(tmp0);
    layers1.push_back(tmp1);
  }

  // iterate over points
  for (const auto point : points0) {
    const float BOUNDARY = 2;
    Eigen::Vector2f motion = Eigen::Vector2f::Zero();
    Eigen::Vector2f motion_old;
    Eigen::Vector2f dmotion = Eigen::Vector2f::Zero();
    float error;
    cv::Point2f new_point;
    int out_flag = 0;
    int patch_out_flag = 0;

    for (int l = maxLevel; l > 0; l--) {
      float scale = pow(2, l - 1);

      const int WINDOW_SIZE = sizeof(pattern) / (sizeof(int) * 2);

      Eigen::Matrix<float, 2, 2> hessian = Eigen::Matrix2f::Zero();

      Eigen::MatrixXf gradient0(WINDOW_SIZE, 2);
      Eigen::MatrixXf position0(WINDOW_SIZE, 2);
      Eigen::MatrixXf intensity0(WINDOW_SIZE, 1);
      Eigen::MatrixXf position1(WINDOW_SIZE, 2);
      Eigen::MatrixXf intensity1(WINDOW_SIZE, 1);
      Eigen::Vector2f grad_sum0;
      float sum0 = 0;
      float sum1 = 0;
      float mean0 = 0;
      float mean1 = 0;
      int valid_num0 = 0;
      int valid_num1 = 0;
      int i = 0;
      intensity0.setZero();
      intensity1.setZero();
      gradient0.setZero();

      // initialize input output image at layer l
      cv::Mat img0, img1;
      img0 = layers0[l - 1];
      img1 = layers1[l - 1];
      cv_gradient_x(img0, gradient_x);
      cv_gradient_y(img0, gradient_y);

      i = 0;
      sum0 = 0;

      mean0 = 0;

      valid_num0 = 0;

      grad_sum0.setZero();

      // calculate patch mean and scaled intensity of source image
      for (int i = 0; i < WINDOW_SIZE; i++) {
        float r = point.y / scale + pattern[i][1];
        float c = point.x / scale + pattern[i][0];
        if (r >= BOUNDARY && r < img0.size().height - BOUNDARY &&
            c >= BOUNDARY && c < img0.size().width - BOUNDARY) {
          float gx = bilinear_interpolate(gradient_x, c, r);
          float gy = bilinear_interpolate(gradient_y, c, r);
          gradient0.row(i) = Eigen::Vector2f(gx, gy);
          grad_sum0 += gradient0.row(i);

          intensity0(i) = bilinear_interpolate(img0, c, r);
          sum0 += intensity0(i);
          valid_num0++;
        } else {
          intensity0(i) = -1;
        }

        position0.row(i) = Eigen::Vector2f(r, c);
      }

      mean0 = sum0 / valid_num0;

      // calculate gradient and hessian inverse
      for (int i = 0; i < WINDOW_SIZE; i++) {
        if (intensity0(i) >= 0) {
          Eigen::Vector2f gradient_i = gradient0.row(i);
          intensity0(i) /= mean0;
          Eigen::Vector2f tmp0 = grad_sum0 * intensity0(i);
          Eigen::Vector2f tmp1 = gradient_i * sum0;
          Eigen::Vector2f tmp = (tmp1 - tmp0);
          gradient0.row(i) = valid_num0 * tmp / (sum0 * sum0);
        } else {
          gradient0.row(i).setZero();
        }
      }
      hessian = gradient0.transpose() * gradient0;
      Eigen::Matrix2f hessian_inv;
      hessian_inv.setIdentity();
      hessian.ldlt().solveInPlace(hessian_inv);

      Eigen::MatrixXf hessian_inv_gradient_trans =
          hessian_inv * gradient0.transpose();

      // start iteration
      for (int iter_num = 0; iter_num < 30; iter_num++) {
        sum1 = 0;
        mean1 = 0;
        valid_num1 = 0;

        // calculate patch mean and scaled intensity of target image
        for (int i = 0; i < WINDOW_SIZE; i++) {
          float r = position0(i, 0);
          float c = position0(i, 1);
          position1.row(i) = Eigen::Vector2f(r + motion(1), c + motion(0));
          if (r + motion(1) >= BOUNDARY &&
              r + motion(1) < img0.size().height - BOUNDARY &&
              c + motion(0) >= BOUNDARY &&
              c + motion(0) < img0.size().width - BOUNDARY) {
            intensity1(i) =
                bilinear_interpolate(img1, c + motion(0), r + motion(1));
            sum1 += intensity1(i);
            valid_num1++;
          } else {
            intensity1(i) = -1;
          }
        }
        mean1 = sum1 / valid_num1;
        int residual_num = 0;

        Eigen::MatrixXf res(WINDOW_SIZE, 1);
        // calculate residual
        for (int i = 0; i < WINDOW_SIZE; i++) {
          if (intensity0(i) >= 0 && intensity1(i) >= 0) {
            intensity1(i) /= mean1;
            res(i) = intensity1(i) - intensity0(i);
            residual_num++;
          } else {
            res(i) = 0;
          }
        }

        // break when 50% of patch is out of range!
        if (residual_num < (WINDOW_SIZE / 2)) {
          patch_out_flag = 1;
          break;
        }

        // calculate dmotion
        dmotion = -hessian_inv_gradient_trans * res;

        // multiply motion by 2 for lower layer
        motion_old = motion;
        motion = (motion + dmotion);

        // break iteration if error is small enough
        if ((motion - motion_old).norm() < 1e-1 * pow(2, l - 1)) {
          break;
        }

        // break iteration if key point is out of range of lower layer
        if (motion(0) + point.x / scale < 0 ||
            motion(0) + point.x / scale >= img0.size().width ||
            motion(1) + point.y / scale < 0 ||
            motion(1) + point.y / scale >= img0.size().height) {
          out_flag = 1;
          break;
        }
      }  // end of iteration loop
      if (out_flag == 1) {
        break;
      }
      if (patch_out_flag == 1) {
        break;
      }
      if (l > 1) {
        motion_old *= 2;
        motion *= 2;
      }

    }  // end of layer loop

    // push back error, status and destination points
    error = (motion - motion_old).norm();
    new_point.x = point.x + motion(0);
    new_point.y = point.y + motion(1);
    points1.push_back(new_point);
    errors.push_back(error);
    if (error < 1 && out_flag == 0) {
      status.push_back(1);

    } else {
      status.push_back(0);
    }
  }  // end of point loop
}

// TODO PROJECT: find trackid corresponding to featureid in current frame and
// camera
TrackId findTrackId(TimeCamId tcid, const Landmarks& landmarks,
                    FeatureId featid) {
  for (const auto trackid_landmark_pair : landmarks) {
    TrackId trackid = trackid_landmark_pair.first;
    if (trackid_landmark_pair.second.obs.find(tcid) !=
        trackid_landmark_pair.second.obs.end()) {
      FeatureId obs_featid = trackid_landmark_pair.second.obs.at(tcid);
      if (obs_featid == featid) return trackid;
    } else {
      continue;
    }
  }
  return -1;
}

// TODO PROJECT: use optical flow to calculate feature points in next left frame
// based on feature points in the last left frame
/*
 * Input:
 *  current_frame     : current frame id
 *  last_key_frame    : last key frame id
 *  imglt0            : left image of last frame
 *  imglt1            : left image of current frame
 *  kdlt0             : key points data of last frame
 *  kdlt1             : key points data of current frame
 *  md_feat2track     : match data that match feature id to track id
 * */
void OpticalFlowBetweenFrame_opencv_version(
    FrameId current_frame, FrameId last_key_frame,
    const pangolin::ManagedImage<uint8_t>& imglt0,
    const pangolin::ManagedImage<uint8_t>& imglt1, const KeypointsData& kdlt0,
    KeypointsData& kdlt1, MatchData& md_feat2track) {
  cv::Mat imglt0_cv(imglt0.h, imglt0.w, CV_8U, imglt0.ptr);
  cv::Mat imglt1_cv(imglt1.h, imglt1.w, CV_8U, imglt1.ptr);

  std::vector<cv::Point2f> points0;
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points0_back;

  std::vector<unsigned char> status;
  std::vector<unsigned char> status_back;

  std::vector<float> errors;
  std::vector<float> errors_back;

  cv::Size winSize(21, 21);

  // convert double vector to float vector in cv
  int start = clock();
  for (const auto p_2dl : kdlt0.corners) {
    cv::Point2f p_2d;
    p_2d.x = float(p_2dl(0));
    p_2d.y = float(p_2dl(1));
    points0.push_back(p_2d);
  }
  int stop = clock();

  start = clock();
  // // opencv implementation of optical flow
  //  cv::calcOpticalFlowPyrLK(imglt0_cv, imglt1_cv, points0, points1, status,
  //                           errors, winSize, 5);  //, winSize, 5);
  //  cv::calcOpticalFlowPyrLK(imglt1_cv, imglt0_cv, points1, points0_back,
  //                           status_back, errors_back, winSize,
  //                           5);  //, winSize, 5);

  // self implemented optical flow
  OpticalFLowLK(imglt0_cv, imglt1_cv, points0, points1, status, errors, 5);
  OpticalFLowLK(imglt1_cv, imglt0_cv, points1, points0_back, status_back,
                errors_back, 5);
  stop = clock();
  double duration = double(stop - start) / double(CLOCKS_PER_SEC);
  std::cout << "optical flow keypoints num: " << points0.size() << std::endl;
  std::cout << "optical flow time: " << duration << std::endl;

  kdlt1.corners.clear();
  kdlt1.trackids.clear();
  int j = 0;
  TimeCamId tcidl = std::make_pair(current_frame, 0);
  TimeCamId tcidl_last_key = std::make_pair(last_key_frame, 0);
  start = clock();
  for (int i = 0; i < points1.size(); i++) {
    if (status[i] && status_back[i]) {
      float distance = norm(points0[i] - points0_back[i]);

      if (distance < 1) {
        TrackId trackid = kdlt0.trackids[i];
        kdlt1.corners.emplace_back(points1[i].x, points1[i].y);
        kdlt1.trackids.push_back(trackid);
        md_feat2track.matches.push_back(std::make_pair(j, trackid));
        md_feat2track.inliers.push_back(std::make_pair(j, trackid));
        j++;
      }
    }
  }
  stop = clock();
  duration = double(stop - start) / double(CLOCKS_PER_SEC);
  std::cout << "optical flow push back time: " << duration << std::endl;
}

// TODO PROJECT: optical flow to right framge, difference is md_stereo is
// stored.
/*
 * Input:
 *  current_frame              : current frame id
 *  imgl                       : left image of current frame
 *  imgr                       : right image of current frame
 *  kdl                        : key points data of left image
 *  kdr                        : key points data of right image
 *  md_feat2track_right        : match data that match right feature
 *                               id to left track id
 *  md_stereo                  : match data that match feature id of
 *                               left and right image
 *  md_stereo_new              : stereo match data of newly detected
 *                               key points
 *  num_newly_added_keypoints  : number of newly detected key points
 *
 * */
void OpticalFlowToRightFrame_opencv_version(
    FrameId current_frame, const pangolin::ManagedImage<uint8_t>& imgl,
    const pangolin::ManagedImage<uint8_t>& imgr, const KeypointsData& kdl,
    KeypointsData& kdr, const Landmarks& landmarks,
    MatchData& md_feat2track_right, MatchData& md_stereo,
    MatchData& md_stereo_new, int num_newly_added_keypoints) {
  cv::Mat imgl_cv(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imgr_cv(imgr.h, imgr.w, CV_8U, imgr.ptr);

  std::vector<cv::Point2f> pointsl;
  std::vector<cv::Point2f> pointsr;
  std::vector<cv::Point2f> pointsl_back;  // points used for backward check

  std::vector<unsigned char> status;
  std::vector<unsigned char> status_back;

  std::vector<float> errors;
  std::vector<float> errors_back;

  cv::Size winSize(31, 31);

  // convert double vector to float vector in cv
  for (const auto p_2dl : kdl.corners) {
    cv::Point2f p_2d;
    p_2d.x = float(p_2dl(0));
    p_2d.y = float(p_2dl(1));
    pointsl.push_back(p_2d);
  }

  // // opencv implemented optical flow
  //  cv::calcOpticalFlowPyrLK(
  //      imgl_cv, imgr_cv, pointsl, pointsr, status,
  //      errors);  //                           winSize, 4);  // winSize, 4
  //  cv::calcOpticalFlowPyrLK(imgr_cv, imgl_cv, pointsr, pointsl_back,
  //  status_back,
  //                           errors_back);  //, winSize, 4);  // backward
  //                           check

  // self implemented optical flows
  OpticalFLowLK(imgl_cv, imgr_cv, pointsl, pointsr, status, errors);
  OpticalFLowLK(imgr_cv, imgl_cv, pointsr, pointsl_back, status_back,
                errors_back);

  kdr.corners.clear();
  md_feat2track_right.matches.clear();
  md_stereo_new.matches.clear();
  int j = 0;
  TimeCamId tcidl = std::make_pair(current_frame, 0);

  for (int i = 0; i < pointsr.size(); i++) {
    // pointsr should be the same size as pointsl and kdl.corners

    if (status[i] && status_back[i]) {
      float distance = norm(pointsl[i] - pointsl_back[i]);

      if (distance < 1) {
        kdr.corners.emplace_back(pointsr[i].x, pointsr[i].y);

        // we link the feature id of right frame to trackid
        // by checking the left corners' trackid
        if (i < (pointsl.size() - num_newly_added_keypoints)) {
          TrackId trackid = kdl.trackids[i];
          // findTrackId(tcidl, landmarks, i);  // trackid according to left
          md_feat2track_right.matches.push_back(std::make_pair(j, trackid));
        }

        md_stereo.matches.push_back(std::make_pair(i, j));
        md_stereo.inliers.push_back(std::make_pair(i, j));

        if (i >= pointsl.size() - num_newly_added_keypoints) {
          md_stereo_new.matches.push_back(std::make_pair(i, j));
          md_stereo_new.inliers.push_back(std::make_pair(i, j));
        }
        j++;
      }
    }
  }
}

// TODO PROJECT: optical flow to right framge, difference is md_stereo is
// stored.
/*
 * Input:
 *  current_frame              : current frame id
 *  imgl                       : left image of current frame
 *  imgr                       : right image of current frame
 *  kdl                        : key points data of left image
 *  kdr                        : key points data of right image
 *  md_feat2track_right        : match data that match right feature
 *                               id to left track id
 *  md_stereo                  : match data that match feature id of
 *                               left and right image
 *
 * */
void OpticalFlowFirstStereoPair_opencv_version(
    FrameId current_frame, const pangolin::ManagedImage<uint8_t>& imgl,
    const pangolin::ManagedImage<uint8_t>& imgr, const KeypointsData& kdl,
    KeypointsData& kdr, MatchData& md_stereo) {
  cv::Mat imgl_cv(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imgr_cv(imgr.h, imgr.w, CV_8U, imgr.ptr);

  std::vector<cv::Point2f> pointsl;
  std::vector<cv::Point2f> pointsr;
  std::vector<cv::Point2f> pointsl_back;  // points used for backward check

  std::vector<unsigned char> status;
  std::vector<unsigned char> status_back;
  std::vector<float> errors;
  std::vector<float> errors_back;
  cv::Size winSize(31, 31);

  // convert double vector to float vector in cv
  for (const auto p_2dl : kdl.corners) {
    cv::Point2f p_2d;
    p_2d.x = float(p_2dl(0));
    p_2d.y = float(p_2dl(1));
    pointsl.push_back(p_2d);
  }

  // // opencv implemented optical flow
  //  cv::calcOpticalFlowPyrLK(imgl_cv, imgr_cv, pointsl, pointsr, status,
  //  errors);
  //  cv::calcOpticalFlowPyrLK(imgr_cv, imgl_cv, pointsr, pointsl_back,
  //  status_back, errors_back);

  // self implemented optical flow
  OpticalFLowLK(imgl_cv, imgr_cv, pointsl, pointsr, status, errors);
  OpticalFLowLK(imgr_cv, imgl_cv, pointsr, pointsl_back, status_back,
                errors_back);
  kdr.corners.clear();
  int j = 0;

  for (int i = 0; i < pointsr.size(); i++) {  // ever input point in left cam

    if (int(status[i]) && int(status_back[i])) {
      float distance = norm(pointsl[i] - pointsl_back[i]);

      if (distance < 1) {
        std::cout << "distance of point " << i << " :" << pointsl[i] << ", "
                  << pointsr[i] << " , distance: " << distance
                  << ", status: " << int(status[i]) << ", "
                  << int(status_back[i]) << std::endl;
        kdr.corners.emplace_back(pointsr[i].x, pointsr[i].y);
        md_stereo.matches.push_back(std::make_pair(i, j));
        md_stereo.inliers.push_back(std::make_pair(i, j));
        j++;
      }
    }
  }
}

// TODO PROJECT: make grid in the image and store the top left corner and bottom
// right corner of each cell in a Cell object. The rnum and cnum should be
// devidable by h and w respectively.
/*
 * Input:
 *  h     : height of image
 *  w     : width of image
 *  rnum  : number of rows of cells
 *  cnum  : number of columns of cells
 *  cells : vector of Cell for output
 * */
void makeCells(int h, int w, int rnum, int cnum, std::vector<Cell>& cells) {
  int cellh = h / rnum;
  int cellw = w / cnum;
  for (int i = 0; i < rnum; i++) {
    for (int j = 0; j < cnum; j++) {
      int rmin = i * cellh;
      int rmax = (i + 1) * cellh - 1;
      int cmin = j * cellw;
      int cmax = (j + 1) * cellw - 1;
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
    if (trackid == -1)
      continue;
    else
      landmarks.at(trackid).obs.emplace(tcid, featid);
  }
}

// TODO PROJECT: add keypoints in right image to existing landmarks's
// observations
void add_points_to_landmark_obs_right(const MatchData& md_feat2track_right,
                                      const KeypointsData& kdr,
                                      Landmarks& landmarks,
                                      FrameId current_frame) {
  TimeCamId tcid = std::make_pair(current_frame, 1);
  for (const auto feat_track_pair : md_feat2track_right.matches) {
    FeatureId featid = feat_track_pair.first;
    TrackId trackid = feat_track_pair.second;
    if (trackid == -1)
      continue;
    else
      landmarks.at(trackid).obs.emplace(tcid, featid);
  }
}

// TODO PROJECT: check the number of keypoints in each grid and fulfill
// the variable kpnum of each cell.
/*
 * Input:
 *  kdlt1  : key points data of current left image
 *  cells  : vector of Cells
 * */
void check_num_points_in_cells(const KeypointsData& kdlt1,
                               std::vector<Cell>& cells) {
  for (auto& kp : kdlt1.corners) {
    for (auto& cell : cells) {
      if (kp[0] >= cell.topleft.second && kp[1] >= cell.topleft.first &&
          kp[0] < cell.bottomright.second && kp[1] < cell.bottomright.first) {
        cell.kpnum++;
        break;
      }
    }
  }
}

// TODO PROJECT: calculate the number of empty cells. empty cells indexes saved
// in empty_indices, and returns the number of empty cells.
/*
 * Input:
 *  cells        : vector of Cells
 *  empty_indices: vector of empty cell indices
 * */
int sparsity(std::vector<Cell>& cells, std::vector<int>& empty_indices) {
  int num_of_empty_cells = 0;
  for (int i = 0; i < cells.size(); i++) {
    if (cells[i].kpnum == 0) {
      num_of_empty_cells = num_of_empty_cells + 1;
      empty_indices.push_back(i);
    }
  }
  return num_of_empty_cells;
}

// TODO PROJECT: add new key points in subimage determined by empty cells.
/*
 * Input:
 *  empty_indices              : vector of empty cell indices
 *  num_newly_added_keypoints  : number of newly detected key points
 *  iml                        : left image
 *  kdl                        : key points data of left image
 *  cells                      : vector of Cells
 *  cellw                      : width of cell
 *  cellh                      : height of cell
 *  rnum                       : number of rows of cells
 *  cnum                       : number of columns of cells
 * */
void add_new_keypoints_from_empty_cells(
    std::vector<int> empty_indices, int& num_newly_added_keypoints,
    pangolin::ManagedImage<uint8_t>& imgl, KeypointsData& kdl,
    const std::vector<Cell> cells, int cellw, int cellh, int rnum, int cnum) {
  KeypointsData kd_new;
  int cell_newly_added_num_kp = 0;

  for (int j = 0; j < empty_indices.size(); j++) {  // every empty cell in cells
    // now the current empty cell is cells[empty_indices[j]]
    pangolin::ManagedImage<uint8_t> subimage;
    subimage.CopyFrom(imgl.SubImage(cells[empty_indices[j]].topleft.second,
                                    cells[empty_indices[j]].topleft.first,
                                    cellw, cellh));

    cell_newly_added_num_kp = 0;
    kd_new.corners.clear();

    int top_cell = (empty_indices[j] < cnum) ? 1 : 0;
    int bottom_cell = (empty_indices[j] >= (rnum - 1) * cnum) ? 1 : 0;
    int left_cell = (empty_indices[j] % cnum == 0) ? 1 : 0;
    int right_cell = (empty_indices[j] % cnum == (cnum - 1)) ? 1 : 0;
    if (top_cell == 0 && bottom_cell == 0 && left_cell == 0 &&
        right_cell == 0) {
      detectKeypoints_optical_flow_version(
          subimage, kd_new, 1,
          cell_newly_added_num_kp);  // -1 means no limit on maximum
                                     // num of detected features.
    }

    // add newly detected keypoints
    for (int i = 0; i < kd_new.corners.size(); i++) {
      kd_new.corners[i] +=
          Eigen::Vector2d(cells[empty_indices[j]].topleft.second,
                          cells[empty_indices[j]].topleft.first);
    }
    kdl.corners.insert(kdl.corners.end(), kd_new.corners.begin(),
                       kd_new.corners.end());
    num_newly_added_keypoints += cell_newly_added_num_kp;
  }
}

// TODO PROJECT: add new key points in whole image and then delete key points in
// non-empty-cells
/*
 * Input:
 *  empty_indices              : vector of empty cell indices
 *  num_newly_added_keypoints  : number of newly detected key points
 *  iml                        : left image
 *  kdl                        : key points data of left image
 *  cells                      : vector of Cells
 *  cellw                      : width of cell
 *  cellh                      : height of cell
 *  rnum                       : number of rows of cells
 *  cnum                       : number of columns of cells
 * */
void add_new_keypoints_from_empty_cells_v2(
    std::vector<int> empty_indices, int& num_newly_added_keypoints,
    pangolin::ManagedImage<uint8_t>& imgl, KeypointsData& kdl, int num_features,
    const std::vector<Cell> cells, int cellw, int cellh, int rnum, int cnum) {
  KeypointsData kd_tmp;
  detectKeypoints(imgl, kd_tmp, num_features);
  num_newly_added_keypoints = 0;
  for (const auto i : empty_indices) {
    for (const auto p_2d : kd_tmp.corners) {
      int top = cells[i].topleft.first;
      int left = cells[i].topleft.second;
      int bottom = cells[i].bottomright.first;
      int right = cells[i].bottomright.second;
      if (p_2d(0) >= left && p_2d(0) < right && p_2d(1) >= top &&
          p_2d(1) < bottom) {
        kdl.corners.push_back(p_2d);
        num_newly_added_keypoints++;
      }
    }
  }
}

// TODO PROJECT: triangulate only newly added key points pairs and add them to
// landmarks
void triangulate_new_part(const TimeCamId tcidl, const TimeCamId tcidr,
                          const KeypointsData& kdl, const KeypointsData& kdr,
                          const Sophus::SE3d& T_w_c0,
                          const Calibration& calib_cam,
                          const std::vector<int> inliers,
                          const MatchData& md_stereo_new_part,
                          const MatchData& md, Landmarks& landmarks,
                          TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // create bearing vectors for triangulation
  opengv::bearingVectors_t bv1;
  opengv::bearingVectors_t bv2;

  for (int i = 0; i < md_stereo_new_part.inliers.size(); i++) {
    FeatureId featidl = md_stereo_new_part.inliers[i].first;
    FeatureId featidr = md_stereo_new_part.inliers[i].second;
    bv1.push_back(
        calib_cam.intrinsics[0]->unproject(kdl.corners[featidl]).normalized());
    bv2.push_back(
        calib_cam.intrinsics[1]->unproject(kdr.corners[featidr]).normalized());
  }

  for (int i = 0; i < md_stereo_new_part.inliers.size(); i++) {
    FeatureId featidl = md_stereo_new_part.inliers[i].first;
    FeatureId featidr = md_stereo_new_part.inliers[i].second;

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

// TODO PROJECT: add new landmark for first stereo pair, add trackid in
// kdl, kdr
void initializeLandmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                         KeypointsData& kdl, KeypointsData& kdr,
                         const Sophus::SE3d& T_w_c0,
                         const Calibration& calib_cam,
                         const MatchData& md_stereo, Landmarks& landmarks,
                         TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

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

  for (int i = 0; i < md_stereo.inliers.size(); i++) {
    FeatureId featidl = md_stereo.inliers[i].first;
    FeatureId featidr = md_stereo.inliers[i].second;

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

// TODO PROJECT: localize camera, but ignore keypoints with trackid -1
void localize_camera_optical_flow(
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const KeypointsData& kdl, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    const MatchData& md, MatchData& md_feat2track_left_recorded,
    Sophus::SE3d& T_w_c, std::vector<int>& inliers) {
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
  md_feat2track_left_recorded.matches.clear();
  for (const auto featid_trackid_pair : md.matches) {
    if (featid_trackid_pair.second == -1) {
      continue;
    } else {
      bearingvec1.push_back(
          cam->unproject(kdl.corners[featid_trackid_pair.first]).normalized());
      points_w.push_back(landmarks.at(featid_trackid_pair.second).p);
      md_feat2track_left_recorded.matches.push_back(featid_trackid_pair);
    }
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

void add_new_landmarks_optical_flow_version(
    const TimeCamId tcidl, const TimeCamId tcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Sophus::SE3d& T_w_c0,
    const Calibration& calib_cam, const std::vector<int> inliers,
    const MatchData& md_stereo, const MatchData& md, Landmarks& landmarks,
    TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  for (const auto inliers_ind : inliers) {
    FeatureId featidl = md.matches[inliers_ind].first;
    TrackId trackid = md.matches[inliers_ind].second;
    if (trackid == -1) continue;

    // add observations for matches in inliers
    landmarks.at(trackid).obs.emplace(tcidl, featidl);

    for (int stereo_i = 0; stereo_i < md_stereo.inliers.size(); stereo_i++) {
      FeatureId featidl_stereo = md_stereo.inliers[stereo_i].first;
      FeatureId featidr = md_stereo.inliers[stereo_i].second;

      // add obervations for stereo matches
      if (featidl_stereo == featidl) {
        landmarks.at(trackid).obs.emplace(tcidr, featidr);
        break;
      }
    }
  }
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
  // 0.0 Transform from imu to cam
  Mat3 R_c_i;
  Vec3 t_c_i;
  std::cout << "assign value to R_c_i" << std::endl;
  R_c_i << 0.0148655429818, -0.999880929698, 0.00414029679422, 0.999557249008,
      0.0149672133247, 0.025715529948, -0.0257744366974, 0.00375618835797,
      0.999660727178;
  t_c_i << -0.0216401454975, -0.064676986768, 0.00981073058949;

  std::cout << "Successfully assign value to R_c_i" << std::endl;

  Mat3X model_wrt_camframe = (R_c_i * model).colwise() + t_c_i;

  std::cout << "Successfully calculate model_wrt_camframe" << std::endl;

  // 0. Centroids
  const Vec3 centroid_data = data.rowwise().mean();
  const Vec3 centroid_model = model_wrt_camframe.rowwise().mean();

  // center both clouds to 0 centroid
  const Mat3X data_centered = data.colwise() - centroid_data;
  const Mat3X model_centered = model_wrt_camframe.colwise() - centroid_model;
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

  model_transformed = (R * model_wrt_camframe).colwise() + t;

  std::cout << "model transformation succeeds." << std::endl;
  // 4. Translational error
  if (1) {  // ate.count == 0) {
    // static_assert(ArrX::ColsAtCompileTime == 1);

    //    const Mat3X diff = data - ((s * R * model).colwise() + t);
    const Mat3X diff = data - ((R * model_wrt_camframe).colwise() + t);
    const ArrX errors = diff.colwise().norm().transpose();

    //  auto& ref = *ate;
    ate.rmse = std::sqrt(errors.square().sum() / errors.cols());
    ate.mean = errors.mean();
    ate.min = errors.minCoeff();
    ate.max = errors.maxCoeff();
    ate.count = errors.cols();
  }
  std::cout << "ate calculation succeeds." << std::endl;
  return Sophus::Sim3d(Sophus::RxSO3d(1, R), t);
}

}  // namespace visnav
