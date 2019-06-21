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
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <pangolin/image/managed_image.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <visnav/common_types.h>

namespace visnav {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

typedef std::bitset<256> Descriptor;

char pattern_31_x_a[256] = {
    8,   4,   -11, 7,   2,   1,   -2,  -13, -13, 10,  -13, -11, 7,   -4,  -13,
    -9,  12,  -3,  -6,  11,  4,   5,   3,   -8,  -2,  -13, -7,  -4,  -10, 5,
    5,   1,   9,   4,   2,   -4,  -8,  4,   0,   -13, -3,  -6,  8,   0,   7,
    -13, 10,  -6,  10,  -13, -13, 3,   5,   -1,  3,   2,   -13, -13, -13, -7,
    6,   -9,  -2,  -12, 3,   -7,  -3,  2,   -11, -1,  5,   -4,  -9,  -12, 10,
    7,   -7,  -4,  7,   -7,  -13, -3,  7,   -13, 1,   2,   -4,  -1,  7,   1,
    9,   -1,  -13, 7,   12,  6,   5,   2,   3,   2,   9,   -8,  -11, 1,   6,
    2,   6,   3,   7,   -11, -10, -5,  -10, 8,   4,   -10, 4,   -2,  -5,  7,
    -9,  -5,  8,   -9,  1,   7,   -2,  11,  -12, 3,   5,   0,   -9,  0,   -1,
    5,   3,   -13, -5,  -4,  6,   -7,  -13, 1,   4,   -2,  2,   -2,  4,   -6,
    -3,  7,   4,   -13, 7,   7,   -7,  -8,  -13, 2,   10,  -6,  8,   2,   -11,
    -12, -11, 5,   -2,  -1,  -13, -10, -3,  2,   -9,  -4,  -4,  -6,  6,   -13,
    11,  7,   -1,  -4,  -7,  -13, -7,  -8,  -5,  -13, 1,   1,   9,   5,   -1,
    -9,  -1,  -13, 8,   2,   7,   -10, -10, 4,   3,   -4,  5,   4,   -9,  0,
    -12, 3,   -10, 8,   -8,  2,   10,  6,   -7,  -3,  -1,  -3,  -8,  4,   2,
    6,   3,   11,  -3,  4,   2,   -10, -13, -13, 6,   0,   -13, -9,  -13, 5,
    2,   -1,  9,   11,  3,   -1,  3,   -13, 5,   8,   7,   -10, 7,   9,   7,
    -1};

char pattern_31_y_a[256] = {
    -3,  2,   9,   -12, -13, -7,  -10, -13, -3,  4,   -8,  7,   7,   -5,  2,
    0,   -6,  6,   -13, -13, 7,   -3,  -7,  -7,  11,  12,  3,   2,   -12, -12,
    -6,  0,   11,  7,   -1,  -12, -5,  11,  -8,  -2,  -2,  9,   12,  9,   -5,
    -6,  7,   -3,  -9,  8,   0,   3,   7,   7,   -10, -4,  0,   -7,  3,   12,
    -10, -1,  -5,  5,   -10, -7,  -2,  9,   -13, 6,   -3,  -13, -6,  -10, 2,
    12,  -13, 9,   -1,  6,   11,  7,   -8,  -7,  -3,  -6,  3,   -13, 1,   -1,
    1,   -9,  -13, 7,   -5,  3,   -13, -12, 8,   6,   -12, 4,   12,  12,  -9,
    3,   3,   -3,  8,   -5,  11,  -8,  5,   -1,  -6,  12,  -2,  0,   -8,  -6,
    -13, -13, -8,  -11, -8,  -4,  1,   -6,  -9,  7,   5,   -4,  12,  7,   2,
    11,  5,   -4,  9,   -7,  5,   6,   6,   -10, 1,   -2,  -12, -13, 1,   -10,
    -13, 5,   -2,  9,   1,   -8,  -4,  11,  6,   4,   -5,  -5,  -3,  -12, -2,
    -13, 0,   -3,  -13, -8,  -11, -2,  9,   -3,  -13, 6,   12,  -11, -3,  11,
    11,  -5,  12,  -8,  1,   -12, -2,  5,   -1,  7,   5,   0,   12,  -8,  11,
    -3,  -10, 1,   -11, -13, -13, -10, -8,  -6,  12,  2,   -13, -13, 9,   3,
    1,   2,   -10, -13, -12, 2,   6,   8,   10,  -9,  -13, -7,  -2,  2,   -5,
    -9,  -1,  -1,  0,   -11, -4,  -6,  7,   12,  0,   -1,  3,   8,   -6,  -9,
    7,   -6,  5,   -3,  0,   4,   -6,  0,   8,   9,   -4,  4,   3,   -7,  0,
    -6};

char pattern_31_x_b[256] = {
    9,   7,  -8, 12,  2,   1,  -2,  -11, -12, 11,  -8,  -9,  12,  -3,  -12, -7,
    12,  -2, -4, 12,  5,   10, 6,   -6,  -1,  -8,  -5,  -3,  -6,  6,   7,   4,
    11,  4,  4,  -2,  -7,  9,  1,   -8,  -2,  -4,  10,  1,   11,  -11, 12,  -6,
    12,  -8, -8, 7,   10,  1,  5,   3,   -13, -12, -11, -4,  12,  -7,  0,   -7,
    8,   -4, -1, 5,   -5,  0,  5,   -4,  -9,  -8,  12,  12,  -6,  -3,  12,  -5,
    -12, -2, 12, -11, 12,  3,  -2,  1,   8,   3,   12,  -1,  -10, 10,  12,  7,
    6,   2,  4,  12,  10,  -7, -4,  2,   7,   3,   11,  8,   9,   -6,  -5,  -3,
    -9,  12, 6,  -8,  6,   -2, -5,  10,  -8,  -5,  9,   -9,  1,   9,   -1,  12,
    -6,  7,  10, 2,   -5,  2,  1,   7,   6,   -8,  -3,  -3,  8,   -6,  -5,  3,
    8,   2,  12, 0,   9,   -3, -1,  12,  5,   -9,  8,   7,   -7,  -7,  -12, 3,
    12,  -6, 9,  2,   -10, -7, -10, 11,  -1,  0,   -12, -10, -2,  3,   -4,  -3,
    -2,  -4, 6,  -5,  12,  12, 0,   -3,  -6,  -8,  -6,  -6,  -4,  -8,  5,   10,
    10,  10, 1,  -6,  1,   -8, 10,  3,   12,  -5,  -8,  8,   8,   -3,  10,  5,
    -4,  3,  -6, 4,   -10, 12, -6,  3,   11,  8,   -6,  -3,  -1,  -3,  -8,  12,
    3,   11, 7,  12,  -3,  4,  2,   -8,  -11, -11, 11,  1,   -9,  -6,  -8,  8,
    3,   -1, 11, 12,  3,   0,  4,   -10, 12,  9,   8,   -10, 12,  10,  12,  0};

char pattern_31_y_b[256] = {
    5,   -12, 2,   -13, 12,  6,   -4,  -8,  -9,  9,   -9,  12,  6,   0,  -3,
    5,   -1,  12,  -8,  -8,  1,   -3,  12,  -2,  -10, 10,  -3,  7,   11, -7,
    -1,  -5,  -13, 12,  4,   7,   -10, 12,  -13, 2,   3,   -9,  7,   3,  -10,
    0,   1,   12,  -4,  -12, -4,  8,   -7,  -12, 6,   -10, 5,   12,  8,  7,
    8,   -6,  12,  5,   -13, 5,   -7,  -11, -13, -1,  2,   12,  6,   -4, -3,
    12,  5,   4,   2,   1,   5,   -6,  -7,  -12, 12,  0,   -13, 9,   -6, 12,
    6,   3,   5,   12,  9,   11,  10,  3,   -6,  -13, 3,   9,   -6,  -8, -4,
    -2,  0,   -8,  3,   -4,  10,  12,  0,   -6,  -11, 7,   7,   12,  2,  12,
    -8,  -2,  -13, 0,   -2,  1,   -4,  -11, 4,   12,  8,   8,   -13, 12, 7,
    -9,  -8,  9,   -3,  -12, 0,   12,  -2,  10,  -4,  -13, 12,  -6,  3,  -5,
    1,   -11, -7,  -5,  6,   6,   1,   -8,  -8,  9,   3,   7,   -8,  8,  3,
    -9,  -5,  8,   12,  9,   -5,  11,  -13, 2,   0,   -10, -7,  9,   11, 5,
    6,   -2,  7,   -2,  7,   -13, -8,  -9,  5,   10,  -13, -13, -1,  -9, -13,
    2,   12,  -10, -6,  -6,  -9,  -7,  -13, 5,   -13, -3,  -12, -1,  3,  -9,
    1,   -8,  9,   12,  -5,  7,   -8,  -12, 5,   9,   5,   4,   3,   12, 11,
    -13, 12,  4,   6,   12,  1,   1,   1,   -13, -13, 4,   -2,  -3,  -2, 10,
    -9,  -1,  -2,  -8,  5,   10,  5,   5,   11,  -6,  -12, 9,   4,   -2, -2,
    -11};

void detectKeypoints(const pangolin::ManagedImage<uint8_t>& img_raw,
                     KeypointsData& kd, int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(image, points, num_features, 0.01, 8);

  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.corners.emplace_back(points[i].x, points[i].y);
    }
  }
}

void computeAngles(const pangolin::ManagedImage<uint8_t>& img_raw,
                   KeypointsData& kd, bool rotate_features) {
  kd.corner_angles.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d& p = kd.corners[i];

    const int cx = p[0];
    const int cy = p[1];

    double angle = 0;

    if (rotate_features) {
      // TODO SHEET 3: compute angle
      double m01 = 0;
      double m10 = 0;
      for (int x = -15; x <= 15; ++x) {
        for (int y = -15; y <= 15; ++y) {
          bool left = (cx - x >= 0);
          bool right = (cx + x < img_raw.w);
          bool up = (cy - y >= 0);
          bool down = (cy + y < img_raw.h);
          if ((x * x + y * y <= 225) && left && right && up && down) {
            m01 += (img_raw(x + cx, y + cy) * y);
            m10 += (img_raw(x + cx, y + cy) * x);
          }
        }
      }
      angle = atan2(m01, m10);
    }

    kd.corner_angles[i] = angle;
  }
}

void computeDescriptors(const pangolin::ManagedImage<uint8_t>& img_raw,
                        KeypointsData& kd) {
  kd.corner_descriptors.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    std::bitset<256> descriptor;

    const Eigen::Vector2d& p = kd.corners[i];
    double angle = kd.corner_angles[i];

    int cx = p[0];
    int cy = p[1];

    // TODO SHEET 3: compute descriptor
    for (int j = 0; j < 256; ++j) {
      double p_a_prime_x = round((pattern_31_x_a[j]) * cos(angle) +
                                 (pattern_31_y_a[j]) * (-sin(angle)) + cx);
      double p_a_prime_y = round((pattern_31_x_a[j]) * sin(angle) +
                                 (pattern_31_y_a[j]) * (cos(angle)) + cy);
      double p_b_prime_x = round((pattern_31_x_b[j]) * cos(angle) +
                                 (pattern_31_y_b[j]) * (-sin(angle)) + cx);
      double p_b_prime_y = round((pattern_31_x_b[j]) * sin(angle) +
                                 (pattern_31_y_b[j]) * (cos(angle)) + cy);
      if (img_raw(p_a_prime_x, p_a_prime_y) < img_raw(p_b_prime_x, p_b_prime_y))
        descriptor[j] = 1;
      else
        descriptor[j] = 0;
    }

    kd.corner_descriptors[i] = descriptor;
  }
}

void detectKeypointsAndDescriptors(
    const pangolin::ManagedImage<uint8_t>& img_raw, KeypointsData& kd,
    int num_features, bool rotate_features) {
  detectKeypoints(img_raw, kd, num_features);
  computeAngles(img_raw, kd, rotate_features);
  computeDescriptors(img_raw, kd);
}

void matchFastHelper(const std::vector<std::bitset<256>>& corner_descriptors_1,
                     const std::vector<std::bitset<256>>& corner_descriptors_2,
                     std::map<int, int>& matches, int threshold,
                     double test_dist) {
  matches.clear();

  for (size_t i = 0; i < corner_descriptors_1.size(); i++) {
    int best_idx = -1, best_dist = 500;
    int best2_dist = 500;

    for (size_t j = 0; j < corner_descriptors_2.size(); j++) {
      int dist = (corner_descriptors_1[i] ^ corner_descriptors_2[j]).count();

      if (dist <= best_dist) {
        best2_dist = best_dist;

        best_dist = dist;
        best_idx = j;
      } else if (dist < best2_dist) {
        best2_dist = dist;
      }
    }

    if (best_dist < threshold && best_dist * test_dist <= best2_dist) {
      matches.emplace(i, best_idx);
    }
  }
}

void matchDescriptors(const std::vector<std::bitset<256>>& corner_descriptors_1,
                      const std::vector<std::bitset<256>>& corner_descriptors_2,
                      std::vector<std::pair<int, int>>& matches, int threshold,
                      double dist_2_best) {
  matches.clear();

  // TODO SHEET 3: match features
  std::map<int, int> map_1_to_2;
  std::map<int, int> map_2_to_1;
  matchFastHelper(corner_descriptors_1, corner_descriptors_2, map_1_to_2,
                  threshold, dist_2_best);
  matchFastHelper(corner_descriptors_2, corner_descriptors_1, map_2_to_1,
                  threshold, dist_2_best);
  for (auto const& p : map_1_to_2) {
    if (p.first == map_2_to_1[p.second]) {
      matches.push_back(std::make_pair(p.first, p.second));
    }
  }
}

// TODO PROJECT: compute patch region for optical flow(inside the radius range)
void computePatch(const cv::Mat& img, const Eigen::Vector2i& patch_center,
                  const int optf_patch_radius,
                  std::vector<Eigen::Vector2i>& patch_region) {
  for (int r = patch_center[0] - optf_patch_radius;
       r < patch_center[0] + optf_patch_radius; r++) {
    for (int c = patch_center[1] - optf_patch_radius;
         c < patch_center[1] + optf_patch_radius; c++) {
      if (!(r < 0 || c < 0 || r >= img.size().height ||
            c >= img.size().width)) {
        if (r * r + c * c <= optf_patch_radius) {
          patch_region.push_back(Eigen::Vector2i(r, c));
        }
      }
    }
  }
}

// TODO PROJECT: compute patch region of target image after T transform
void transformPatch(const cv::Mat& img, const Eigen::Matrix2d& rot,
                    const Eigen::Vector2d& trans,
                    const std::vector<Eigen::Vector2i>& patch_region0,
                    std::vector<Eigen::Vector2i> patch_region1,
                    std::vector<std::pair<int, int>> corresponding_pixel_inds) {
  int i0 = 0;
  int i1 = 0;
  for (const auto p_2d0 : patch_region0) {
    Eigen::Vector2i p_2d1;
    p_2d1(0) = round(rot(0, 0) * p_2d0(0) + rot(0, 1) * p_2d0(1) + trans(0));
    p_2d1(1) = round(rot(1, 0) * p_2d0(0) + rot(1, 1) * p_2d0(1) + trans(1));
    if (!(p_2d1(0) < 0 || p_2d1(1) < 0 || p_2d1(0) >= img.size().height ||
          p_2d1(1) >= img.size().width)) {
      patch_region1.push_back(p_2d1);
      corresponding_pixel_inds.push_back(std::make_pair(i0, i1));
      i1++;
    }
    i0++;
  }
}

// TODO PROJECT: compute patch mean
double computePatchMean(const cv::Mat& img,
                        const std::vector<Eigen::Vector2i> patch_region) {
  double mean = 0;
  for (const auto p_2d : patch_region) {
    mean += img.at<uchar>(p_2d(0), p_2d(1));
  }
  mean = mean / double(patch_region.size());
  return mean;
}

// TODO PROJECT: construct a cost functor for optimal flow
struct OpticalFlowCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OpticalFlowCostFunctor(const cv::Mat img0, const cv::Mat img1,
                         Eigen::Vector2i p_2d0, double img0m, double img1m)
      : img0(img0), img1(img1), p_2d0(p_2d0), img0m(img0m), img1m(img1m) {}

  template <class T>
  bool operator()(T const* const sT_0_1, T* sResidual) const {
    Eigen::Map<Sophus::SE2<T> const> const T_0_1(sT_0_1);

    Eigen::Matrix2d rot = T_0_1.roatationMatrix();
    Eigen::Vector2d trans = T_0_1.translation();

    Eigen::Vector2i est_p_2d1;

    std::vector<Eigen::Vector2i> patch_region1;
    std::vector<Eigen::Vector2i> patch_region0;

    Eigen::Map<T, 1> residual(sResidual);

    est_p_2d1(0) =
        round(rot(0, 0) * p_2d0(0) + rot(0, 1) * p_2d0(1) + trans(0));
    est_p_2d1(1) =
        round(rot(1, 0) * p_2d0(0) + rot(1, 1) * p_2d0(1) + trans(1));

    residual[0] = T(img1.at<uchar>(est_p_2d1[0], est_p_2d1[1])) / T(img1m) -
                  T(img0.at<uchar>(p_2d0[0], p_2d0[1])) / T(img0m);
  }

 private:
  cv::Mat img0;
  cv::Mat img1;
  Eigen::Vector2i p_2d0;
  double img0m;
  double img1m;
};

// TODO PROJECT: use optical flow for features matching
/**
void matchOpticalFlow(const pangolin::ManagedImage<uint8_t>& imgl,
                      const pangolin::ManagedImage<uint8_t>& imgr,
                      KeypointsData& kdl, KeypointsData& kdr,
                      std::vector<std::pair<int, int>>& matches,
                      double test_dist_2_best, double threshold,
                      int optf_patch_radius) {
  cv::Mat imgl_cv(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imgr_cv(imgr.h, imgr.w, CV_8U, imgr.ptr);
  for (int i = 0; i < kdl.corners.size(); i++) {
    ceres::Problem problem;
    Sophus::SE2d T_0_1;
    std::vector<Eigen::Vector2i> patch_region0;
    std::vector<Eigen::Vector2i> patch_region1;
    std::vector<std::pair<int, int>> corresponding_pixel_inds;
    Eigen::Vector2i patch_center0 = kdl.corners[i];
    Eigen::Vector2i patch_center1;
    double img0m;
    double img1m;

    Eigen::Matrix2d rot = T_0_1.rotationMatrix();
    Eigen::Vector2d trans = T_0_1.translation();

    //    patch_center1(0) = round(rot(0, 0) * patch_center0(0) +
    //                             rot(0, 1) * patch_center0(1) + trans(0));
    //    patch_center1(1) = round(rot(1, 0) * patch_center0(0) +
    //                             rot(1, 1) * patch_center0(1) + trans(1));

    computePatch(imgl_cv, patch_center0, optf_patch_radius, patch_region0);
    transformPatch(imgr_cv, rot, trans, patch_region0, patch_region1,
                   corresponding_pixel_inds);
    computePatchMean(imgl_cv, patch_region0);
    computePatchMean(imgr_cv, patch_region1);

    // problem.AddParameterBlock(T_0_1.data(), Sophus::SE2d::num_parameters, new
    // Sophus::test::LocalParameterizationSE2);

    for (const auto ind_pair : corresponding_pixel_inds) {
      Eigen::Vector2i p_2d0 = patch_region0[ind_pair.first];
      Eigen::Vector2i p_2d1 = patch_region1[ind_pair.second];

      OpticalFlowCostFunctor* optf_cf =
          new OpticalFlowCostFunctor(imgl_cv, imgr_cv, p_2d0, img0m, img1m);

      ceres::CostFunction* optf_cost_function = new ceres::AutoDiffCostFunction<
          OpticalFlowCostFunctor, Sophus::SE2d::num_parameters, 1>(optf_cf);

      problem.AddResidualBlock(optf_cost_function, NULL, T_0_1.data());
    }

    // Solve
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 20;
    ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    Solve(ceres_options, &problem, &summary);

    // Compute target patch_center
    rot = T_0_1.rotationMatrix();
    trans = T_0_1.translation();

    patch_center1(0) = round(rot(0, 0) * patch_center0(0) +
                             rot(0, 1) * patch_center0(1) + trans(0));
    patch_center1(1) = round(rot(1, 0) * patch_center0(0) +
                             rot(1, 1) * patch_center0(1) + trans(1));

    double best_dist = 10000;
    double second_best_dist = 10000;
    int best_ind = -1;

    for (int j = 0; j < kdr.corners.size(); j++) {
      Eigen::Vector2i kp_2d1 = kdr.corners[j];
      double dist = (kp_2d1 - patch_center1).norm();
      if (dist < best_dist) {
        second_best_dist = best_dist;
        best_dist = dist;
        best_ind = j;
      } else if (dist < second_best_dist) {
        second_best_dist = dist;
      }
    }
    if (best_dist < threshold &&
        best_dist * test_dist_2_best < second_best_dist) {
      matches.push_back(std::make_pair(i, best_ind));
    }
  }
}

// TODO PROJECT: use optical flow in opencv to replace features matching
void matchStereoOpticalFlow_opencv_version(
    const pangolin::ManagedImage<uint8_t>& imgl,
    const pangolin::ManagedImage<uint8_t>& imgr, KeypointsData& kdl,
    KeypointsData& kdr, std::vector<std::pair<int, int>>& stereo_matches) {
  cv::Mat imgl_cv(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imgr_cv(imgr.h, imgr.w, CV_8U, imgr.ptr);
  std::vector<Eigen::Vector2f> pointsl;
  std::vector<Eigen::Vector2f> pointsr;
  std::vector<unsigned char> status;
  std::vector<float> error;
  cv::Size winSize(31, 31);
  for (const auto p_2dl : kdl.corners) {
    Eigen::Vector2f p_2d = p_2dl;
    pointsl.push_back(p_2d);
  }
  cv::calcOpticalFlowPyrLK(imgl_cv, imgr_cv, pointsl, pointsr, status, error,
                           winSize, 4);

  kdr.corners.clear();
  int j = 0;
  for (int i = 0; i < pointsr.size(); i++) {
    if (status[i]) {
      Eigen::Vector2d p_2d = pointsr[i];
      kdr.corners.push_back(p_2d);
      stereo_matches.push_back(std::make_pair(i, j));
      j++;
    }
  }
}
**/
}  // namespace visnav
