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

#include <fstream>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const TimeCamId& tcid0,
                                   const TimeCamId& tcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<TimeCamId> tcids = {tcid0, tcid1};
  if (!GetTracksInImages(tcids, feature_tracks, shared_track_ids)) {
    // GetTracksInImages returns all tracks ids that appear in tcid0 and tcid1
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map

  opengv::bearingVectors_t bearingvec1;
  opengv::bearingVectors_t bearingvec2;
  for (const auto i : shared_track_ids) {
    Eigen::Vector2d p_2d_1 =
        feature_corners.at(tcid0).corners[feature_tracks.at(i).at(tcid0)];
    Eigen::Vector2d p_2d_2 =
        feature_corners.at(tcid1).corners[feature_tracks.at(i).at(tcid1)];
    bearingvec1.push_back(
        calib_cam.intrinsics.at(tcid0.second)->unproject(p_2d_1).normalized());
    bearingvec2.push_back(
        calib_cam.intrinsics.at(tcid1.second)->unproject(p_2d_2).normalized());
  }
  Sophus::SE3d T_0_1 =
      cameras.at(tcid0).T_w_c.inverse() * cameras.at(tcid1).T_w_c;
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingvec1, bearingvec2, T_0_1.translation(), T_0_1.rotationMatrix());
  for (int i = 0; i < shared_track_ids.size(); i++) {
    opengv::point_t p_3dc = opengv::triangulation::triangulate(adapter, i);
    opengv::point_t p_3dw = cameras.at(tcid0).T_w_c * p_3dc;
    Landmark lm;
    lm.p = p_3dw;
    for (const auto featuretrack : feature_tracks.at(shared_track_ids[i])) {
      // only add observation to existing cameras
      if (cameras.find(featuretrack.first) != cameras.end()) {
        lm.obs.emplace(featuretrack);
      }
    }
    // if track already exists in landmarks, then don't add it
    // to speed up, we can overwrite existing landmark so that we don't
    // need to find track id
    if (landmarks.find(shared_track_ids[i]) == landmarks.end()) {
      landmarks[shared_track_ids[i]] = lm;
      new_track_ids.push_back(shared_track_ids[i]);
    }
  }
  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const TimeCamId& tcid0,
                                       const TimeCamId& tcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(tcid0.first == tcid1.first && tcid0.second != tcid1.second)) {
    std::cerr << "Images " << tcid0 << " and " << tcid1
              << " don't for a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  cameras[tcid0].T_w_c =
      Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  cameras[tcid1].T_w_c = calib_cam.T_i_c[1];

  add_new_landmarks_between_cams(tcid0, tcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);
  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const TimeCamId& tcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  opengv::bearingVectors_t bearingvecs;
  opengv::points_t p_3ds_w;
  for (const auto trackid : shared_track_ids) {
    Eigen::Vector2d p_2d =
        feature_corners.at(tcid).corners[feature_tracks.at(trackid).at(tcid)];
    opengv::bearingVector_t p_3d =
        calib_cam.intrinsics[tcid.second]->unproject(p_2d).normalized();
    bearingvecs.push_back(p_3d);
    p_3ds_w.push_back(landmarks.at(trackid).p);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingvecs, p_3ds_w);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;

  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();

  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  opengv::bearingVectors_t bearingvecs_inliers;
  opengv::points_t p_3ds_w_inliers;
  for (const auto inlier_id : ransac.inliers_) {
    bearingvecs_inliers.push_back(bearingvecs[inlier_id]);
    p_3ds_w_inliers.push_back(p_3ds_w[inlier_id]);
  }
  opengv::absolute_pose::CentralAbsoluteAdapter adapter_inliers(
      bearingvecs_inliers, p_3ds_w_inliers);
  adapter_inliers.sett(best_transformation.block(0, 3, 3, 1));
  adapter_inliers.setR(best_transformation.block(0, 0, 3, 3));
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter_inliers);
  T_w_c = Sophus::SE3d(nonlinear_transformation.block(0, 0, 3, 3),
                       nonlinear_transformation.block(0, 3, 3, 1));

  std::vector<int> inliers;
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, inliers);
  for (const auto inlier_id : inliers) {
    inlier_track_ids.push_back(shared_track_ids[inlier_id]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<TimeCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem

  // add parameter block for camera pose and set constant if cams are fixed
  for (const auto cam : cameras) {
    problem.AddParameterBlock(cameras[cam.first].T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    if (fixed_cameras.find(cam.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(cameras[cam.first].T_w_c.data());
    }
  }
  problem.AddParameterBlock(calib_cam.intrinsics[0]->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics[1]->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  }

  // add residual blocks
  for (const auto landmark : landmarks) {
    for (const auto feat : landmark.second.obs) {
      //      auto it = std::find(landmark.second.outlier_obs.begin(),
      //                          landmark.second.outlier_obs.end(), feat);
      //      if (it != landmark.second.outlier_obs.end()) {
      //        std::cout << "one outlier" << std::endl;
      //        continue;
      //      }
      TimeCamId tcid = feat.first;
      FeatureId featid = feat.second;

      BundleAdjustmentReprojectionCostFunctor* baf =
          new BundleAdjustmentReprojectionCostFunctor(
              feature_corners.at(tcid).corners[featid],
              calib_cam.intrinsics[tcid.second]->name());

      ceres::CostFunction* costfunction = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(baf);

      if (options.use_huber) {
        problem.AddResidualBlock(
            costfunction, new ceres::HuberLoss(options.huber_parameter),
            cameras[tcid].T_w_c.data(), landmarks[landmark.first].p.data(),
            calib_cam.intrinsics[tcid.second]->data());
      } else {
        problem.AddResidualBlock(costfunction, NULL, cameras[tcid].T_w_c.data(),
                                 landmarks[landmark.first].p.data(),
                                 calib_cam.intrinsics[tcid.second]->data());
      }
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
