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

#include <visnav/common_types.h>
#include <sophus/se3.hpp>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implementntit

  Eigen::Matrix<T, 3, 1> a = xi;
  Eigen::Matrix<T, 3, 3> R;
  Eigen::Matrix<T, 3, 3> A;
  A = Eigen::Matrix<T, 3, 3>::Zero();
  a.normalize();
  T theta = sqrt(xi(0) * xi(0) + xi(1) * xi(1) + xi(2) * xi(2));
  A(1, 0) = a(2, 0);
  A(2, 0) = -a(1, 0);
  A(2, 1) = a(0, 0);
  A(0, 1) = -a(2, 0);
  A(0, 2) = a(1, 0);
  A(1, 2) = -a(0, 0);
  R = Eigen::Matrix<T, 3, 3>::Identity() * cos(theta) +
      (1 - cos(theta)) * a * a.transpose() + sin(theta) * A;
  return R;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  T mat_trace = mat.trace();
  T theta = acos((mat_trace - 1) / 2);
  Eigen::Matrix<T, 3, 1> a;
  Eigen::Matrix<T, 3, 1> cosine;
  if (theta == 0)
    cosine << 0, 0, 0;
  else {
    a(0, 0) = mat(2, 1) - mat(1, 2);
    a(1, 0) = mat(0, 2) - mat(2, 0);
    a(2, 0) = mat(1, 0) - mat(0, 1);
    a = a / (2 * sin(theta));
    cosine = theta * a;
  }

  return cosine;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> rol = xi.block(0, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> a = xi.block(3, 0, 3, 1);
  Eigen::Matrix<T, 3, 3> R;
  Eigen::Matrix<T, 3, 3> A = Eigen::Matrix<T, 3, 3>::Zero();
  Eigen::Matrix<T, 3, 3> J;
  Eigen::Matrix<T, 3, 1> J_times_rol;
  Eigen::Matrix<T, 4, 4> Ti;

  T theta = sqrt(a(0) * a(0) + a(1) * a(1) + a(2) * a(2));
  a.normalize();
  A(1, 0) = a(2, 0);
  A(2, 0) = -a(1, 0);
  A(2, 1) = a(0, 0);
  A(0, 1) = -a(2, 0);
  A(0, 2) = a(1, 0);
  A(1, 2) = -a(0, 0);
  R = Eigen::Matrix<T, 3, 3>::Identity() * cos(theta) +
      (1 - cos(theta)) * a * a.transpose() + sin(theta) * A;
  if (theta == 0)
    J_times_rol << 0, 0, 0;
  else {
    J = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() +
        (1 - sin(theta) / theta) * a * a.transpose() +
        ((1 - cos(theta)) / theta) * A;
    J_times_rol = J * rol;
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Ti(i, j) = R(i, j);
    }
  }
  for (int i = 0; i < 3; i++) {
    Ti(i, 3) = J_times_rol(i, 0);
  }
  for (int i = 0; i < 3; i++) {
    Ti(3, i) = 0;
  }
  Ti(3, 3) = 1;
  return Ti;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 6, 1> cosine;
  Eigen::Matrix<T, 3, 1> rol;
  Eigen::Matrix<T, 3, 1> phi;

  Eigen::Matrix<T, 3, 3> A = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> a;
  Eigen::Matrix<T, 3, 3> a_up = Eigen::Matrix<T, 3, 3>::Zero();
  Eigen::Matrix<T, 3, 3> J;
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);
  T mat_trace = A.trace();
  T theta = acos((mat_trace - 1) / 2);
  if (theta == 0)
    cosine << 0, 0, 0, 0, 0, 0;

  else {
    a(0, 0) = A(2, 1) - A(1, 2);
    a(1, 0) = A(0, 2) - A(2, 0);
    a(2, 0) = A(1, 0) - A(0, 1);
    a = a / (2 * sin(theta));
    phi = theta * a;
    a_up(1, 0) = a(2, 0);
    a_up(2, 0) = -a(1, 0);
    a_up(2, 1) = a(0, 0);
    a_up(0, 1) = -a(2, 0);
    a_up(0, 2) = a(1, 0);
    a_up(1, 2) = -a(0, 0);
    J = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() +
        (1 - sin(theta) / theta) * a * a.transpose() +
        ((1 - cos(theta)) / theta) * a_up;
    rol = J.inverse() * t;
    for (int i = 0; i < 3; i++) {
      cosine(i, 0) = rol(i, 0);
    }
    for (int i = 3; i < 6; i++) {
      cosine(i, 0) = phi(i - 3, 0);
    }
  }

  return cosine;
}

}  // namespace visnav
