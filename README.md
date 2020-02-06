## Vision-based Navigation

This code is a part of the practical course "Vision-based Navigation" (IN2106) taught at the Technical University of Munich.

It was originally developed for the winter term 2018.

The authors are Vladyslav Usenko and Nikolaus Demmel.

### License

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.

Parts of the code (`include/tracks.h`, `include/union_find.h`) are adapted from OpenMVG and distributed under an MPL 2.0 licence.

Parts of the code (`include/local_parameterization_se3.hpp`, `src/test_ceres_se3.cpp`) are adapted from Sophus and distributed under an MIT license.

Note also the different licenses of thirdparty submodules.


You can find [setup instructions here.](wiki/Setup.md)

## Project

The code from branch `project`, `project_eval` is extended from the original code by Huang. In the project, we use the optical flow method to replace the point descriptors for visual odometry.

### How to run the code

1. Follow the [setup instructions](wiki/Setup.md).

2. Download dataset.


```bash
cd data/
sh download_dataset.sh
```

3. To run the original visual odometry, checkout to branch `exercise-05`. To run optical flow version, checkout to branch `project`.

```bash
git checkout exercise-05 # to checkout to original code
git checkout project # to checkout to optical flow version
cmake .
cd ./build
make
./odometry --dataset-path ../data/V1-01-easy/mav0
```
