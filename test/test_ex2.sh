#!/bin/bash

set -x
set -e

if md5sum --status -c ../test/reprojection.md5;
then 
exit 0
fi

./calibration --dataset-path ../data/euroc_calib/ --cam-model ds --show-gui 0
echo "$GT_CALIB_DS" > gt_calib.json
../test/compare_json.py gt_calib.json opt_calib.json
rm opt_calib.json gt_calib.json
./calibration --dataset-path ../data/euroc_calib/ --cam-model kb4 --show-gui 0
echo "$GT_CALIB_KB4" > gt_calib.json
../test/compare_json.py gt_calib.json opt_calib.json
rm opt_calib.json gt_calib.json
./calibration --dataset-path ../data/euroc_calib/ --cam-model pinhole --show-gui 0
echo "$GT_CALIB_PINHOLE" > gt_calib.json
../test/compare_json.py gt_calib.json opt_calib.json
rm opt_calib.json gt_calib.json
./calibration --dataset-path ../data/euroc_calib/ --cam-model eucm --show-gui 0
echo "$GT_CALIB_EUCM" > gt_calib.json
../test/compare_json.py gt_calib.json opt_calib.json
rm opt_calib.json gt_calib.json

