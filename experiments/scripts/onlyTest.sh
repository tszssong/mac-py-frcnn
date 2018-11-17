#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=150000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
echo "70000"
time ./tools/test_net.py --gpu ${GPU_ID} \
--def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
--net output/faster_rcnn_end2end/voc_2007_trainval/mmcv5_gesture__iter_70000.caffemodel \
--imdb ${TEST_IMDB} \
--cfg experiments/cfgs/faster_rcnn_end2end.yml \
${EXTRA_ARGS}
echo "80000"
time ./tools/test_net.py --gpu ${GPU_ID} \
--def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
--net output/faster_rcnn_end2end/voc_2007_trainval/mmcv5_gesture__iter_80000.caffemodel \
--imdb ${TEST_IMDB} \
--cfg experiments/cfgs/faster_rcnn_end2end.yml \
${EXTRA_ARGS}
echo "90000"
time ./tools/test_net.py --gpu ${GPU_ID} \
--def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
--net output/faster_rcnn_end2end/voc_2007_trainval/mmcv5_gesture__iter_90000.caffemodel \
--imdb ${TEST_IMDB} \
--cfg experiments/cfgs/faster_rcnn_end2end.yml \
${EXTRA_ARGS}
echo "100000"
time ./tools/test_net.py --gpu ${GPU_ID} \
--def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
--net output/faster_rcnn_end2end/voc_2007_trainval/mmcv5_gesture__iter_100000.caffemodel \
--imdb ${TEST_IMDB} \
--cfg experiments/cfgs/faster_rcnn_end2end.yml \
${EXTRA_ARGS}
echo "110000"
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net output/faster_rcnn_end2end/voc_2007_trainval/mmcv5_gesture__iter_110000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
