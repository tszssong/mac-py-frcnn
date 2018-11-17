path=/home/lijian/~/wowo/128_256_512
LOG=$path/log/log-`date +%Y-%m-%d-%H-%M-%S`.log
SOLVER=./stable_facealignment_fast_v3_new_solver.prototxt
WEIGHTS=./stable_facealignment_fast_v3_new.caffemodel
CAFFE=~/Caffe_Long/build/tools/caffe
$CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0 2>&1 | tee $LOG

