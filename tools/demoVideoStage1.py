# -*- coding: utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys
# Make sure that caffe is on the python path:
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse
from PIL import Image
CLASSES = ('__background__','hand')

def demo(net, im, nframe):
    """Detect object classes in an image using pre-computed object proposals."""

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    print scores, boxes

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.99
    NMS_THRESH = 0.01

    dets = np.hstack((boxes, scores)).astype(np.float32)

    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
        print "not detected hand"
#        im = im[:, :, (2, 1, 0)]
    else:
        print "scores:\n",scores
        print "boxes:\n", boxes
        print "dets:\n", dets
        for i in xrange(dets.shape[0]):
            if(dets[i][4]>CONF_THRESH):
                cv2.rectangle(im, (dets[i][0], dets[i][1]),(dets[i][2], dets[i][3]),(255,0,0),1)
                cv2.putText(im,str(dets[i][4] ), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
#
        # cv2.imshow("detections", im);

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/MMCV5S8/faster_rcnn_end2end/test.prototxt"
    caffemodel = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/MMCV5S8/mmcv5stride8bn128_neg01_iter_100000.caffemodel"

    # prototxt = "/Users/momo/Desktop/gesture/from113/MMCV5_stride16/test.prototxt"
    # caffemodel = "/Users/momo/Desktop/sdk/momocv2_model/original_model/object_detect/mmcv5stride16_iter_5250000.caffemodel"

    caffe.set_mode_cpu()
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    cfg.TRAIN.IMAGES_LIST ="/Users/momo/wkspace/BC1D8DD0-85AD-11E8-982C-994D6EEB64BE20180712_L.jpg"
    cfg.TEST.SCALES = [144,]
    cfg.TEST.MAX_SIZE = 256
    cfg.DEDUP_BOXES = 1./8.

    video = cv2.VideoCapture(0)
#    video = cv2.VideoCapture("/Users/momo/wkspace/Data/gesture/ali_five_1/1.mp4")
    numFrame = 0
    while(1):
        ret,frame = video.read()
        numFrame = numFrame + 1
        demo(net, frame, numFrame)
        cv2.imshow("capture", frame)
        cv2.waitKey(1)
    plt.show()
