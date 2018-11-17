# -*- coding: utf-8 -*-
#!/usr/bin/env python

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

    CONF_THRESH = 0.7
    NMS_THRESH = 0.01

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # 300*4矩阵
        cls_scores = scores[:, cls_ind]  # 300行
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print "post nms, dets", dets.shape
        if dets[0, -1] > CONF_THRESH:
            print dets[0]
            for i in xrange(dets.shape[0]):
                if (dets[i][4] > CONF_THRESH):
                    cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (255, 0, 0), 1)
                    cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True     # Use RPN for proposals
    prototxt = "/Users/momo/Desktop/gesture/from113/MMCV5_stride16/test.prototxt"
    caffemodel = "/Users/momo/Desktop/sdk/momocv2_model/original_model/object_detect/mmcv5stride16_iter_5250000.caffemodel"
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    cfg.TEST.SCALES = [144,]
    cfg.TEST.MAX_SIZE = 256
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10
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
