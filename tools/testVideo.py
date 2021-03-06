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

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.95
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
    cfg.TEST.HAS_RPN = False  # Use RPN for proposals

    # prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/MMCV5S8/faster_rcnn_end2end/test.prototxt"
    # caffemodel = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/MMCV5S8/mmcv5stride8bn128_neg01_iter_100000.caffemodel"
    prototxt = "/Users/momo/Desktop/gesture/from113/MMCV5_stride16/test.prototxt"
    caffemodel = "/Users/momo/Desktop/sdk/momocv2_model/original_model/object_detect/mmcv5stride16_iter_5250000.caffemodel"

    caffe.set_mode_cpu()
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    cfg.TEST.SCALES = [144,]
    cfg.TEST.MAX_SIZE = 256
    cfg.DEDUP_BOXES = 1./16.
    CONF_THRESH = 0.98
    NMS_THRESH = 0.01

    fromDir = "/Volumes/song/testVideos/0627/all/"
    readFile = open(fromDir + "../testlists.txt", "r")
    retfilename = '627_'+caffemodel.split('.')[0].split('/')[-1] + '_' + str(CONF_THRESH).split('.')[-1]
    toDir = '/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/retTests/ret/' + retfilename
    print toDir
    if not os.path.isdir(toDir):
        os.makedirs(toDir)
    writeFile = open(toDir + "/../" + retfilename + ".txt", "w")

    filelists = readFile.readlines()
    print filelists
    for filename in filelists:
        print "filename:", filename
        video_name = filename.split()[0]
        video = cv2.VideoCapture(fromDir + video_name + '.mp4')
        success, im = video.read()
        numFrame = 0
        while success:
            if video_name == "20180627_momo_0007":
                im = cv2.transpose(im)
            if video_name == "20180627_momo_0023":
                im = cv2.flip(im, 0)
            numFrame += 1
            savename = filename.split()[0] + '_f' + str(numFrame) + '.jpg'

            scores, boxes = im_detect(net, im)

            dets = np.hstack((boxes, scores)).astype(np.float32)

            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]


            nhand = 0
            for i in xrange(dets.shape[0]):
                if (dets[i][4] > CONF_THRESH):
                    nhand += 1

            if nhand > 0:
                writeFile.write(savename +' '+ str(nhand)+ ' ')
                for i in xrange(dets.shape[0]):
                    if (dets[i][4] > CONF_THRESH):
                        writeFile.write('hand ' \
                                        + str( int( dets[i][0]) ) +' ' \
                                        + str( int( dets[i][1]) ) +' ' \
                                        + str( int( dets[i][2]) ) +' ' \
                                        + str( int( dets[i][3]) ) + ' ')
                writeFile.write('\n')

            # for i in xrange(dets.shape[0]):
            #     if (dets[i][4] > CONF_THRESH):
            #         cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (255, 0, 0), 1)
            #         cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            #         cv2.imwrite(toDir+'/' + savename, im)

            # cv2.imshow("capture", im)
            # cv2.waitKey(1)
            success, im = video.read()

writeFile.close()
readFile.close()