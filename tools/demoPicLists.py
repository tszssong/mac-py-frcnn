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

CLASSES = ('__background__', 'heart', 'yeah', 'one', 'baoquan', 'five', 'bainian', 'zan')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'mmcv5': ('MMCV5',
                  'MMCV5_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def demo(net, im, nframe):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    print im_file

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]  #300*4矩阵
        cls_scores = scores[:, cls_ind]   #300行
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            break;
        cv2.imshow("detections", im);
        cv2.waitKey()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='mmcv5')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    #caffemodel = "/Users/momo/Desktop/gesture/from113/mmcv5_gesture__iter_550000.caffemodel"
    prototxt = "/Users/momo/Desktop/gesture/models/raw_prototxts/bn_test.prototxt"
    caffemodel = "/Users/momo/Desktop/gesture/from113/mmbn_iter_500000.caffemodel"
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    numberImg = 0;
    numberImgWithGester = 0;
    
    toDir = "/Users/momo/Desktop/gestureOut/out/"
    listfile=open(toDir+"../py_ret.txt", "w")
#    fromDir = "/Users/momo/wkspace/Data/test1w/"
    fromDir = "/Users/momo/wkspace/Data/test1w_notdetected/"
#    fromDir = "/Users/momo/wkspace/Data/gesture/xs_gesture/zan/"
    timeUsed = 0
    totalGesture = 0
    for filename in os.listdir(fromDir):
#    for filename in cfg.TRAIN.IMAGES_LIST:
        if not 'jpg' in filename:                          #skin DS_Store
            continue
        print(filename)
        numberImg += 1
        cfg.TRAIN.IMAGES_LIST = filename
       
        im = cv2.imread(fromDir+filename)
        cfg.TRAIN.IMAGES_LIST = fromDir+filename
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        print ('No.{:d} took {:.3f}s for '
           '{:d} object proposals').format(numberImg, timer.total_time, boxes.shape[0])
        timeUsed =timeUsed+timer.total_time
        CONF_THRESH = 0.9
        NMS_THRESH = 0.01
        numGesture = 0
       
        gestureboxes = {}
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1                                     #because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]  #300*4矩阵
            cls_scores = scores[:, cls_ind]                  #300行
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            print inds,dets[inds[0]]

            list = []
            n=0
            for i in inds:
                n = n+1
                numGesture = numGesture + 1
                bbox = dets[i, :4]
                score = dets[i, -1]
                list.append( int(bbox[0]) )
                list.append( int(bbox[1]) )
                list.append( int(bbox[2]) )
                list.append( int(bbox[3]) )
                cv2.rectangle(im, (bbox[0], bbox[1]),(bbox[2],bbox[3]),(0,255,0),6)
                cv2.putText(im, '{:s} {:.2f}'.format(cls, score),(10,100),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0),6,4,0)
            if n>0:
                list.insert(0,n)
                gestureboxes[cls] = list
        print 'list:', list
        print 'dict:',gestureboxes
        if numGesture>0:
            cv2.imshow("detections", im);
            cv2.waitKey()
            cv2.imwrite("/Users/momo/Desktop/gestureOut/pyTest/det/"+filename, im)
            listfile.write(filename+' ' +str(int(numGesture))+' ')
            for gkey,gvalue in gestureboxes.items():
                print gkey, ':',gvalue # gestureboxes.get(gkey)
                for idxbbox in xrange( gvalue[0] ):
                    listfile.write(gkey+' ')
                    listfile.write( str( gvalue[idxbbox*4+1] ) +' ' \
                                   +str( gvalue[idxbbox*4+2] ) +' ' \
                                   +str( gvalue[idxbbox*4+3] ) +' ' \
                                   +str( gvalue[idxbbox*4+4] ) +' ' )
            listfile.write('\n')
        
        totalGesture = totalGesture + numGesture
print '{:.2f}s used for {:d} pic, and {:d} gester detected'.format(timeUsed,numberImg,totalGesture)
