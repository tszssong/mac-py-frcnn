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
import numpy.random as npr
import scipy.io as sio
import os, sys
# Make sure that caffe is on the python path:
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse

import time
regNetSize = 128
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

def bbox_reg(boxes, deltas, nw, nh):
    print "bbox_trans_inv, boxes:", boxes
    print "bbox_trans_inv, deltas:", deltas
    print "new w, h:", nw,nh
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
#    print deltas[0]*float(nw/128),deltas[1]*float(nh/128),deltas[2]*float(nw/128),deltas[3]*float(nh/128)
    pred_boxes[0] = boxes[0] + deltas[0]*float(nw)# x1
    pred_boxes[1] = boxes[1] + deltas[1]*float(nh)# y1
    pred_boxes[2] = boxes[2] + deltas[2]*float(nw)# x2
    pred_boxes[3] = boxes[3] + deltas[3]*float(nh)# y2

    print "bbox_trans_inv, pred_boxes:", pred_boxes
    return pred_boxes

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    caffemodel = "/Users/momo/Desktop/gesture/from113/mmcv5_gesture__iter_550000.caffemodel"
#    caffemodel = "/Users/momo/Desktop/gesture/from113/mmcv5gfiveEnhance_iter_550000.caffemodel"
#    caffemodel = "/Users/momo/Desktop/gesture/fromAli/reg/mmcv5g_fiverot_iter_550000.caffemodel"

    caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

#    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/get_box_symbol.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/only7_4_6iter6w.caffemodel",  caffe.TEST)
#    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/48net.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/48net1iter74w.caffemodel",  caffe.TEST)
#48net3iter10w.caffemodel 1wRatio_0716-2323_iter_130000.caffemodel
#1wRatio_0716-2323_iter_1000000.caffemodel 1wneg_0717-2040_iter_100000.caffemodel
    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/48net.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/48net5five24w0935flipblur_iter_840000.caffemodel", caffe.TEST)
#    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/48net_nodropout.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/48net4_five0712_5k4stable_b256_iter_140000.caffemodel", caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    numberImg = 0;
    numberImgWithGester = 0;
    video = cv2.VideoCapture(0)
#    video = cv2.VideoCapture("/Users/momo/wkspace/Data/gesture/ali_five_1/my2.mp4")
    timeUsed = 0
    totalGesture = 0
    success, im = video.read()
    numFrame = 0
    detectFlag = 0
    print im.shape
    width = im.shape[1]
    height = im.shape[0]
    pred_boxes = np.array([width/5,height/5,width*4/5,height*4/5])
    lastbox = [0,0,0,0]
    showboxArray = np.array([0,0,0,0])
    showbox = np.array([width/5,height/5,width*4/5,height*4/5])
    while success:
        numberImg += 1
        filename = 'my'+str(numberImg)+'.jpg'
#        im=np.rot90(im,3)

        print "No.",numberImg, ":",im.shape
        width = im.shape[1]
        height = im.shape[0]
        cfg.TRAIN.IMAGES_LIST = '/Users/momo/wkspace/npd.jpg'
        timer = Timer()
        timer.tic()
#        if not detectFlag:
        startTfr = 0
        endTfr = 0
        startT48 = 0
        endT48 = 0
#        if numberImg%60==0 or detectFlag==0:
        if detectFlag==0:
            startTfr = time.clock()
            scores, boxes = im_detect(net, im)
            endTfr = time.clock()
            timer.toc()
#            print ('No.{:d} took {:.3f}s for '
#               '{:d} object proposals').format(numberImg, timer.total_time, boxes.shape[0])
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
                bbox = dets[0,:4]
                score = dets[0,-1]  #only use first box
#                print bbox
                pred_boxes = bbox
                detectFlag = 1
                cv2.rectangle(im, (pred_boxes[0], pred_boxes[1]),(pred_boxes[2],pred_boxes[3]),(0,255,0),6)
        else:
            inputSize = 48
            mean = 128
            w = pred_boxes[2] - pred_boxes[0]
            h = pred_boxes[3] - pred_boxes[1]
            ctrx = pred_boxes[0] + w/2
            ctry = pred_boxes[1] + h/2
#            print ctrx, ctry, w, h
#            nw = npr.randint(int(w * 0.9), np.ceil(2 * w))
            nw =  1.5*w
#            nh =  nw
            nh =  1.5*h
            nx1 = max(0, ctrx-nw/2)
            ny1 = max(0, ctry-nh/2)
            nx2 = min(im.shape[1], ctrx+nw/2)
            ny2 = min(im.shape[0], ctry+nh/2)
                    
#            print "new box:",nx1,ny1,nx2,ny2, nw,nh
            nbox=np.array([nx1, ny1, nx2, ny2])
            img_roi = im[ int(ny1):int(ny2),int(nx1):int(nx2) ]
            cv2.rectangle(im, (int(nx1), int(ny1)),(int(nx2), int(ny2)),(50,122,122),4)

            h,w,ch = img_roi.shape
            if h!=inputSize or w!=inputSize:
                img_roi = cv2.resize(img_roi,(int(max(1,inputSize)),int(max(1,inputSize))))
            img_roi = np.swapaxes(img_roi, 0, 2)
            img_roi -= mean

            bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
            bbox_reg_net.blobs['data'].data[...]=img_roi

            startT48 = time.clock()
            out_ = bbox_reg_net.forward()
            endT48 = time.clock()
                    
            box_deltas = out_['fc6-2'][0]
            prob = out_['prob'][0][1]
            if prob >= 0.2:
                detectFlag = 1
                pred_boxes = bbox_reg(nbox, box_deltas, nw, nh)
#                if showboxArray.shape[0]>5:
#                    showboxArray = np.delete(showboxArray, 0, axis = 0)
#                showboxArray = np.vstack((showboxArray, pred_boxes))
#                showbox = np.mean(showboxArray,axis=0)
#                cv2.rectangle(im, (int(showbox[0]), int(showbox[1])),(int(showbox[2]),int(showbox[3])),(0,0,255),6)
                cv2.rectangle(im, (pred_boxes[0], pred_boxes[1]),(pred_boxes[2],pred_boxes[3]),(0,0,255),6)
                print "change:",pred_boxes-lastbox
            else:
                detectFlag = 0
#                while showboxArray.shape[0]>0:
#                    showboxArray = np.delete(showboxArray, 0, axis = 0)
#                showbox = np.array([0,0,0,0])

            lastbox = pred_boxes
#
#        print "fr-CNN used:",(endTfr-startTfr)*1000,"ms"
#        print "boxReg used:", (endT48-startT48)*1000, "ms"
#        cv2.imwrite("/Users/momo/Desktop/gestureOut/det/"+filename, im)
#        im = cv2.resize(im,(width/2,height/2))
        cv2.imshow("detections", im);
        cv2.imwrite("/Users/momo/Desktop/gestureOut/out/"+str(numberImg)+'.jpg', im)
        cv2.waitKey(1)
        
        success, im = video.read()  #next frame

        
        totalGesture = totalGesture + numGesture
print '{:.2f}s used for {:d} pic, and {:d} gester detected'.format(timeUsed,numberImg,totalGesture)
