import numpy as np
import numpy.random as npr
import scipy.io as sio
import os, sys
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# Make sure that caffe is on the python path:
#TP/total: 0.831 1662 2000 reg loss mean= 0.0578566870533 reg loss std= 0.0755053833573 time: 8.7376425 ms
#porb mean= 0.849588467642 prob std= 0.281988721547
#TN/total: 0.9855 1971 2000
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse
import time
TestMode = 'new'
NumTest = 20000
TH=0.7
dataDir = "/Users/momo/wkspace/Data/gesture/mtcnnTests/"
modelDir = "/Users/momo/Desktop/gesture/fromAli/reg/"
if __name__ == '__main__':
    caffe.set_mode_cpu()
    inputSize = 48
    mean = 128
    bbox_reg_net = caffe.Net(modelDir+"/48net.prototxt", modelDir+"/0717AsFrcnn1218tight_iter_110000.caffemodel", caffe.TEST)

    if TestMode == 'new':
        fid = open(dataDir + "/tightTestnotResize/posShuffed12696.txt","r")
    else:
        fid = open(dataDir + "/randomSize/pos.txt","r")

    TP=0
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    regloss = np.array([])
    probs = np.array([])
    roi_n = 0
    cls_n = 0
    totalTime = 0
    for line in lines:
        cur_+=1
        if not line or cur_ == NumTest:
            break;
        words = line.split()
        image_file_name = dataDir + words[0] + '.jpg'

        if cur_%500 == 0:
            print cur_,
            sys.stdout.flush()
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        im = np.swapaxes(im, 0, 2)
        im -= mean
        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]

        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()
        totalTime += (endT48-startT48)
        
        prob = out_['prob'][0][1]
        probs = np.append(probs,prob)
        if label != 0:
            roi_n+=1
            box_deltas = out_['fc6-2'][0]
            regloss = np.append(regloss,np.sum((box_deltas-roi)**2)/2)
        if prob>TH:
            TP+=1
#        print words[0],':',box_deltas,prob
    print "TP/total:", float(TP)/float(cur_),TP, cur_,
    print "reg loss mean=", np.mean(regloss),"reg loss std=", np.std(regloss),"time:", totalTime*1000/cur_, "ms"
    print "porb mean=", np.mean(probs),"prob std=", np.std(probs)
    #####################Tests4Neg###################
    if TestMode == 'new':
        fid = open(dataDir + "/tightTestnotResize/negShuffed14768.txt","r")
    else:
        fid = open(dataDir + "/randomSize/pos.txt","r")
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    TN = 0
    cls_n = 0
    for line in lines:
        cur_+=1
        if not line or cur_ == NumTest:
            break;
        words = line.split()
        image_file_name = dataDir + words[0] + '.jpg'
        #        print cur_, image_file_name, reglosssum, roi_n
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        im = np.swapaxes(im, 0, 2)
        im -= mean
        label    = int(words[1])
        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()

        prob = out_['prob'][0][1]
        if prob<TH:
            TN+=1

    print "TN/total:",float(TN)/float(cur_), TN, cur_

