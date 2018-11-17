import numpy as np
import numpy.random as npr
import scipy.io as sio
import os, sys
import _init_paths
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse
import time
TestMode = 'old'
NumTest = 20000
dataDir = "/Users/momo/wkspace/Data/gesture/mtcnnTests/"
dataDir = "/Users/momo/Desktop/gesture/fromAli/reg/"
modelDir = "/Users/momo/Desktop/gesture/fromAli/reg/"
if __name__ == '__main__':
    caffe.set_mode_cpu()
    inputSize = 48
    mean = 128
    bbox_reg_net = caffe.Net(modelDir+"/48net.prototxt", modelDir+"/0717AsFrcnn1218tight_iter_110000.caffemodel", caffe.TEST)

    if TestMode == 'new':
        fid = open(dataDir + "/tightTestnotResize/posShuffed12696.txt","r")
    else:
        fid = open(dataDir + "/tightIOU/pos.txt","r")

    lines = fid.readlines()
    fid.close()
    cur_=0
   
    totalTime = 0
    for line in lines:
        cur_+=1
        if not line or cur_ == NumTest:
            break;
        words = line.split()
        image_file_name = dataDir + words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        
        label  = int(words[1])
        ctx = w/2
        cty = h/2
        
        dx = float(words[2])
        dy = float(words[3])
        dw  = float(words[4])
        dh  = float(words[5])
        
        roi_cx = ctx + dx*w
        roi_cy = cty + dy*h
        roi_w  = np.exp(dw)*w
        roi_h  = np.exp(dh)*h
        print roi_cx, roi_cy, roi_w, roi_h
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
        print roi
        
        cv2.rectangle(im,(int(roi_cx-roi_w/2), int(roi_cy-roi_h/2)),(int(roi_cx+roi_w/2), int(roi_cy+roi_h/2)),(0,255,255),4)
                      
        cv2.imshow("roi",im)
        cv2.waitKey()
