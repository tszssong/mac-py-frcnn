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
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse

import time


if __name__ == '__main__':
    caffe.set_mode_cpu()
    inputSize = 128
    mean = 128
    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/get_box_symbol.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/128_2w2_0130R04_iter_50000.caffemodel",  caffe.TEST)

    fid = open("/Users/momo/wkspace/Data/gesture/mtcnnTests/randomSize/pos.txt","r")
    fwrite = open("/Users/momo/Desktop/mmcv2_frcnn_proposal/pyimg.txt",'w')
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    roi_list = []
    for line in lines:
        cur_+=1
        if cur_ == 2:
            break;
        words = line.split()
        print words
#        image_file_name = "/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/" + words[0] + '.jpg'
        image_file_name = "/Users/momo/wkspace/Data/gesture/mtcnnTests/" + words[0] + '.jpg'
        print image_file_name
        im = cv2.imread(image_file_name)
#        display = cv2.imread(image_file_name)
#        cv2.imshow("display", im);
#        cv2.waitKey()
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
#        im = cv2.transpose(im)
        print im.shape
        im = np.swapaxes(im, 0, 2)
        print im.shape
#        print "before",im
#        im = (im - 127.5)/127.5
        im = im.astype(np.int)
#        for i in range(im.shape[0]):
#            for j in range(im.shape[1]):
#        fwrite.write(im[:])
        np.savetxt("/Users/momo/Desktop/mmcv2_frcnn_proposal/pyimg.txt",im[:,:,1])
#        im.tofile("/Users/momo/Desktop/mmcv2_frcnn_proposal/pyimg.txt")
        im -= mean
#        print "after",im
        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
        roi_list.append([im,label,roi])
        
        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()
        
        box_deltas = out_['fullyconnected1'][0]
        print "bbox_reg:",box_deltas
#        cv2.imshow("img",im)
#        cv2.waitKey()
#        print "      gt:",roi_list[cur_-1][2], "loss:", loss

#        print roi_list



