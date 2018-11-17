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
    NumTest = 100
    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/get_box_symbol.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/only7_4_6iter34w.caffemodel", caffe.TEST)

#    only7_4_6iter104w.caffemodel
    fid = open("/Users/momo/wkspace/Data/gesture/mtcnnTests/randomSize/pos.txt","r")

    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    roi_list = []
    regloss = np.array([])
    roi_n = 0
    cls_n = 0
    totalTime = 0
    for line in lines:
        cur_+=1
        if cur_ == NumTest:
            break;
        words = line.split()
#        image_file_name = "/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/" + words[0] + '.jpg'
        image_file_name = "/Users/momo/wkspace/Data/gesture/mtcnnTests/" + words[0] + '.jpg'
#        print cur_, image_file_name
        im = cv2.imread(image_file_name)
#        display = cv2.imread(image_file_name)
#        cv2.imshow("display", im);
#        cv2.waitKey()
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        im = np.swapaxes(im, 0, 2)
        
        im -= mean
        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
#        roi_list.append([im,label,roi])

        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()
        totalTime += (endT48-startT48)
#        loss = np.sum((box_deltas-roi_list[cur_-1][2])**2)/2
        if label != 0:
            roi_n+=1
            box_deltas = out_['fullyconnected1'][0]
            box_deltas[0] = box_deltas[0] / w
            box_deltas[2] = box_deltas[2] / w
            box_deltas[1] = box_deltas[1] / h
            box_deltas[3] = box_deltas[3] / h
            regloss = np.append(regloss,np.sum((box_deltas-roi)**2)/2)
    
        print "bbox_reg:",box_deltas
    print "num:", cur_,  "reg loss mean=", np.mean(regloss),"reg loss std=", np.std(regloss), "time:", totalTime*1000/cur_, "ms"

#        print roi_list



