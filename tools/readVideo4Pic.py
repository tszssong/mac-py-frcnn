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
fromDir = "/Users/momo/wkspace/Data/gesture/videos4test/xjVideos/"
toDir = "/Users/momo/wkspace/Data/gesture/videos4test/negPic/"
if __name__ == '__main__':
    caffe.set_mode_cpu()
    fid = open(fromDir + "../xjRet.txt","r")

    lines = fid.readlines()
    fid.close()
    videolists = []
    imagelists = []

    cur_=0
    sum_=len(lines)
    for line in lines:
        cur_+=1
        if not line:
            break;
        words = line.split(' ')

        image_file_name = words[0].split('.')[0]
        if not image_file_name in imagelists:
            imagelists.append(image_file_name)
        video_file_name = image_file_name.split('_f')[0]
        #print video_file_name
        if not video_file_name in videolists:
            videolists.append(video_file_name)

    print len(imagelists)
    print len(videolists)
    print imagelists
    print videolists

    for videoname in videolists:
        videofilename = videoname + '.mp4'                           #skin DS_Store
        print(videofilename)
        video = cv2.VideoCapture(fromDir+videofilename)

        success, im = video.read()
        numFrame = 0
        while success:
            numFrame += 1
            picfilename = videoname + '_f' + str(numFrame)
            if picfilename in imagelists:
                cv2.imwrite(toDir + picfilename + '.jpg', im)
            success, im = video.read()

