# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from IOU4cls_single import ansDetResults
SAVE_SMALL = False
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
gt_file = wkdir + '/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
xml_path = wkdir + "/data/VOCdevkit2007/VOC2007/Annotations/"
pic_path = wkdir + "/data/VOCdevkit2007/VOC2007/JPEGImages/"
save_path = './'
re_file = open('ret_iter115_big6_iter_870000.txt', 'r')
relists = re_file.readlines()
start = 1.0
end = 3.0
step = 0.1
for multi in np.arange(start,end, step):
    total_box, distriDictO, distriDictS = ansDetResults(relists, xml_path, pic_path, save_path, multi, SAVE_SMALL)
    print total_box, " boxes in ",len(relists), "pics"
    if multi == start:
        print "\nbefore enlarge area distribute:"
        acc = 0
        for key in sorted( distriDictO, reverse=True):
            acc += distriDictO[key]
            print "%2d: %6d, %5.1f %%, %5.1f %%"%(key, distriDictO[key], 100*distriDictO[key]/float(total_box), 100*acc/float(total_box)  )
    print "\nenlarge scale %.1f :"%(multi)
    acc = 0
    for key in sorted( distriDictS, reverse=True):
        acc += distriDictS[key]
        print "%2d: %6d, %5.1f %%, %5.1f %%"%(key, distriDictS[key], 100*distriDictS[key]/float(total_box), 100*acc/float(total_box)  )


#plt.subplot(3,1,1)  # up
#plt.hist(maxIOUs, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
#plt.title("ious with gt by detection")
#plt.xlabel("iou with gt")
#
#plt.subplot(3,1,2)  # middle
#plt.hist(area_before, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
#plt.title("overlap with gt by detection")
#plt.xlabel("re areas in gt before")
#
#plt.subplot(3,1,3)  # down
#plt.hist(area_afterS, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
#plt.title("overlap with gt after enlarge %.2f"%(multi_s))
#plt.xlabel("re areas in gt after enlarge %.2f"%(multi_s))
#plt.show()
