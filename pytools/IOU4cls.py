# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils_iou_box import get_re_from_line, get_gt_from_xml, compBox, IOU_multi, crop_image
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
gt_file = wkdir + '/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
xml_path = wkdir + "/data/VOCdevkit2007/VOC2007/Annotations/"
pic_path = wkdir + "/data/VOCdevkit2007/VOC2007/JPEGImages/"
re_file = 'ret_iter115_big6_iter_870000.txt'
multi_s = 1.6
multi_b = 1.8

distriDictS = {0.6:0, 0.7:0, 0.8:0, 0.9:0, 1:0}
distriDictB = {0.6:0, 0.7:0, 0.8:0, 0.9:0, 1:0}
maxIOUs = np.array([])
area_before = np.array([])
area_afterS = np.array([])
area_afterB = np.array([])
total_box_re = 0
for line in open(re_file, 'r'):
    picname = line.strip('\n').split('.')[0]
    nhand_re, handcls_re, boxes_re = get_re_from_line(line)
    total_box_re += nhand_re
    # print "re:", nhand_re, handcls_re, boxes_re
    im = cv2.imread(pic_path+picname+'.jpg')
    # cv2.imshow("im", im)
    xmlfilepath = xml_path+"/"+picname + '.xml'
    nhand_gt, handcls_gt, boxes_gt = get_gt_from_xml(xmlfilepath)
    # print "gt:", nhand_gt, handcls_gt, boxes_gt
    for rebox_idx in xrange(boxes_re.shape[0]):
        box = boxes_re[ rebox_idx ]
        max_iou, gtbox_idx = IOU_multi(box, boxes_gt)

        maxIOUs = np.append(maxIOUs, max_iou)
        # if maxiou < 0.7:
        #     print "iou not valid:",box, boxes_gt[gtbox_idx]
        #     continue
        b, a, crop_box = compBox(boxes_re[ rebox_idx ], boxes_gt[gtbox_idx], multi_s)
        area_before = np.append(area_before, b)
        area_afterS = np.append(area_afterS, a)
        cropped_im = crop_image(im, crop_box)
        for key in distriDictS:
            if a >= key:
                distriDictS[key] += 1
                savepath = './' + str(key) +'/'
                if not os.path.isdir( savepath ):
                    os.mkdir(savepath)
                cv2.imwrite(savepath + picname +'.jpg', cropped_im)

        _, a, _ = compBox(boxes_re[rebox_idx], boxes_gt[gtbox_idx], multi_b)
        area_afterB = np.append(area_afterB, a)
        for key in distriDictB:
            if a > key:
                distriDictB[key] += 1
        # print "iou = %.2f, overlapbefore = %.2f, overlapafter = %.2f"%(max_iou, b, a)

print "number box recalled: ",total_box_re
print "\nenlarge scale ", multi_s, ":"
for key in distriDictS:
    print key, distriDictS[key], distriDictS[key]/float(total_box_re)
print "\nenlarge scale ",multi_b, ": "
for key in distriDictB:
    print key, distriDictB[key], distriDictB[key]/float(total_box_re)

plt.subplot(2,2,1)  # left_up
plt.hist(maxIOUs, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("ious with gt by detection")
plt.xlabel("iou with gt")

plt.subplot(2,2,3)  # left_down
plt.hist(area_before, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("overlap with gt by detection")
plt.xlabel("re areas in gt before")

plt.subplot(2,2,2)  # right_up
plt.hist(area_afterS, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("overlap with gt after enlarge %.2f"%(multi_s))
plt.xlabel("re areas in gt after enlarge %.2f"%(multi_s))

plt.subplot(2,2,4)  # right_down
plt.hist(area_afterB, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("overlap with gt after enlarge %.2f"%(multi_b))
plt.xlabel("re areas in gt after enlarge %.2f"%(multi_b))
plt.show()