# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils_iou_box import get_re_from_line, get_gt_from_xml, compBox, IOU_multi, crop_image
SAVE_SMALL = False
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
gt_file = wkdir + '/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
xml_path = wkdir + "/data/VOCdevkit2007/VOC2007/Annotations/"
pic_path = wkdir + "/data/VOCdevkit2007/VOC2007/JPEGImages/"
re_file = 'ret_iter115_big6_iter_870000.txt'
multi_s = 1.8

distriDictO = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
distriDictS = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

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
        r_area = int(b*10)                  #python int same as floor: 1 = [1.0,1.999999~]
        if(distriDictO.has_key(r_area)):
            distriDictO[r_area] += 1
            if SAVE_SMALL:
                im = cv2.imread(pic_path+picname+'.jpg')
                savepath = './det_' + str(r_area) + '/'
                if not os.path.isdir(savepath):
                    os.mkdir(savepath)
                cropped_im = crop_image(im, boxes_re[ rebox_idx ])
                cv2.imwrite(savepath + picname + '.jpg', cropped_im)
    
        r_area = int(a*10)                  #python int same as floor: 1 = [1.0,1.999999~]
        if(distriDictS.has_key(r_area)):
            distriDictS[r_area] += 1
            if SAVE_SMALL:
                im = cv2.imread(pic_path+picname+'.jpg')
                savepath = './enlarge_' + str(multi_s)+'_'+ str(r_area)  + '/'
                if not os.path.isdir(savepath):
                    os.mkdir(savepath)
                cropped_im = crop_image(im, crop_box)
                cv2.imwrite(savepath + picname + '.jpg', cropped_im)



print "number box recalled: ",total_box_re
print "\nbefore enlarge area distribute:"
acc = 0
for key in sorted( distriDictO, reverse=True):
    acc += distriDictO[key]
    print "%2d: %6d, %5.1f %%, %5.1f %%"%(key, distriDictO[key], 100*distriDictO[key]/float(total_box_re), 100*acc/float(total_box_re)  )
print "\nenlarge scale ", multi_s, ":"
acc = 0
for key in sorted( distriDictS, reverse=True):
    acc += distriDictS[key]
    print "%2d: %6d, %5.1f %%, %5.1f %%"%(key, distriDictS[key], 100*distriDictS[key]/float(total_box_re), 100*acc/float(total_box_re)  )


plt.subplot(3,1,1)  # up
plt.hist(maxIOUs, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("ious with gt by detection")
plt.xlabel("iou with gt")

plt.subplot(3,1,2)  # middle
plt.hist(area_before, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("overlap with gt by detection")
plt.xlabel("re areas in gt before")

plt.subplot(3,1,3)  # down
plt.hist(area_afterS, bins=20, facecolor="blue", edgecolor="black", alpha=0.9)
plt.title("overlap with gt after enlarge %.2f"%(multi_s))
plt.xlabel("re areas in gt after enlarge %.2f"%(multi_s))
plt.show()
