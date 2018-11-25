# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils_iou_box import get_re_from_line, get_gt_from_xml, compBox, IOU_multi, crop_image

def ansDetResults(relists, xml_path, pic_path, save_path_root, multi_s, save_flag = False):
    distriDictO = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    distriDictS = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

    maxIOUs = np.array([])
    area_before = np.array([])
    area_afterS = np.array([])
    area_afterB = np.array([])
    total_box_re = 0
    for line in relists:
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
            b, a, crop_box = compBox(boxes_re[ rebox_idx ], boxes_gt[gtbox_idx], multi_s)
            area_before = np.append(area_before, b)
            area_afterS = np.append(area_afterS, a)
            r_area = int(b*10)                  #python int same as floor: 1 = [1.0,1.999999~]
            if(distriDictO.has_key(r_area)):
                distriDictO[r_area] += 1
                if save_flag:
                    im = cv2.imread(pic_path+picname+'.jpg')
                    savepath = save_path_root+'/det_' + str(r_area) + '/'
                    if not os.path.isdir(savepath):
                        os.mkdir(savepath)
                    cropped_im = crop_image(im, boxes_re[ rebox_idx ])
                    cv2.imwrite(savepath + picname + '.jpg', cropped_im)
        
            r_area = int(a*10)                  #python int same as floor: 1 = [1.0,1.999999~]
            if(distriDictS.has_key(r_area)):
                distriDictS[r_area] += 1
                if save_flag:
                    im = cv2.imread(pic_path+picname+'.jpg')
                    savepath = save_path_root+'/enlarge_' + str(multi_s)+'_'+ str(r_area)  + '/'
                    if not os.path.isdir(savepath):
                        os.mkdir(savepath)
                    cropped_im = crop_image(im, crop_box)
                    cv2.imwrite(savepath + picname + '.jpg', cropped_im)

    return total_box_re, distriDictO, distriDictS


