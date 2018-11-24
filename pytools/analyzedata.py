# -*- coding: utf-8 -*-
import sys
import os
import re
import cv2
import random
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import numpy as np
import math
import matplotlib.pyplot as plt
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
fileName = wkdir + '/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
xml_path = "../data/VOCdevkit2007/VOC2007/Annotations/"
pic_path = "../data/VOCdevkit2007/VOC2007/JPEGImages/"

n_box = 0

distribute = np.array([])
distriDict = {}
for line in open(fileName, 'r'):
    xml_name = line.strip('\n') + ".xml"
    tree = ET.parse(xml_path+"/"+xml_name)  # 打开xml文档
    root = tree.getroot()  # 获得root节点
    filename = root.find('filename').text
    

    size = root.find('size')
    oriW = int(size.find('width').text)
    oriH = int(size.find('height').text)
#    print filename
#    im = cv2.imread(pic_path+filename)
#    if not (oriH - im.shape[0]<2 and oriW - im.shape[1] <2):
#        print filename, "annotaiton size error:",im.shape, oriH, oriW

    n_gesture_per_img = 0
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        n_gesture_per_img += 1
        n_box+=1
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        if xmin > xmax:
            tmp = xmin
            xmin = xmax
            xmax = tmp
            # print "xml x anno err: ", filename, xmin, xmax
        if ymin > ymax:
            tmp = ymin
            ymin = ymax
            ymax = tmp
            # print "xml y anno err: ", filename, ymin, ymax
        # print xmin, ymin, xmax, ymax, oriH, oriW, np.min([oriW, oriH])
        box_min_side = np.min([xmax-xmin, ymax - ymin])
        im_min_side = np.min([oriW, oriH])
        if xmax-xmin < 4 or ymax-ymin<4:
            print "xml size err: ", filename, xmax, xmin, ymax, ymin
            continue

        scale = float(im_min_side)/(box_min_side)
        distribute = np.append(distribute, scale)
        scale_int = round(scale)

        if not distriDict.has_key(scale_int):
            distriDict[scale_int] = 0
        distriDict[scale_int] += 1

print distriDict
for key in distriDict:
    print ("%3d: %4d, %.3f")%( key, distriDict[key], distriDict[key]/float(n_box) )
print "2 means box makes up 1/2 of shortSide [1.5, 2.5]"
#plt.bar(range(len(distriDict)), distriDict.values())
plt.hist(distribute, bins=10, facecolor="blue", edgecolor="black", alpha=0.7)
plt.show()
