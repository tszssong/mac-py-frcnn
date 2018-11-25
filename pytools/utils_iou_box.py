# -*- coding: utf-8 -*-
import numpy as np
import copy
import numpy.random as npr
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2
def IOU(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio
def REarea_inGT(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height + 1
        Area2 = width2 * height2 + 1
        ratio = Area * 1. / Area2  #重叠面积/gt面积，表示re在gt里的部分
    return ratio

def IOU_multi(box, boxes):
    maxIou = 0.0
    maxIdx = 0
    for box_idx in xrange(boxes.shape[0]):
        compBox = boxes[box_idx]
        iou = IOU(box, compBox)
        if iou > maxIou:
            maxIou = iou
            maxIdx = box_idx
    return maxIou, maxIdx

def containBox(box, boxes):
    contain = False
    for box_idx in xrange(boxes.shape[0]):

        compBox = boxes[box_idx]
        compW = compBox[2] - compBox[0]
        compH = compBox[3] - compBox[1]
        if( box[0]<compBox[0]-compW/10 and box[1]<compBox[1]-compH/10 and box[2]>compBox[2]+compW/10 and box[3]< compBox[3]+compH/10):
            contain = True
    return contain

def overlapSelf(Reframe,GTframe):
    """Compute overlap between detect box and gt boxes

        """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        ratio = Area * 1. / Area1
    return ratio

def overlapingOtherBox(crop_box, box_idx, f_boxes):

    overlap_flag = 0
    for otherbox_idx in xrange(f_boxes.shape[0]):
        if not box_idx == otherbox_idx:
            iou = IOU(crop_box, f_boxes[otherbox_idx])
            if iou > 0.01:
                overlap_flag = 1
    if overlap_flag == 1:
        return True
#*************image******************************#
def crop_image(img, crop_box, param='black'):
    height, width, channel = img.shape
    right_x = 0
    left_x = 0
    top_y = 0
    down_y = 0
    
    nx1, ny1, nx2, ny2 = crop_box
    
    if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
        if nx2 > width:
            right_x = nx2 - width
        if ny2 > height:
            down_y = ny2 - height
        if nx1 < 0:
            left_x = 0 - nx1
        if ny1 < 0:
            top_y = 0 - ny1
    
        if param == 'black':
            black = [0, 0, 0]
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_CONSTANT,
                                          value=black);
        else:
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_REPLICATE);
    else:
        constant = copy.deepcopy(img)
    # constant
    return constant[int(ny1 + top_y):int(ny2 + top_y), int(nx1 + left_x):int(nx2 + left_x), :]

def crop_image_multi(img, box, enlargeScale=2.0, param='black'):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    
    nw = w*enlargeScale
    nh = nw
    
    nx1 = cx - nw / 2
    ny1 = cy - nh / 2
    nx2 = cx + nw / 2
    ny2 = cy + nh / 2
    
    height, width, channel = img.shape
    
    right_x = 0
    left_x = 0
    top_y = 0
    down_y = 0
    
    if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
        if nx2 > width:
            right_x = nx2 - width
        if ny2 > height:
            down_y = ny2 - height
        if nx1 < 0:
            left_x = 0 - nx1
        if ny1 < 0:
            top_y = 0 - ny1
        
        if param == 'black':
            black = [0, 0, 0]
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_CONSTANT,
                                          value=black);
        else:
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_REPLICATE);
    else:
        constant = copy.deepcopy(img)
    # constant
    return constant[int(ny1 + top_y):int(ny2 + top_y), int(nx1 + left_x):int(nx2 + left_x), :]
#********************box***********************************#
def validBox(box, width, height):
    rx1, ry1, rx2, ry2 = box
    if rx1 >= rx2 or ry1 >= ry2:
        return False
    rw = rx2 - rx1
    rh = ry2 - ry1
    rcx = rx1 + rw / 2
    rcy = ry1 + rh / 2
    
    if max(rw, rh) < 60 or rx1 < 0 or ry1 < 0:
        return False
    
    if rx1 > rx2 or ry1 > ry2:
        print ":", x1, y1, x2, y2, "-", width, height
    if rx2 > width or ry2 > height or rx1 < 0 or ry1 < 0:
        return False
    
    return True
# gt_outside: pix allowed to go outside ground truth box
# p_ratio: False - crop a square ; True - crop a reactage
def compBox(box_re, box_gt, enlargeScale, gt_outside=10):
    # TODO: validBox and enlarge params
    # newBox = np.array([])
    
    x1, y1, x2, y2 = box_re
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    
    nw = w*enlargeScale
    nh = nw
    
    nx1 = cx - nw / 2
    ny1 = cy - nh / 2
    nx2 = cx + nw / 2
    ny2 = cy + nh / 2

    nbox_re = np.array([nx1, ny1, nx2, ny2])
#    iou_before = IOU(box_re, box_gt)
#    iou_after = IOU(nbox_re, box_gt)
#    return iou_before, iou_after
    aera_before = REarea_inGT(box_re, box_gt)
    aera_after = REarea_inGT(nbox_re, box_gt)

    return aera_before, aera_after, nbox_re
#***************************gt re****************************#
def get_re_from_line(line):
    nhand_re = int(line.split(' ')[1])
    handcls_re = np.array([])  # line.split(' ')[2]
    box_re = np.array([])
    for i in xrange(nhand_re):
        handcls_re = np.append(handcls_re, line.split(' ')[i * 5 + 2])
        box_re = np.append(box_re, np.array([int(line.split(' ')[i * 5 + 3]), \
                                             int(line.split(' ')[i * 5 + 4]), \
                                             int(line.split(' ')[i * 5 + 5]), \
                                             int(line.split(' ')[i * 5 + 6])]))
    handcls_re = np.reshape(handcls_re, (-1, 1))
    box_re = np.reshape(box_re, (-1, 4))
    return nhand_re, handcls_re, box_re
def get_gt_from_xml(xml_path):
    tree = ET.parse(xml_path)  # 打开xml文档
    root = tree.getroot()  # 获得root节点
    filename = root.find('filename').text

    nhand_in_gt = 0
    handcls_gt = np.array([])
    box_gt = np.array([])
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        nhand_in_gt += 1

        hand_name = object.find('name').text
        handcls_gt = np.append(handcls_gt, hand_name)
        # print hand_name
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
        box_gt = np.append(box_gt, np.array([xmin, ymin, xmax, ymax]))
    handcls_gt = np.reshape( handcls_gt, (-1, 1) )
    box_gt = np.reshape( box_gt, (-1, 4) )
    return nhand_in_gt, handcls_gt, box_gt
