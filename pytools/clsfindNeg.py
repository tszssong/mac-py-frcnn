# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os, sys
os.environ['GLOG_minloglevel'] = '3'
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import copy
from utils_iou_box import IOU_multi, get_gt_from_xml, crop_image_multi
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2

CLASSES = ('__background__','hand')
gesturelists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart']
errDict = {}
tp_dict = {}
gt_dict = {}
re_dict = {}
for gname in gesturelists:
    tp_dict[gname] = 0
    gt_dict[gname] = 0
    re_dict[gname] = 0

wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"

cls_prototxt = "/Users/momo/wkspace/caffe_space/caffe/models/from113/8clsNother_frcnnUse.prototxt"
cls_model = "/Users/momo/wkspace/caffe_space/caffe/models/from113/8clsNother10w_iter_250000.caffemodel"
cls_multi_ = float(sys.argv[1])
clsInputSize_ = 64
cls_mean_ = np.array([104, 117, 123])
#xml_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/gzAndroidTest/Annotations/"
#pic_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/gzAndroidTest/JPEGImages/"
xml_path = "/Volumes/song/handgVGGali_27G/img_alifist-zan_rot270_zan-xml/"
pic_path = "/Volumes/song/handgVGGali_27G/img_alifist-zan_rot270_zan-img/"
testTxt = pic_path + "../test.txt"
testTxt = "/Users/momo/Downloads/gtName_img_alifist-zan_rot270_zan-xml.txt"
toBaseDir = wkdir + "output/"+cls_model.split('.')[0].split('/')[-1] + "_ERRORS7/"
if not os.path.isdir(toBaseDir):
    os.makedirs(toBaseDir)

for clsname in gesturelists:
    errDict[clsname] = []
    if not os.path.isdir(toBaseDir + clsname +'/'):
        os.makedirs(toBaseDir + clsname +'/')


if __name__ == '__main__':
   
    caffe.set_mode_cpu()
    classify_net = caffe.Net(cls_prototxt, cls_model, caffe.TEST)
    print 'Loaded network {:s}'.format(cls_model)

    readFile = open(testTxt, 'r')
    filelists = readFile.readlines()
    numFrame = 0
    n_error = 0
    n_pos_gt = 0
    n_pos_tp = 0
    for filename in filelists:
#        picName =  filename.strip() + '.jpg'
#        xmlName = xml_path + filename.strip()+'.xml'
        picName =  filename.strip().split('.')[0] + '.jpg'
        xmlName = xml_path + filename.strip().split('.')[0] +'.xml'
        nhand_gt, handcls_gt, box_gt = get_gt_from_xml(xmlName)
#        print nhand_gt, handcls_gt, box_gt
        for i in xrange(handcls_gt.shape[0]):
            gt_dict[ handcls_gt[i][0] ] += nhand_gt
        n_pos_gt += nhand_gt
        numFrame+=1
        frame = cv2.imread(pic_path+picName)
        frame_copy = frame.copy()
        
        if nhand_gt > 0:
            for i in xrange(0, nhand_gt):
                cropped_im = crop_image_multi(frame_copy, box_gt[i], cls_multi_)

                small_h,small_w,small_ch = cropped_im.shape
                if small_h!=clsInputSize_ or small_w!=clsInputSize_:
                    cls_img = cv2.resize(cropped_im,(int(clsInputSize_),int(clsInputSize_)))
                cls_img_copy = cls_img.copy()
                cls_img = cls_img.astype(np.int)
                cls_img -= cls_mean_
                cls_img = np.swapaxes(cls_img, 0, 2)
                cls_img = np.swapaxes(cls_img, 1, 2)
                classify_net.blobs['data'].reshape(1,3,clsInputSize_,clsInputSize_)
                classify_net.blobs['data'].data[...]=cls_img
                out_ = classify_net.forward()
                prob = out_['prob'][0]
                
                cls_prob = np.max(prob)
                cls = np.where(prob==np.max(prob))[0][0]
                re_dict[ gesturelists[cls] ] += 1
                if not gesturelists[cls] == handcls_gt[i][0]:
#                    print cls, gesturelists[cls], handcls_gt[i][0], picName
                    n_error+=1
                    errDict[handcls_gt[i][0]].append(picName)
                    cv2.imwrite(toBaseDir + handcls_gt[i][0] +'/' + str(cls)+'-' + picName, cls_img_copy)
                else:
                    tp_dict[ gesturelists[cls] ] += 1
                    n_pos_tp+=1
    
    
#        if numFrame%100 == 0:
#            print numFrame, "pics: gt=%d, tp=%d"%(n_pos_gt, n_pos_tp)

print n_error,"in", numFrame, "pics with ", n_pos_gt, "hands"
for gsturename in errDict:
    f = open(toBaseDir + gsturename + ".txt" , "w")
    for filename in errDict[gsturename]:
        f.write( filename + '\n' )
    f.close()
