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
from utils_iou_box import IOU_multi, get_gt_from_xml
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
CONF_THRESH = 0.7
NMS_THRESH = 0.01
iouP = 0.3
print "CONF_TH:", CONF_THRESH, "IOU:", iouP
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/180X320/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
modelPath = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/180X320/"
modelName = "iter115_big6_iter_1420000.caffemodel"
xml_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/skip627928/skip_627_3k+928_1k_resize240_xml/"
pic_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/skip627928/skip_627_3k+928_1k_resize240/"
testTxt = pic_path + "../test.txt"
toBaseDir = wkdir + "output/"+modelName.split('.')[0]+"_results/"
if not os.path.isdir(toBaseDir):
    os.makedirs(toBaseDir)
savePicDir = toBaseDir + "retPics/"
if not os.path.isdir(savePicDir):
    os.makedirs(savePicDir)
writeTxt = toBaseDir + "/Recall.txt"
def special_config():
    cfg.TEST.HAS_RPN = True     # Use RPN for proposals
    caffe.set_mode_cpu()
    cfg.TEST.SCALES = [180,]
    cfg.TEST.MAX_SIZE = 320
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10
#返回手势类别、坐标、分数，每行对应同一个手势
def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    scores, boxes = im_detect(net, im)
    handcls = np.array([])
    handbox = np.array([])
    handscore = np.array([])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1                                            # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]     # 300*4矩阵
        cls_scores = scores[:, cls_ind]                         # 300行
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # if dets[0, -1] > CONF_THRESH:
        #     print dets[0]
        for i in xrange(dets.shape[0]):
            if (dets[i][4] > CONF_THRESH):
                # nhand += 1
                handcls = np.append( handcls, CLASSES[cls_ind] )
                handbox = np.append( handbox, [ dets[i][0], dets[i][1], dets[i][2], dets[i][3] ] )
                handscore = np.append( handscore, [dets[i][4]] )
                cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (0, 255, 0), 4)
                cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        handcls = np.reshape(handcls, (-1,1))
        handbox = np.reshape(handbox, (-1, 4))
        handscore = np.reshape(handscore, (-1,1))
        # print dets.shape[0], " handcls:", handcls, "handbox:\n", handbox, "score:", handscore
        return handcls, handbox, handscore

if __name__ == '__main__':
    special_config()
    caffemodel = modelPath + modelName
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    writeFile = open(writeTxt, "w")
    readFile = open(testTxt, 'r')
    filelists = readFile.readlines()
    numFrame = 0
    n_pos_gt = 0
    n_pos_re = 0
    n_pos_tp = 0
    for filename in filelists:
        picName =  filename.strip() + '.jpg'
        xmlName = xml_path + filename.strip()+'.xml'
        nhand_gt, handcls_gt, box_gt = get_gt_from_xml(xmlName)
        n_pos_gt += nhand_gt
#        print "nhand gt:", nhand_gt, handcls_gt, box_gt
        numFrame+=1
        frame = cv2.imread(pic_path+picName)
        handcls, handbox, handscore = demo(net, frame)
        nhand = handbox.shape[0]
        n_pos_re += nhand
        nTP = 0
        if nhand > 0:
            cv2.imwrite(savePicDir+picName, frame)
            writeFile.write(picName + ' ' + str(nhand))
            for i in xrange(0, nhand):
                writeFile.write(' hand %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
            writeFile.write("\n")
            # cv2.imshow("capture", frame)
            # cv2.waitKey(1)
            if ( IOU_multi(handbox[i],box_gt) > iouP):
                nTP += 1
        n_pos_tp += nTP
        if numFrame%100 == 0:
            print numFrame, "pics: gt=%d, re=%d, tp=%d"%(n_pos_gt, n_pos_re, n_pos_tp)

print "gt:", n_pos_gt, "re:", n_pos_re, "tp:", n_pos_tp
print "recall: ", float(n_pos_tp)/float(n_pos_gt)
print "precision:" , float(n_pos_tp)/float(n_pos_re)
