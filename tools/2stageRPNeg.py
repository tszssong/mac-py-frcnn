# -*- coding: utf-8 -*-
#!/usr/bin/env python
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from iouutils import IOU_multi
import numpy as np
import os, sys
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
iouP = 0.3
fromDir = "/Volumes/song/handg_neg_test32G/0627Pic/627PicTest/JPEGImages/"
toDir = "/Volumes/song/handg_neg_test32G/0627Pic/ret/retPic/"
writeFile = "/Volumes/song/handg_neg_test32G/0627Pic/ret/ret.txt"

#返回手势类别、坐标、分数，每行对应同一个手势
def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    scores, boxes = im_detect(net, im)
    CONF_THRESH = 0.7
    NMS_THRESH = 0.01
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
        for i in xrange(dets.shape[0]):
            if (dets[i][4] > CONF_THRESH):
                handcls = np.append( handcls, CLASSES[cls_ind] )
                handbox = np.append( handbox, [ dets[i][0], dets[i][1], dets[i][2], dets[i][3] ] )
                handscore = np.append( handscore, [dets[i][4]] )
                cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (0, 255, 0), 4)
                cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        handcls = np.reshape(handcls, (-1,1))
        handbox = np.reshape(handbox, (-1, 4))
        handscore = np.reshape(handscore, (-1,1))
        return handcls, handbox, handscore

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True     # Use RPN for proposals
    prototxt = "/Users/momo/Desktop/gesture/from113/MMCV5_stride16/test.prototxt"
    caffemodel = "/Users/momo/Desktop/sdk/momocv2_model/original_model/object_detect/mmcv5stride16_iter_5250000.caffemodel"
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    cfg.TEST.SCALES = [144,]
    cfg.TEST.MAX_SIZE = 256
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10

    if not os.path.isdir(toDir):
        os.makedirs(toDir)
    writeFile = open(toDir+"/../ret.txt", "w")
    readFile = open(fromDir + "../0627_5_1258.txt", "r")
    filelists = readFile.readlines()
    numFrame = 0
    n_pos_gt = 0
    n_pos_re = 0
    n_pos_tp = 0
    for filename in filelists:
        picName = filename.split(' ')[0].split('.')[0]+'.jpg'
        nhand_gt = int(filename.split(' ')[1])
        n_pos_gt += nhand_gt

        handcls_gt = np.array([])  #filename.split(' ')[2]

        box_gt = np.array([])
        for i in xrange(nhand_gt):
            handcls_gt = np.append(handcls_gt, filename.split(' ')[i*5 + 2] )
            box_gt = np.append(box_gt, np.array( [int( filename.split(' ')[i*5 + 3] ) , \
                                                  int( filename.split(' ')[i*5 + 4] ), \
                                                  int( filename.split(' ')[i*5 + 5] ), \
                                                  int( filename.split(' ')[i*5 + 6] ) ]  ))
        handcls_gt = np.reshape(handcls_gt, (-1,1))
        box_gt = np.reshape(box_gt, (-1,4))
        # print "nhand gt:", nhand_gt, handcls_gt, box_gt
        numFrame+=1
        frame = cv2.imread(fromDir+picName)
        handcls, handbox, handscore = demo(net, frame)
        nhand = handbox.shape[0]
        n_pos_re += nhand
        nTP = 0
        if nhand > 0:
            cv2.imwrite(toDir+picName, frame)
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
