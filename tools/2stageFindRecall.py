# -*- coding: utf-8 -*-
#!/usr/bin/env python
import _init_paths

import os, sys
os.environ['GLOG_minloglevel'] = '3'
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from iouutils import IOU_multi
import numpy as np
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
caffe.set_mode_cpu()
CLASSES = ('__background__','hand')
iouP = 0.3
CONF_THRESH = 0.9
NMS_THRESH = 0.01
fromDir = "/Volumes/song/handg_neg_test32G/0627Pic/627PicTest/JPEGImages/"
toDir = "/Volumes/song/handg_neg_test32G/0627Pic/ret/retPic/"
testlist = "0627_5_10.txt"
readFile = open(fromDir + "../0627_5_1258.txt", "r")
readFile = open(fromDir + "../" +testlist, "r")
def speical_config():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = [144,]
    cfg.TEST.MAX_SIZE = 256
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
    return handcls, handbox, handscore
def get_gt(filename):

    nhand_gt = int(filename.split(' ')[1])

    handcls_gt = np.array([])  # filename.split(' ')[2]

    box_gt = np.array([])
    for i in xrange(nhand_gt):
        handcls_gt = np.append(handcls_gt, filename.split(' ')[i * 5 + 2])
        box_gt = np.append(box_gt, np.array([int(filename.split(' ')[i * 5 + 3]), \
                                             int(filename.split(' ')[i * 5 + 4]), \
                                             int(filename.split(' ')[i * 5 + 5]), \
                                             int(filename.split(' ')[i * 5 + 6])]))
    handcls_gt = np.reshape(handcls_gt, (-1, 1))
    box_gt = np.reshape(box_gt, (-1, 4))
    return nhand_gt, handcls_gt, box_gt

if __name__ == '__main__':
    speical_config()
    if not os.path.isdir(toDir):
        os.makedirs(toDir)
    prototxt = "/Users/momo/Desktop/gesture/from113/MMCV5_stride16/test.prototxt"
    modelPath = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/"
    filelists = readFile.readlines()
    modelprefix = "mmcv5s16f66w_iter_"
    recallSet = np.array([], dtype=np.float32)
    precSet = np.array([], dtype=np.float32)
    totalResultsFile = open("/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/Recall/models/totalResults.txt", "w")
    totalResultsFile.write("\ntesting "+modelprefix + "s with iou= " + str(iouP) +"CONF_THRESH= "+str(CONF_THRESH) )
    totalResultsFile.write( "\npre nms:"+ str(cfg.TEST.RPN_PRE_NMS_TOP_N) )
    totalResultsFile.write( "\npost nms:"+ str(cfg.TEST.RPN_POST_NMS_TOP_N) )
    totalResultsFile.write("\ntest lists:"+testlist )
    for model_idx in xrange(1,5):
        caffemodel = modelprefix +str(model_idx)+"0000.caffemodel"
        writeFile = open("/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/Recall/models/ret"+ caffemodel.split('.')[0]+".txt", "w")
        net = caffe.Net(prototxt, modelPath+caffemodel, caffe.TEST)
        # print '\n\nLoaded network: {:s}'.format(caffemodel)
        numFrame = 0
        n_pos_gt = 0
        n_pos_re = 0
        n_pos_tp = 0
        for filename in filelists:
            picName = filename.split(' ')[0].split('.')[0]+'.jpg'
            nhand_gt, handcls_gt, box_gt = get_gt( filename )
            n_pos_gt += nhand_gt
            # print "nhand gt:", nhand_gt, handcls_gt, box_gt
            frame = cv2.imread(fromDir+picName)
            numFrame+=1
            handcls, handbox, handscore = demo(net, frame)
            nhand = handbox.shape[0]
            n_pos_re += nhand
            nTP = 0
            if nhand > 0:
                # cv2.imwrite(toDir+picName, frame)
                writeFile.write(picName + ' ' + str(nhand))
                for i in xrange(0, nhand):
                    # writeFile.write(' hand %d %d %d %d'%(handbox[i*4],handbox[i*4+1],handbox[i*4+2],handbox[i*4+3]))
                    writeFile.write(' hand %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
                    if (IOU_multi(handbox[i], box_gt) > iouP):
                        nTP += 1
                writeFile.write("\n")
                # cv2.imshow("capture", frame)
                # cv2.waitKey(1)
            n_pos_tp += nTP
            # if numFrame%100 == 0:
            #     print numFrame, "pics: gt=%d, re=%d, tp=%d"%(n_pos_gt, n_pos_re, n_pos_tp)
        modelrecall = float(n_pos_tp) / float(n_pos_gt)
        recallSet=np.append(recallSet, modelrecall)
        modelprec = float(n_pos_tp) / float(n_pos_re)
        precSet = np.append(precSet, modelprec)

        # print model_idx, "- gt:", n_pos_gt, "re:", n_pos_re, "tp:", n_pos_tp, "precision:" , float(n_pos_tp)/float(n_pos_re), "recall: ", float(n_pos_tp)/float(n_pos_gt)
        print "model%3dw: gt =%4d, re =%4d, tp =%4d, prec=%.2f, recall=%.2f"%(model_idx, n_pos_gt,n_pos_re,  n_pos_tp, \
                                                                    float(n_pos_tp) / float(n_pos_re), float(n_pos_tp) / float(n_pos_gt))
        totalResultsFile.write("\nmodel%3dw: gt =%4d, re =%4d, tp =%4d, prec=%.2f, recall=%.2f"%(model_idx, n_pos_gt,n_pos_re,  n_pos_tp, \
                                                                    float(n_pos_tp) / float(n_pos_re), float(n_pos_tp) / float(n_pos_gt)) )
        writeFile.close()
    print "best precision %.2f, at %d"%( np.max(precSet), np.where(precSet==max(precSet)) )
    print "best recall %.2f, at %d"%( np.max(recallSet), np.where(recallSet==max(recallSet)) )
    print np.mean(recallSet)
    totalResultsFile.write( "\nbest recall: %.2f"%np.max(recallSet) )
    maxlist = np.where( recallSet==max(recallSet) )
    for i in xrange(maxlist[0].shape[0]):
        totalResultsFile.write("\nbest recall idx: %d" % maxlist[0][i])
    readFile.close()