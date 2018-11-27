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
CONF_THRESH = 0.9
NMS_THRESH = 0.01
wkdir = "/Users/momo/wkspace/caffe_space/testdadian/"
testlist = "five.txt"
fromDir = wkdir + testlist.split('.')[0] +'/'
print fromDir
scorelists = [ 0.95,       0.97,      0.99,      1.0]
toDirLists = ['ret_95/','ret_97/','ret_98/','ret_1/']
txtLists = [ open(wkdir + '/'+testlist.split('.')[0]+'90_ret_95.txt', 'w'), \
             open(wkdir + '/'+testlist.split('.')[0]+'95_ret_97.txt', 'w'), \
             open(wkdir + '/'+testlist.split('.')[0]+'97_ret_99.txt', 'w'), \
             open(wkdir + '/'+testlist.split('.')[0]+'99_ret_1.txt', 'w')  ]
for toDir in toDirLists:
    if not os.path.isdir(wkdir+toDir):
        os.makedirs(wkdir+toDir)
    # txtLists[i] = open(toDir.split('/')[0] + testlist.split('.')[0] + '.txt', 'w')

readFile = open(wkdir +testlist, "r")
def speical_config():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
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
                
                handcls = np.append( handcls, CLASSES[cls_ind] )
                handbox = np.append( handbox, [ dets[i][0], dets[i][1], dets[i][2], dets[i][3] ] )
                handscore = np.append( handscore, [dets[i][4]] )
                cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (0, 255, 0), 4)
                cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

        handcls = np.reshape(handcls, (-1,1))
        handbox = np.reshape(handbox, (-1, 4))
        handscore = np.reshape(handscore, (-1,1))
        # print dets.shape[0], " handcls:", handcls, "handbox:\n", handbox, "score:", handscore
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
    prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/180X320/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
    caffemodel = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/180X320/alimmcv5stride16bn_iter_530000.caffemodel"
    filelists = readFile.readlines()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network: {:s}'.format(caffemodel)
    # writeFile = open(toDir+"ret.txt", "w")
    numFrame = 0
    n_hand_gt = 0
    n_false_re = 0
    handcls_gt = np.array([])
    box_gt = np.array([])
    for filename in filelists:
        picName = filename.strip().split('\n')[0]+'.jpg'
        frame = cv2.imread(fromDir+picName)
        try:
            frame.shape
        except:
            print fromDir+picName
            continue
        # cv2.imshow("img", frame)
        # cv2.waitKey(1)
        numFrame+=1
        handcls, handbox, handscore  = demo(net, frame)
        nhand = handbox.shape[0]
        n_false_re += nhand    
        for i in xrange(0, nhand):
            for th_score in sorted( scorelists) :
                if handscore[i][0] <= th_score:
                    idx = scorelists.index(th_score)
                    cv2.imwrite(wkdir + toDirLists[idx]+picName, frame)
                    txtLists[idx].write(picName + ' ' + str(nhand) )
                    for ng in xrange(nhand):
                        txtLists[idx].write(' %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]) )
                    txtLists[idx].write('\n' )
                    break
        if numFrame%100 == 0:
            print numFrame, "faults=%d"%(n_false_re)

    