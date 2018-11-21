# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os, sys
os.environ['GLOG_minloglevel'] = '3'
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import random
from iouutils import IOU_multi
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
CONF_THRESH = 0.7
NMS_THRESH = 0.01
def speical_config():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = [180,]
    cfg.TEST.MAX_SIZE = 320
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10

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

    prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/180X320/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
    caffemodel = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/180X320/models/alimmcv5stride16bn_iter_650000.caffemodel"
    speical_config()
    caffe.set_mode_cpu()
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    fromDir = "/Volumes/song/video4gesture/total20181120/"

    readFile = open(fromDir + "../total20181120.txt", "r")
    retfilename = caffemodel.split('.')[0].split('/')[-1] + '_' + str(CONF_THRESH).split('.')[-1]
    toDir = '/Users/momo/Desktop/pyNeg/' + retfilename +'_r/'
    toDirSmall = '/Users/momo/Desktop/pyNeg/' + retfilename +'_small_r/'
    print toDir
    if not os.path.isdir(toDir):
        os.makedirs(toDir)
    if not os.path.isdir(toDirSmall):
        os.makedirs(toDirSmall)
    writeFile = open(toDir + "/../" + retfilename + "_r.txt", "w")

    filelists = readFile.readlines()
    print filelists
    for filename in filelists:
        print "filename:", filename
        video_name = filename.split()[0]
        video = cv2.VideoCapture(fromDir + video_name)
        success, frame = video.read()
        numFrame = 0
        while success:

            numFrame += 1
            savename = filename.split()[0].split('.')[0] + '_f' + str(numFrame) + '_r.jpg'

            rot_d = random.randint(-180,180);
            rows, cols = frame.shape[:2]
            M = cv2.getRotationMatrix2D( (cols/2, rows/2), rot_d,1)
            frame = cv2.warpAffine( frame, M, (cols, rows) )
            copyOri = frame.copy()

            handcls, handbox, handscore = demo(net, frame)
            nhand = handbox.shape[0]
            nTP = 0
            picName = filename.split(' ')[0].split('.')[0] + '_f'+ str(numFrame)+'_r.jpg'
            if nhand > 0:
                cv2.imwrite(toDir+picName, copyOri)

                writeFile.write(picName + ' ' + str(nhand))
                for i in xrange(0, nhand):
                    roiImg = copyOri[ int(handbox[i][1]) : int(handbox[i][3]), int(handbox[i][0]):int(handbox[i][2]), : ]
                    roiImg = cv2.resize( roiImg,(128, 128))
                    cv2.imwrite(toDirSmall + picName.split('.')[0]+str(i)+'.jpg', roiImg)
                    writeFile.write(' %d %d %d %d' % (handbox[i][0], handbox[i][1], handbox[i][2], handbox[i][3]))
                writeFile.write("\n")
            # cv2.imshow("capture", frame)
            # cv2.waitKey(1)
            success, frame = video.read()

writeFile.close()
readFile.close()