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
from utils_iou_box import IOU_multi, get_gt_from_xml, crop_image_multi, get_re_from_line
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
gesturelists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart']
tp_dict = {}
gt_dict = {}
re_dict = {}
for gname in gesturelists:
    tp_dict[gname] = 0
    gt_dict[gname] = 0
    re_dict[gname] = 0

iouP = 0.3
wkdir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/"
det_prototxt = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/180X320/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
det_model = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/180X320/iter115_big6_iter_1420000.caffemodel"

cls_prototxt = "/Users/momo/wkspace/caffe_space/caffe/models/from113/8clsNother_frcnnUse.prototxt"
cls_model = "/Users/momo/wkspace/caffe_space/caffe/models/from113/8clsNother10w_iter_1000000.caffemodel"
cls_multi_ = 2.0
clsInputSize_ = 64
cls_mean_ = np.array([104, 117, 123])
video_path = "/Volumes/song/testVideos/test4neg/momoLive/"
testTxt = video_path + "../momoLive4neg.txt"

toBaseDir = wkdir + "output/"+det_model.split('.')[0].split('/')[-1] + '+' + cls_model.split('.')[0].split('/')[-1] + "_results/"
if not os.path.isdir(toBaseDir):
    os.makedirs(toBaseDir)
detPicDir = toBaseDir + "det_neg_Pics/"
if not os.path.isdir(detPicDir):
    os.makedirs(detPicDir)
clsPicDir = toBaseDir + "cls_neg_Pics/"
if not os.path.isdir(clsPicDir):
    os.makedirs(clsPicDir)
writeTxt = toBaseDir + "/neg_" + det_model.split('.')[0].split('/')[-1] + '+' + cls_model.split('.')[0].split('/')[-1] + ".txt"
def special_config():
    cfg.TEST.HAS_RPN = True     # Use RPN for proposals
    caffe.set_mode_cpu()
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
    det_net = caffe.Net(det_prototxt, det_model, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(det_model)
    
    classify_net = caffe.Net(cls_prototxt, cls_model, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(cls_model)

    writeFile = open(writeTxt, "w")
    readFile = open(testTxt, 'r')
    filelists = readFile.readlines()
    numFrame = 0
    n_pos_gt = 0
    n_pos_re = 0
    n_pos_tp = 0
    for filename in filelists:
        video_name = filename.split()[0]
        video = cv2.VideoCapture(video_path + video_name + '.mp4')
        success, frame = video.read()
        numFrame = 0
        while success:
            cv2.imshow("capture", frame)
            cv2.waitKey(1)
            numFrame+=1
            picName =  filename.split()[0] + '_f' + str(numFrame) + '.jpg'
#            nhand_gt, handcls_gt, box_gt = get_re_from_line(filename)
#            print nhand_gt, handcls_gt, box_gt
#            for i in xrange(handcls_gt.shape[0]):
#                gt_dict[ handcls_gt[i][0] ] += nhand_gt
#            n_pos_gt += nhand_gt

            frame_copy = frame.copy()
            handcls, handbox, handscore = demo(det_net, frame)
            nhand = handbox.shape[0]
            n_pos_re += nhand
            nTP = 0
            if nhand > 0:
    #            cv2.imwrite(savePicDir+picName, frame)
                writeFile.write(picName + ' ' + str(nhand))
                for i in xrange(0, nhand):
                    cropped_im = crop_image_multi(frame_copy, handbox[i], cls_multi_)
    #                cropped_im = crop_image(frame_copy, handbox[i])
    #                cv2.imshow("small", cropped_im)
    #                cv2.waitKey(1)
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
                    if not gesturelists[cls] == handcls_gt:
                        print cls,handcls_gt
                        cv2.imwrite(clsPicDir + str(cls)+'-' + picName, cls_img_copy)
                    else:
                        tp_dict[ gesturelists[cls] ] += 1
                    
                    writeFile.write(' '+ gesturelists[cls] )
                
                    writeFile.write(' %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
                writeFile.write("\n")
                if ( IOU_multi(handbox[i],box_gt) > iouP):
                    nTP += 1
            n_pos_tp += nTP
            if numFrame%100 == 0:
                print numFrame, "pics: gt=%d, re=%d, tp=%d"%(n_pos_gt, n_pos_re, n_pos_tp)
            success, frame = video.read()


