# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import cv2
DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
#        anchor_scales = layer_params.get( 'scales', (4, 8, 16) )
#         anchor_scales = layer_params.get( 'scales', (2, 4, 8) )
        anchor_scales = layer_params.get( 'scales', (8, 16, 32) )
#        print "anchor_scales:",anchor_scales
        self._anchors = generate_anchors(base_size=4, scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        timerProposalLayer = Timer()
        timerProposalLayer.tic()
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
#        print "conv5_3 layer shape:", bottom[0].data.shape
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'
        ###################img input#####################

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
#        print "scores.shape:", scores.shape
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
#        print "rpn size",height, width

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
#        print "box_ds.shape:", bbox_deltas.shape, bbox_deltas[0]

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))


        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
#        print "proposals.shape:", proposals.shape, proposals[0]
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        # print "pre nms protosals:", proposals.shape, proposals[0]
        # print "post nms proposals:", proposals.shape, proposals[0]
        #        for i in range(proposals.shape[0]):
        # for i in range(min(20, len(order))):
        #     cv2.rectangle(pre_display, (proposals[i][0]/im_info[2], proposals[i][1]/im_info[2]), (proposals[i][2]/im_info[2],proposals[i][3]/im_info[2]), (0, 255, 0),1, 4, 0)
        #     cv2.putText(pre_display,str(i),(int(proposals[i][0]/im_info[2]), int(proposals[i][1]/im_info[2])), \
        #     cv2.FONT_HERSHEY_DUPLEX,0.8, (0, 255, 0),1, 4)
            #pre_display = cv2.resize(pre_display, (pre_display.shape[0]/ 2, pre_display.shape[1]/ 2))

            # cv2.imshow("pre proposals", pre_display);
            # cv2.waitKey(1)
        # cv2.imwrite("/Users/momo/Desktop/gestureOut/pyTest/pre_anchor/"+saveName, pre_display)
#         6. apply nms (e.g. threshold = 0.7)
#         7. take after_nms_topN (e.g. 300)
#         8. return the top proposals (-> RoIs top)

#        timerNMS = Timer()
#        timerNMS.tic()
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
#        timerNMS.toc()
#        print ('NMS took {:.6f}s for '
#               'pre = {:d} pos = {:d} th = {:.2f} ').format(timerNMS.total_time, pre_nms_topN, post_nms_topN, nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        # print "bolb shape:", blob.shape, blob[0]
        # for i in range(blob.shape[0]):
        #     print 'Proposal {:d}, {:.0f}[{:5.1f} {:5.1f} {:5.1f} {:5.1f}]'.format(i, blob[i][0], blob[i][1], blob[i][2], blob[i][3],blob[i][4])
        #{'0':red, '1': blue, '2':yello, '3':pink, '4':acid blue}
        # color = {'0':(0, 0, 255), '1':(255, 0, 0), '2':(0, 255, 255), '3': (255, 0, 255),  '4':(255, 255, 0)}
        # size = {'0':4, '1':2, '2':1, '3': 1,  '4':1}
#        for i in range(blob.shape[0]):
#        print len(keep)
#         for i in range( min(20, len(keep)) ):
#             cv2.rectangle(pos_display, (blob[i][1]/im_info[2], blob[i][2]/im_info[2]), (blob[i][3]/im_info[2],blob[i][4]/im_info[2]), color.get(str(i%5)), 2, 4, 0)
#             cv2.rectangle(pos_display, (blob[i][1]/im_info[2], blob[i][2]/im_info[2]), (blob[i][3]/im_info[2],blob[i][4]/im_info[2]), color.get(str(i/5)), size.get(str(i/5)), 4, 0)
#             cv2.putText(pos_display,str(i), (int(proposals[i][0]/im_info[2]), int(proposals[i][1]/im_info[2])), cv2.FONT_HERSHEY_DUPLEX,0.8, color.get(str(i/5)),1, 4)
#             #pos_display = cv2.resize(pos_display, (pos_display.shape[0]/2, pos_display.shape[1]/2))
#             cv2.imshow("post proposals", pos_display);
#             cv2.waitKey(1)
#         cv2.imwrite("/Users/momo/Desktop/gestureOut/pyTest/pos_anchor/"+saveName, pos_display)
        #############################
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
#        print "top data shape:", top[0].data.shape, top[0].data[0]
        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores
        
        timerProposalLayer.toc()
#        print ('Proposcal Layer took {:.6f}s ').format(timerProposalLayer.total_time)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
