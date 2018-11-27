# -*- coding: utf-8 -*-
import sys
import os
iouP = 0.3
#计算IOU
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

gesturelist = ['heart','yearh','one','baoquan','five','bainian','zan','fingerheart','ok','call','rock','big_v','fist']

gt_path = "gt/gt928_1257.txt"
re_path = "ret/928_2cls_hand.txt"
totalGT = 0
totalRE = 0
totalTP = 0
for gname in gesturelist:
    re_dict = {}
    re_list = []
    gt_dict = {}
    gt_list = []
    f = open(re_path)
    while True:
        line = f.readline()
        if not line:
            break
        if (' hand ') in line:
            tmp = line.strip(' \n').split(' ')
            re_list.append(tmp[0])
            re_dict[tmp[0]] = []
            re_dict[tmp[0]].append(int(tmp[1]))  #gesture数量
            i = 2
            while i<len(tmp):
                if tmp[i] == 'hand':
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                else:
                    i = i+1
    f.close()
    ##############re hands#################
    nre_hand = 0
    for i in re_list:
        nre_hand = nre_hand + re_dict.get(i)[0] #box
    totalRE = totalRE + nre_hand                #总box
    #########################################
    f = open(gt_path)
    while True:
        line = f.readline()
        if not line:
            break
        if  (' '+gname+' ')  in line:
            tmp = line.strip(' \n').split(' ')
            gt_list.append(tmp[0])
            gt_dict[tmp[0]] = []
            gt_dict[tmp[0]].append(int(tmp[1]))
            i = 2
            while i<len(tmp):
                if tmp[i] == gname:
                    i = i+1
                    gt_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    gt_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    gt_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    gt_dict[tmp[0]].append( int(tmp[i]) )
                else:
                    i = i+1
    f.close()
    ##############gt hands#################
    ngt_hand = 0
    for i in gt_list:
        ngt_hand = ngt_hand + gt_dict.get(i)[0]
    totalGT = totalGT + ngt_hand
    ###############re hands in gt########################
    hand_regt = 0
    for i in re_list:
        re_rect = re_dict.get(i)
        if(gt_dict.has_key(i)):
            gt_rect = gt_dict.get(i)
        else:
            continue
        n_re = re_rect[0]
        n_gt = gt_rect[0]
        count = 0
        false = 0
        for j in range(n_re):
            hand_re = re_rect[1 + j * 4:1 + (j + 1) * 4]
            for m in range(n_gt):
                hand_gt = gt_rect[1 + m * 4:1 + (m + 1) * 4]
                ratio = IOU(hand_re, hand_gt)
                if ratio >= iouP:
                    count += 1
        hand_regt = hand_regt + count
    totalTP = totalTP + hand_regt
    print "******** %s"%(gname),"********"
    print " TP= %4d"%(hand_regt)," RE= %4d"%(nre_hand)," GT= %4d"%(ngt_hand)
    print " precison= %6.2f"%(float(hand_regt)*100/nre_hand)," recall= %6.2f"%(float(hand_regt)*100/ngt_hand)
print " total TP= %4d"%(totalTP)," total RE= %4d"%(totalRE)," total GT= %4d"%(totalGT)
print " total precision= %6.2f"%(float(totalTP)*100/totalRE)," total recall= %6.2f"%(float(totalTP)*100/totalGT)
print "iou:", iouP
