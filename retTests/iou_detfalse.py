    # -*- coding: utf-8 -*-
import sys
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

gesturelist = ['hand']
#gesturelist = ['heart', 'yearh', 'one', 'bainian', 'five', 'baoquan', 'zan', 'fingerheart', 'ok', 'call', 'rock', 'big_v', 'fist']
re_path = "ret/" + sys.argv[1]
gt_path = "gt/gt_all169all.txt" #sys.argv[2]
fpic = "err/" + sys.argv[1]
totalGT = 0
totalGTpic = 0
totalRE = 0
totalREpic = 0
totalTP = 0
totalFP = 0
Fvideonum = 0
F_numv = []
oF_numv = []
gt_dict = {}
gt_list = []
gt_nolist = []
f = open(gt_path)
while True:
    line = f.readline()
    if not line:
        break
    tmp = line.strip(' \n').split(' ')
    if  (' '+'hand'+' ')  in line:
        gt_list.append(tmp[0])
        gt_dict[tmp[0]] = []
        gt_dict[tmp[0]].append(int(tmp[1]))
        i = 2
        while i<len(tmp):
            if tmp[i] == 'hand':
                i = i+1
                gt_dict[tmp[0]].append( int(tmp[i]) )
                i = i+1
                gt_dict[tmp[0]].append( int(tmp[i]) )
                i = i+1
                gt_dict[tmp[0]].append( int(tmp[i]) )
                i = i+1
                gt_dict[tmp[0]].append( int(tmp[i]) )
            else:
                i = i+ 1
    else:
        gt_nolist.append(tmp[0])
f.close()
##############gt hands#################
ngt_hand = 0
for i in gt_list:
    ngt_hand = ngt_hand+gt_dict.get(i)[0]
totalGT = totalGT + ngt_hand#box
totalGTpic = totalGTpic + len(gt_list) + len(gt_nolist)#pic
#########################################
for gname in gesturelist:
    re_dict = {}
    re_list = []
    f_list = []
    of_list = []
    f = open(re_path)
    while True:
        line = f.readline()
        if not line:
            break
        if (' '+gname+' ') in line:
            tmp=line.strip(' \n').split(' ')
            re_list.append(tmp[0])
            re_dict[tmp[0]] = []
            re_dict[tmp[0]].append(line.count(gname)) #数量
            i = 2
            while i<len(tmp):
                if tmp[i] == gname:
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                    i = i+1
                    re_dict[tmp[0]].append( int(tmp[i]) )
                else:
                    i = i+ 1
    f.close()
    ##############re hands#################
    nre_hand = 0
    for i in re_list:
        nre_hand = nre_hand + re_dict.get(i)[0]#各手势box
    totalRE = totalRE + nre_hand#总box数量
    totalREpic = len(re_list)#各手势pic数量
    ###############re hands in gt########################
    iouP = float(sys.argv[2])
#iouP = sys.argv[2]
    hand_regt = 0
    hand_fault = 0
#    ratio_list = []
    g = open(fpic,'a+')
    g.write(gname+'\n')
    for i in re_list:
        false = 0
        count = 0
        ratio_list = []
        if i in gt_nolist:
            false +=1
            n = i.split('momolive_')[1].split('_f')[0]
            f_list.append(n)
            F_numv.append(n)
            g.write(i+'\n')
        elif i in gt_list:
            re_rect = re_dict.get(i)
            if(gt_dict.has_key(i)):
                gt_rect = gt_dict.get(i)
                n_re = re_rect[0]
                n_gt = gt_rect[0]
                for j in range(n_re):
                    hand_re = re_rect[1 + j * 4:1 + (j + 1) * 4]
                    for m in range(n_gt):
                        hand_gt = gt_rect[1 + m * 4:1 + (m + 1) * 4]
                        ratio = IOU(hand_re, hand_gt)
                        ratio_list.append(ratio)
                ratio_list.sort(reverse=True)#降序排列
                ratio = ratio_list[0]#取最大值
                if ratio >= iouP:
                    count += 1
#                    print ratio
                else:
                    false +=1
                    n = i.split('momolive_')[1].split('_f')[0]
                    f_list.append(n)
                    F_numv.append(n)
                    g.write(i+'\n')
        else:
            continue
        hand_regt = hand_regt + count
        hand_fault = hand_fault + false
    of_list = list(set(f_list))
    of_list.sort(key = f_list.index)
    fnum = len(of_list)
    oF_numv = list(set(F_numv))
    oF_numv.sort(key = F_numv.index)
    Fvideonum = len(oF_numv)
    g.close()
    totalTP = totalTP + hand_regt
    totalFP = totalFP + hand_fault
    print "******************************** %s"%(gname),"********************************"
    print " f_num_pic:",hand_fault,"  f_num_video:",fnum,"  falseDR=%6.2f"%(float(hand_fault)*100/totalGTpic),"%","  RE pic:",totalREpic
    print " video_index:",of_list

print "GT:",totalGTpic,"Fvideonum",Fvideonum,"FP:",totalFP
print "iou:", iouP

