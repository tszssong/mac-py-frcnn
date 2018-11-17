import matplotlib.pyplot as plt
import re

with open('experiments/logs/faster_rcnn_end2end_MMCV5S8_.txt.2018-11-09_14-55-52') as f:
    data = f.read()
#    print data

pattern = re.compile('solver.cpp:218] Iteration (\d+)')
results = re.findall(pattern, data)
iter_num = []
for result in results:
    iter_num.append(int(result))

#pattern = re.compile(': ClassifyLoss = ([\.\deE+-]+)')
pattern = re.compile(', loss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
total_loss = []
for result in results:
    total_loss.append(float(result))

pattern = re.compile(': bbox_loss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
mbox_loss = []
for result in results:
    regloss = float(result)
    mbox_loss.append(regloss)

pattern = re.compile(': cls_loss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
cls_loss = []
for result in results:
    cls_loss.append(float(result))

pattern = re.compile(': rpn_cls_loss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
rpn_cls_loss = []
for result in results:
    rpn_cls_loss.append(float(result))

pattern = re.compile(': rpn_loss_bbox = ([\.\deE+-]+)')
results = re.findall(pattern, data)
rpn_bbox_loss = []
for result in results:
    rpn_bbox_loss.append(float(result))

#pattern = re.compile(', lr = ([\.\deE+-]+)')
#results = re.findall(pattern, data)
#learning_rate = []
#for result in results:
#    learning_rate.append(float(result))
#print learning_rate
plt.figure(1)
short = min(len(iter_num), len(total_loss), len(mbox_loss), len(cls_loss))
#short = 100
print len(iter_num), len(total_loss), len(mbox_loss), len(cls_loss), short

plt.subplot(411)
plt.plot(iter_num[:short], mbox_loss[:short])
#plt.ylim(0.001,0.5)
plt.title("bbox loss")
plt.subplot(412)
plt.plot(iter_num[:short], cls_loss[:short])
#plt.ylim(0.001,0.5)
plt.title("cls loss")

plt.subplot(413)
plt.plot(iter_num[:short], rpn_cls_loss[:short])
#plt.ylim(0.001,0.2)
plt.title("rpn cls loss")
plt.subplot(414)
plt.plot(iter_num[:short], rpn_bbox_loss[:short])
plt.grid()
#plt.ylim(0.001,0.02)
plt.title("rpn bbox loss")

plt.figure(2)
short = min(len(iter_num), len(total_loss), len(mbox_loss), len(cls_loss))
print len(iter_num), len(total_loss), len(mbox_loss), len(cls_loss), short
#plt.ylim(0.001,0.2)
plt.plot(iter_num[:short], total_loss[:short])
plt.title("total loss")
plt.grid()
plt.show()
