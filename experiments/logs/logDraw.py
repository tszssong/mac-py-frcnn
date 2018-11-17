import matplotlib.pyplot as plt
import re

with open('./log.log') as f:
    data = f.read()
#    print data

pattern = re.compile('solver.cpp:239] Iteration (\d+)')
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
#print total_loss

pattern = re.compile(': RegressionLoss = ([\.\deE+-]+)')
results = re.findall(pattern, data)
mbox_loss = []
for result in results:
    regloss = float(result)
    mbox_loss.append(regloss)
#    if regloss < 0.02 :
#        mbox_loss.append(regloss)
#    else:
#        iter_num.pop()

#print mbox_loss

pattern = re.compile(', lr = ([\.\deE+-]+)')
results = re.findall(pattern, data)
learning_rate = []
for result in results:
    learning_rate.append(float(result))
#print learning_rate
print len(iter_num), len(total_loss), len(mbox_loss), len(learning_rate)

plt.subplot(311)
plt.plot(iter_num, total_loss)
plt.subplot(312)
plt.plot(iter_num, mbox_loss)
plt.subplot(313)
plt.plot(iter_num, learning_rate)

plt.show()
