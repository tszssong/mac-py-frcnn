# -*- coding: utf-8 -*-
"""
    Created on Thu Nov  2 14:35:42 2017
    
    @author: hans
    
    http://blog.csdn.net/renhanchi
    """

import matplotlib.pyplot as plt
import numpy as np
import commands
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
                    '-p','--log_path',
                    type = str,
                    default = '',
                    help = """\
                        path to log file\
                        """
                    )

FLAGS = parser.parse_args()

train_log_file = FLAGS.log_path


display = 10 #solver
test_interval = 100 #solver

time = 5

train_output = commands.getoutput("cat " + train_log_file + " | grep 'Train net output #0' | awk '{print $11}'")  #train mbox_loss
accu_output = commands.getoutput("cat " + train_log_file + " | grep 'Test net output #0' | awk '{print $11}'") #test detection_eval

train_loss = train_output.split("\n")
test_accu = accu_output.split("\n")

def reduce_data(data):
    iteration = len(data)/time*time
    _data = data[0:iteration]
    if time > 1:
    data_ = []
    for i in np.arange(len(data)/time):
        sum_data = 0
            for j in np.arange(time):
                index = i*time + j
                    sum_data += float(_data[index])
                        data_.append(sum_data/float(time))
                            else:
                                data_ = data
                                    return data_

train_loss_ = reduce_data(train_loss)
test_accu_ = reduce_data(test_accu)

_,ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(time*display*np.arange(len(train_loss_)), train_loss_)
ax2.plot(time*test_interval*np.arange(len(test_accu_)), test_accu_, 'r')

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Train Loss')
ax2.set_ylabel('Test Accuracy')
plt.show()  
