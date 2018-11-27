# coding: utf-8

import glob
import numpy as np
import math

temp1 = [1,2,3]
temp2 = [4,5,6]
temp = np.array([temp1, temp2])
temp = temp.transpose()
print(np.random.shuffle(temp))