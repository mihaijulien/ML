import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

path = os.getcwd() + '/data/ex6data1.mat'
data = loadmat(path)

for key in data:
	print(data[key])