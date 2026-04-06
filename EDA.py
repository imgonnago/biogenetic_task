import numpy as np
import pandas as pd

data = np.load('C:\\Users\\zxfg0\\biogenetic_task\\biogenetic\\X_SNP 1.npy')

print(data[0:10])
print(data.min())
print(data.max())
print(data.mean())
print(data.std())
