import numpy as np


def frange_cycle_linear( n_iter, start=0.0, stop=1.0,  n_cycle=10, ratio=1):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    

L  = frange_cycle_linear(1000, start=0.0, stop=1.0, n_cycle=4, ratio=1)

# print(L)

import pandas as pd
df = pd.read_csv("/home/clu98753cs13/Desktop/DL/LAB4/Lab4_template/result/submission.csv", header=None)
print(df.shape)  # ➜ 應該是 (630, 6144) (3151, 6145)
