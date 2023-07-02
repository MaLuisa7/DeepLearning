import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x =[31.42,32.77,32.64,33.95]
x_input = np.reshape(x,(4,1))

wx = [[0.1866,1.2369]]
wh = [[0.8698,0.4933],
      [0.4933,0.8698]]
bh = [0,0]
wy = [[0.4635998],
      [0.6538409]]
by = [0]

m =2 # dos neuronas recurrentes
h0 = np.zeros(m)
h1 = np.dot(x[0],wx) + h0 + bh
h2 = np.dot(x[1],wx) + np.dot(h1,wh)   + bh
h3 = np.dot(x[2],wx) + np.dot(h2,wh)  + bh
h4 = np.dot(x[3],wx) + np.dot(h3,wh)  + bh
y4 = np.dot(h4,wy) + by
y4

