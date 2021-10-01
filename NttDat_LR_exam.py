# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
# adsverment cost
X = np.array([[13,
17,
20,
18,
15,
22,
21
],[344,
844,
944,
544,
244,
744,
694
]]).T
# popular
y = np.array([[266.667,
466.667,
566.667,
366.667,
266.667,
566.667,
466.667
]]).T
#
# Visualize data 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
print('Ham mat mat: y = ', round(w_1, 4), '*x_1 +',round(w_2, 4), '*x_2 +(',round(w_0, 4),")")
#Calculating home price
x_1=float(input("Nhap vao chi phi quang cao:"))
x_2=float(input("Nhap vao dan so:"))
y1 = w_1*x_1 + w_2*x_2 + w_0
print( u'Du doan doanh thu: %.2f ' %(y1))


