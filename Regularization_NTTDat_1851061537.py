# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import math
# size (m2)
X = np.array([[42,
44.4138,
46.8276,
49.2414,
51.6552,
54.069,
56.4828,
58.8966,
61.3103,
63.7241,
66.1379,
68.5517,
70.9655,
73.3793,
75.7931,
78.2069,
80.6207,
83.0345,
85.4483,
87.8621,
90.2759,
92.6897,
95.1034,
97.5172,
99.931,
102.3448,
104.7586,
107.1724,
109.5862,
112
]]).T
# price (USD)
y = np.array([[ 460.524,
521.248,
547.104,
563.432,
635.418,
637.992,
667.248,
713.377,
760.918,
769.881,
843.004,
867.409,
878.707,
914.545,
964.261,
1007.531,
1081.78,
1086.42,
1115.88,
1150.69,
1165.13,
1252.27,
1263.9,
1299.97,
1332.47,
1386.92,
1422.16,
1481.69,
1490.54,
1527.28
]]).T
#
plt.plot(X.T, y.T, 'ro') 
plt.axis([ 60, 130,460, 1527])
plt.axis([30, 120, 400, 1600])
plt.xlabel('Dien tich (m2)')
plt.ylabel('Gia (USD)')
plt.show()
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
w_x=math.sqrt(math.pow(w_0,2)+math.pow(w_1,2))
print('tham so ham mat mat w = ', w)
print('regularization = ', w_x)
x0 = np.linspace(35, 125, 2)
y0 = w_0 + w_1*x0+w_x

#Calculating home price
y1 = w_1*537 + w_0+w_x
print( u'Du doan gia nha co dien tich 537 m2: %.2f (USD)' %(y1))


# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([30, 120, 400, 1600])
plt.xlabel('Dien tich (m2)')
plt.ylabel('Gia (USD)')
plt.show()
