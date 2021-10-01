from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
N = 10
X0 = np.array([[9.37319011,8.71875981],
[8.51261889,8.40558943],
[9.4696794,9.02144973],
[8.78736889,8.29380961],
[8.81231157,8.56119497],
[9.03717355,8.93397133],
[8.53790057,8.87434722],
[9.29312867,9.76537389],
[8.38805594,8.86419379],
[8.57279694,7.90707347]
])
X1 = np.array([[10.42746579,7.71254431],
[11.24760864,9.39846497],
[10.33595491,8.61731637],
[10.69420104,8.94273986],
[11.53897645,9.54957308],
[10.3071994,7.19362396],
[11.13924705,9.09561534],
[11.47383468,9.41269466],
[11.00512009,8.89290099],
[11.28205624,8.79675607]
])
X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels

#==================

from cvxopt import matrix, solvers
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((2*N, 1))) # all-one vector 
# build A, b, G, h 
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
print(l.T)
epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)
