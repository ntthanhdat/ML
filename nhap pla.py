import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
#X0 = np.random.multivariate_normal(means[0], cov, N).T
#X1 = np.random.multivariate_normal(means[1], cov, N).T
X0=np.array([[9.37319011, 
8.51261889, 
9.4696794, 
8.78736889, 
8.81231157, 
9.03717355, 
8.53790057, 
9.29312867, 
8.38805594, 
8.57279694],
[8.71875981, 
8.40558943, 
9.02144973, 
8.29380961, 
8.56119497, 
8.93397133, 
8.87434722, 
9.76537389, 
8.86419379, 
7.90707347]])
X1=np.array([[10.42746579, 
11.24760864, 
10.33595491, 
10.69420104, 
11.53897645, 
10.3071994, 
11.13924705, 
11.47383468, 
11.00512009, 
11.28205624],
[7.71254431, 
9.39846497, 
8.61731637, 
8.94273986, 
9.54957308, 
7.19362396, 
9.09561534, 
9.41269466, 
8.89290099, 
8.79675607]])
X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
#==================du lieu can tinh
Datasheet= np.array([[8.84539996, 8.55967159], 
[9.08296992, 10.02486694], 
[9.39101392, 9.17637385], 
[11.87226421, 10.05096564], 
[11.67924793, 9.36893594], 
[10.5072737, 8.74738715]])
Returnsheet=np.empty((Datasheet.shape[0], 1))

def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    
    return np.array_equal(h(w, X), y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 

                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)
true_w=w[-1].T
print(true_w)
#=======tra ve nhan cua du lieu can tinh
print("#"*10+"vector du lieu la:")
print(Datasheet)

for i in range(Datasheet.shape[0]):
    aa=true_w[0][0]
    for j in range(Datasheet.shape[1]):
        s=Datasheet[i][j]*true_w[0][j+1]
        aa+=s
    if(aa<0):
        Returnsheet[i][0]=-1
    else:
        Returnsheet[i][0]=1
print("#"*10+"vector label la:")
print(Returnsheet)
