from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)

Datasheet= np.array([[8.84539996, 8.55967159], 
[9.08296992, 10.02486694], 
[9.39101392, 9.17637385], 
[11.87226421, 10.05096564], 
[11.67924793, 9.36893594], 
[10.5072737, 8.74738715]]
)
Returnsheet=np.empty((Datasheet.shape[0], 1))
w =  np.array([[-2.00984378,  0.64068339]])
b =  14.252683259127915
print(Datasheet)
print("###########")
print(Datasheet.shape[1])
print("###########")
for i in range(Datasheet.shape[0]):
    aa=b
    for j in range(Datasheet.shape[1]):
        s=Datasheet[i][j]*w[0][j]
        aa+=s
    print(aa)
    if(aa<0):
        Returnsheet[i][0]=-1
    else:
        Returnsheet[i][0]=1
print(Returnsheet)

