from __future__ import print_function
import numpy as np 
sl=1
sl=int(input("sl="))
print(sl)
Datasheet=np.empty((sl,2))
for i in range(sl):
    Datasheet[i][0]=input("nhap x%s" %i)
    Datasheet[i][1]=input("nhap y%s" %i)
#=======tra ve nhan cua du lieu can tinh
