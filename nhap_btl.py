import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
from sklearn import linear_model
#import statsmodels.api as sm
#path ='dataset/'
path = 'test.csv'
dataset = pd.read_csv(path)
print('\nNumber of rows and columns in the data set: ',dataset.shape)
print('')
Y = dataset[['price']]

#X = dataset.drop(['price', 'id', 'year'],  axis=1)
X = dataset[['living_square']]
x = np.array(X.values)
y = np.array(Y.values)
regr = linear_model.LinearRegression()
regr.fit(x, y)
print(x)
print(y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
y1 = regr.coef_[0][0]*3+  regr.intercept_
print( u'Du doan gia nha co dien tich 537 m2: %.2f (USD)' %(y1))

#regr.coef_[0][1]*1+ regr.coef_[0][2]*1+regr.coef_[0][3]*200 +
