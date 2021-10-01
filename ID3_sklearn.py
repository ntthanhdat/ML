from __future__ import division, print_function, unicode_literals
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

x = np.array([["Gioi","Gioi","Gioi"],
              ["Kha","TB","TB"],
              ["Gioi","Gioi","Kha"],
              ["TB","Gioi","Gioi"],
              ["Gioi","Gioi","TB"],
              ["TB","Kha","Kha"],
              ["Kha","Gioi","Gioi"],
              ["TB","Kha","TB"],
              ["Kha","TB","Gioi"],
              ["Gioi","TB","TB"],
              ["TB","TB","Gioi"],
              ["Gioi","Kha","TB"],
              ["Kha","Gioi","TB"],
              ["Gioi","TB","Kha"],
              ["TB","TB","TB"]
             ])
y= np.array(["Do","Truot","Do","Do","Do","Do","Do","Truot","Do","Truot","Truot","Do","Do","Do","Truot"])

clf = LabelEncoder()
for i in range(x.shape[1]):
    x[:, i] = clf.fit_transform(x[:,i])

print(x)
model = tree.DecisionTreeClassifier(criterion = 'entropy').fit(x,y)
tree.plot_tree(model)
