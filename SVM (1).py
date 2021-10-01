import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X1 = [[4.37319011,3.71875981], [3.51261889,3.40558943], [4.4696794,4.02144973], [3.78736889,3.29380961], [3.81231157, 3.56119497], [4.03717355,3.93397133], [3.53790057,3.87434722], [4.29312867,4.76537389], [3.38805594,3.86419379], [3.57279694, 2.9070734]]
y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [[5.42746579,2.71254431], [6.24760864,4.39846497], [5.33595491,3.61731637], [5.69420104,3.94273986], [6.53897645,4.54957308], [5.3071994,2.19362396], [6.13924705,4.09561534], [6.47383468,4.41269466], [6.00512009,3.89290099], [6.28205624,3.79675607]]
y2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
X = np.array(X1 + X2)
y = y1 + y2

clf = SVC(kernel='linear', C=1E10)
clf.fit(X, y)
print(clf.support_vectors_)

def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = clf.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='brg');
plot_svc_decision_function(clf)
plt.show()
