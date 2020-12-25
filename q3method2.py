import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model 

# 由实验一得到的已知数据
E = 214.6e9
A = 81e-6
I_y = 29.9e-6 * 9e-4
# import data
df = pd.read_csv('input/data123.csv')
experiments = np.arange(1, 7)
df_exprmnts = [0] * 7
nd_dif = [0] * 7
sgs = np.arange(2, 13)
for expr in experiments:
    df_exprmnts[expr] = df[df['experiment'] == expr]
    nd_dif[expr] = np.array(df_exprmnts[expr][['sg' + str(i) for i in sgs]])
    for i in range(4):
        nd_dif[expr][i, :] = nd_dif[expr][i + 1, :] - nd_dif[expr][i, :]
    nd_dif[expr] = np.delete(nd_dif[expr], -1, axis = 0)

    print(nd_dif[expr])

sg_shorts = [2, 3]
sg_longs = [7, 8, 9]
sg_short_places = [[0.02, 0], [0.005, 0]] 
sg_long_places = [[0,0.006],[0,0.026],[0, 0.046]] 

y_mids = []
z_mids = []
for expr in [3, 4]:
    for itime in range(4): 
        xs = []
        ys = []
        for i, sg in enumerate(sg_shorts):
            xs += [[sg_short_places[i][0]]]
            ys += [nd_dif[expr][itime, sg - 2] * 1e-6 * E]
        clf = linear_model.LinearRegression()
        clf.fit(xs, ys)
        # print(xs)
        print(ys)
        # print(type(clf.intercept_), - clf.intercept_ / clf.coef_[0])
        print(800 / A)
        print( clf.intercept_)
        y_mids += [(800 / A - clf.intercept_ )/ clf.coef_[0]]
        x2s = [] 
        y2s = []
        for i, sg in enumerate(sg_longs):
            x2s += [[sg_long_places[i][1]]]
            y2s += [nd_dif[expr][itime, sg - 2] * 1e-6 * E]
        clf2 = linear_model.LinearRegression() 
        clf2.fit(x2s, y2s)
        z_mids += [(800 / A - clf2.intercept_ )/ clf2.coef_[0]]


print(y_mids)
print(z_mids)

print(pd.Series(y_mids).describe())
print(pd.Series(z_mids).describe())

# visualization
plt.subplot(1, 2, 1)
x = np.arange(0, 40, 1)
k = clf.coef_ 
b = clf.intercept_
y = x * k  * 1e-3 + b - 800 / A
plt.plot(x, y)
plt.plot((800 / A - b )/ k / 1e-3, 0, 'r*')
print((800 / A - b )/ k / 1e-3)

plt.grid()
#
plt.subplot(1, 2, 2)
x = np.arange(0, 50, 1)
k = clf2.coef_ 
b = clf2.intercept_
y = x * k  * 1e-3 + b - 800 / A
plt.plot(x, y)
plt.plot((800 / A - b )/ k / 1e-3, 0, 'r*')
plt.xticks(np.arange(0, 51, 10))
plt.grid()

plt.show()
print((800 / A - b )/ k / 1e-3)
