import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

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


sg_xs = [2, 5, 8, 11]
sg_ys = [4, 6, 10, 12]

mus = []
for expr in [5, 6]:
    for i in range(4):
        sgx_name = (sg_xs[i] - 2)
        sgy_name = (sg_ys[i] - 2)
        for itime in range(4):
            mus += [ - nd_dif[expr][itime, sgy_name] / nd_dif[expr][itime, sgx_name]]
            # print(mus)

print(mus)
plt.bar(np.arange(0, len(mus)), mus)
plt.show()
print(pd.Series(mus).describe())