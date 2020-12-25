import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

sg_xs = [2, 3, 5, 7, 8, 9, 11]
cols = ['sg' + str(i) for i in sg_xs]
print(cols)

df = pd.read_csv('input/data123.csv')
# print(df)
# print(df[df['experiment'] == 1])
# print(type(df['sg2']))

experiments = np.arange(1, 7)
df_exprmnts = [0] * 7
print(df_exprmnts)
for i in experiments:
    df_exprmnts[i] = df[df['experiment'] == i]
    # print(df_exprmnts[i])

lst = df_exprmnts[5]['sg2'].tolist()

print(lst)

def print_a_sg(epsilon_sg):
    forces = [800 * i for i in np.arange(1, 7)]

total_dif = []
for expr in [5, 6]:
    for sg in sg_xs:
        col_name = 'sg' + str(sg)
        epsilon_sg = df_exprmnts[expr][col_name].tolist()
        for i in range(4):
            total_dif += [epsilon_sg[i + 1] - epsilon_sg[i]]

plt.bar(np.arange(0, len(total_dif)), total_dif)
# plt.hist(total_dif)
# plt.show()
total_dif = pd.Series(total_dif)

print(total_dif)
print(total_dif.describe())