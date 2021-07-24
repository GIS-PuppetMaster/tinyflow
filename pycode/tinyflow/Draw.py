from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import xlabel
from matplotlib.ticker import MultipleLocator
from numpy import arange
from MakeCSV import make_csv

make_csv()
net_type = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
method_type = ['TENSILE', 'vDNN', 'Capuchin']
df_list = []
with open('./log/MultiWorkloadsMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open('./log/MultiWorkloadsEOR.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open('./log/MultiWorkloadsCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open('./log/MultiWorkloadsMSR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open('./log/MultiWorkloadsEOR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open('./log/MultiWorkloadsCBR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
# plt.figure(figsize=(40, 16))
fig, axis = plt.subplots(3, 5, sharex='col', sharey='row', figsize=(9, 6))
# plt.subplots_adjust(wspace=, hspace=0.3)
for i in range(5):
    df = df_list[0]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[0, i].plot(x, TENSILE, 'bD-', label='TENSILE')
    df = df_list[3]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[0, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$')
    axis[0, i].plot(x, vDNN, 'y^-', label='vDNN')
    axis[0, i].plot(x, Capuchin, 'go-', label='Capuchin')
    axis[0, i].yaxis.set_major_locator(MultipleLocator(0.1))

    # axis[0, i].set_xlabel(f'{net_type[i]}')
    if i == 0:
        axis[0, i].set_ylabel('MSR')

    df = df_list[1]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[1, i].plot(x, TENSILE, 'bD-', label='TENSILE')
    df = df_list[4]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[1, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$')
    axis[1, i].plot(x, vDNN, 'y^-', label='vDNN')
    axis[1, i].plot(x, Capuchin, 'go-', label='Capuchin')
    axis[1, i].yaxis.set_major_locator(MultipleLocator(0.5))
    # axis[1, i].set_xlabel(f'{net_type[i]}')
    if i == 0:
        axis[1, i].set_ylabel('EOR')

    df = df_list[2]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[2, i].plot(x, TENSILE, 'bD-', label='TENSILE')
    df = df_list[5]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[2, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$')
    axis[2, i].plot(x, vDNN, 'y^-', label='vDNN')
    axis[2, i].plot(x, Capuchin, 'go-', label='Capuchin')
    axis[2, i].set_xlabel(f'{net_type[i]}')
    if i == 0:
        # axis[2, i].set_yscale('log', basey=2)
        axis[2, i].set_ylabel('CBR')
    axis[2, i].yaxis.set_major_locator(MultipleLocator(1))
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
fig.savefig(f'./log/pic/MultiWorkloads.png')
fig.show()
plt.figure(figsize=plt.rcParams['figure.figsize'])
with open('./log/BatchSizeMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['2', '4', '8', '16', '32']
plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    plt.plot(list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Memory Saving Ratio')
plt.savefig('./log/pic/BatchSizeMSR.png')
plt.show()

with open('./log/BatchSizeEOR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv('./log/BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    plt.plot(list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Extra Overhead Ratio')
plt.savefig('./log/pic/BatchSizeEOR.png')
plt.show()

with open('./log/BatchSizeCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv('./log/BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    plt.plot(list(df[i]))
plt.legend(net_type)
plt.xlabel('batch size')
plt.ylabel('Cost Benefit Ratio')
plt.savefig('./log/pic/BatchSizeCBR.png')
plt.show()
