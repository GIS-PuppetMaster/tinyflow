from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import xlabel
from matplotlib.ticker import MultipleLocator
from numpy import arange
from MakeCSV import make_csv

# make_csv()
net_type = ['VGG-16', 'InceptionV3', 'InceptionV4', 'ResNet-50', 'DenseNet']
method_type = ['TENSILE', 'vDNN', 'Capuchin']
df_list = []
csv_path = 'D:\PycharmProjects\TENSILE实验数据/TENSILE/log/'
with open(csv_path + 'MultiWorkloadsMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open(csv_path + 'MultiWorkloadsTIME.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
# with open(csv_path + 'MultiWorkloadsEOR.csv') as f:
#     df = pd.read_csv(f, index_col=False)
#     df_list.append(df)
with open(csv_path + 'MultiWorkloadsCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open(csv_path + 'MultiWorkloadsMSR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open(csv_path + 'MultiWorkloadsEOR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
with open(csv_path + 'MultiWorkloadsCBR_cold_start.csv') as f:
    df = pd.read_csv(f, index_col=False)
    df_list.append(df)
# plt.figure(figsize=(40, 16))
markersize=8
fig, axis = plt.subplots(3, 5, sharex='col', sharey='row', figsize=(12, 6))
# plt.subplots_adjust(wspace=, hspace=0.3)
for i in range(5):
    df = df_list[0]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[0, i].plot(x, TENSILE, 'bD-', label='TENSILE', markersize=markersize)
    df = df_list[3]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[0, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$', markersize=markersize)
    axis[0, i].plot(x, vDNN, 'y^-', label='vDNN', markersize=markersize)
    axis[0, i].plot(x, Capuchin, 'go-', label='Capuchin', markersize=markersize)
    axis[0, i].yaxis.set_major_locator(MultipleLocator(0.1))

    # axis[0, i].set_xlabel(f'{net_type[i]}')
    if i == 0:
        axis[0, i].set_ylabel('MSR', size=12)

    df = df_list[1]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[1, i].plot(x, TENSILE, 'bD-', label='TENSILE', markersize=markersize)
    df = df_list[4]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[1, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$', markersize=markersize)
    axis[1, i].plot(x, vDNN, 'y^-', label='vDNN', markersize=markersize)
    axis[1, i].plot(x, Capuchin, 'go-', label='Capuchin', markersize=markersize)
    # axis[1, i].yaxis.set_major_locator(MultipleLocator(0.5))
    # axis[1, i].set_xlabel(f'{net_type[i]}')
    if i == 0:
        axis[1, i].set_ylabel('Time Cost(s)', size=12)

    df = df_list[2]
    df = np.array(df)[..., 1:]
    x = ['x1', 'x2', 'x3']
    TENSILE = df[i, :]
    vDNN = df[i + 5, :]
    Capuchin = df[i + 10, :]
    axis[2, i].plot(x, TENSILE, 'bD-', label='TENSILE', markersize=markersize)
    df = df_list[5]
    df = np.array(df)[..., 1:]
    TENSILE = df[i, :]
    axis[2, i].plot(x, TENSILE, 'rx-', label='TENSILE$_{cs}$', markersize=markersize)
    axis[2, i].plot(x, vDNN, 'y^-', label='vDNN', markersize=markersize)
    axis[2, i].plot(x, Capuchin, 'go-', label='Capuchin', markersize=markersize)
    axis[2, i].set_xlabel(f'{net_type[i]}',size=12)
    if i == 0:
        # axis[2, i].set_yscale('log', basey=2)
        axis[2, i].set_ylabel('CBR', size=12)
    axis[2, i].yaxis.set_major_locator(MultipleLocator(1))
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, fontsize=10)
fig.savefig(csv_path + 'pic/MultiWorkloads.png')
fig.show()

markersize = 8
markers = ['D-','x-','^-','o-', 'h-']
plt.figure(figsize=plt.rcParams['figure.figsize'])
fig, axis = plt.subplots(1, 3, sharex='col', figsize=(12, 4))
plt.subplots_adjust(wspace=0.3)
with open(csv_path + 'BatchSizeMSR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
x = ['2', '4', '8', '16', '32']
# plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    axis[0].plot(x, list(df[i]), markers[i], label=net, markersize=markersize)
axis[0].set_xlabel(f'Batch Size', size=12)
axis[0].set_ylabel(f'Memory Saving Ratio',size=12)
# plt.legend(net_type)
# plt.xlabel('batch size')
# plt.ylabel('Memory Saving Ratio')
# plt.savefig(csv_path + 'pic/BatchSizeMSR.png')
# plt.show()

with open(csv_path + 'BatchSizeEOR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv(csv_path + 'BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
# plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    axis[1].plot(x, list(df[i]), markers[i], label=net, markersize=markersize)
axis[1].set_xlabel(f'Batch Size',size=12)
axis[1].set_ylabel(f'Extra Overhead Ratio',size=12)
# plt.legend(net_type)
# plt.xlabel('batch size')
# plt.ylabel('Extra Overhead Ratio')
# plt.savefig(csv_path + 'pic/BatchSizeEOR.png')
# plt.show()

with open(csv_path + 'BatchSizeCBR.csv') as f:
    df = pd.read_csv(f, index_col=False)
df = np.array(df)[..., 1:]
# df.index =['2', '4', '8', '16', '32']
# df.to_csv(csv_path + 'BatchsizeMSR_temp.csv')
x = ['2', '4', '8', '16', '32']
# plt.xticks(arange(len(x)), x)
for i, net in enumerate(net_type):
    axis[2].plot(x, list(df[i]), markers[i], label=net, markersize=markersize)
axis[2].set_xlabel(f'Batch Size',size=12)
axis[2].set_ylabel(f'Cost Benefit Ratio',size=12)
# plt.legend(net_type)
# plt.xlabel('batch size')
# plt.ylabel('Cost Benefit Ratio')
lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)
plt.savefig(csv_path + 'pic/BatchSize.png', box_inches='tight')
plt.show()
