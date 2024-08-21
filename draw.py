import re

import numpy as np
from matplotlib import pyplot as plt


def read_result(result_path):
    # 打开并读取txt文件
    with open(result_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    accuracies = []

    # 遍历每一行
    for line in lines:
        # 提取epoch信息
        if "epoch" in line:
            epoch = int(re.search(r'epoch=(\d+)', line).group(1))
            epochs.append(epoch)

        # 提取accuracy信息
        if "accuracy" in line:
            accuracy = float(re.search(r'accuracy\s+(\d+\.\d+)', line).group(1))
            accuracies.append(accuracy)

    return epochs, accuracies


def draw(epochs, accuracies):

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('Accuracy Varies With The Training Cycle')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, len(epochs) + 1, step=1))

    # 显示网格和图形
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    result_path = 'artifact/result/Ranger.txt'
    epochs, accuracies = read_result(result_path)
    # 加入baseline
    epochs.insert(0, 0)
    accuracies.insert(0, 0.14)
    print(max(accuracies))
    draw(epochs, accuracies)