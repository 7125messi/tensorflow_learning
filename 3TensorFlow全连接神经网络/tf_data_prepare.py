from __future__ import print_function, division

# 数据预处理流程
"""
1. 下载数据  http://ufldl.stanford.edu/housenumbers/
2. 探索数据
3. 处理数据
4. 构建一个基本网络, 基本的概念+代码 ， TensorFlow的世界
5. 卷积ji
6. 来实验吧
7. 微调与结果
"""

from scipy.io import loadmat as load  # 可以读取 *.mat 数据
import matplotlib.pyplot as plt
import numpy as np


# 定义函数：改变原始数据samples的形状
def reformat(samples, labels):
    """
      0       1       2      3          3       0      1      2
    (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
    """
    new_samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)


    """
    one_hot:
    labels 变成 one-hot encoding, [0] -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels 变成 one-hot encoding, [1] -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    labels 变成 one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ...
    labels 变成 one-hot encoding, [9] -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    labels 变成 one-hot encoding, [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        # copy10个 0.0 ————> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            # 设置某一标签数字为1.0
            one_hot[num] = 1.0
        # 添加某一标签数字到one_hot_labels里面
        one_hot_labels.append(one_hot)

    # 转换为后面Tensorflow张量计算的数据类型格式
    labels = np.array(one_hot_labels).astype(np.float32)
    # 函数返回new_samples和labels
    return new_samples, labels

# 定义归一化函数
def normalize(samples):
    """
    灰度化: 从三色通道 -> 单色通道     省内存 + 加快训练速度
    (R + G + B) / 3

    将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
    @samples: numpy array
    """
    # shape (图片数，图片高，图片宽，通道数) axis=3通道数
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a / 3.0
    return a / 128.0 - 1.0




# 定义函数：用来查看标签分布，画统计图
"""
keys:
0
1
2
...
9
"""
def distribution(labels, name):
    count = {}
    for label in labels:
        if label[0] == 10:
            key = 0
        else:
            key = label[0]
        # key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()



def inspect(dataset, labels, i):
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()


##### 加载数据
train = load('D:/git/tensorflow/code/data/train_32x32.mat')
test = load('D:/git/tensorflow/code/data/test_32x32.mat')
# extra = load('D:/git/tensorflow/code/data/extra_32x32.mat')

# print('Train Samples Shape:', train['X'].shape)
# print('Train  Labels Shape:', train['y'].shape)
#
# print('Test Samples Shape:', test['X'].shape)
# print('Test  Labels Shape:', test['y'].shape)

# print('Extra Samples Shape:', extra['X'].shape)
# print('Extra  Labels Shape:', extra['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
# test_samples = extra['X']
# test_labels = extra['y']




##### 转换原始数据形状，利用reformat函数
n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)


##### 均一化数据，利用normalize函数
_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    pass
    # inspect(n_train_samples, _train_labels, 1234)
    # inspect(_train_samples, _train_labels, 1234)
    #
    # inspect(n_test_samples, _test_labels, 1234)
    # inspect(_test_samples, _test_labels, 1234)
    #
    # distribution(train_labels, 'Train Labels')
    # distribution(test_labels, 'Test Labels')
