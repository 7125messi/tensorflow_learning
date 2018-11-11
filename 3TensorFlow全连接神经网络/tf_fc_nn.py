from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import tf_data_prepare # 导入数据预处理包
from sklearn.metrics import confusion_matrix

train_samples, train_labels = tf_data_prepare._train_samples, tf_data_prepare._train_labels
test_samples, test_labels = tf_data_prepare._test_samples, tf_data_prepare._test_labels

print('Training set', train_samples.shape, train_labels.shape)
print('Test set', test_samples.shape, test_labels.shape)

image_size = tf_data_prepare.image_size
num_labels = tf_data_prepare.num_labels
num_channels = tf_data_prepare.num_channels

"""
@ 启动tensorboard
tensorboard --logdir='D:\git\tensorflow\code\Tensorflow1\3TensorFlow全连接神经网络\board'
"""


def get_chunk(samples, labels, chunkSize):
    """
    这个函数是一个迭代器/生成器(yield)，用于每一次只得到 chunkSize 这么多的数据
    用于for/while，就像range函数
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd



class Network():
    def __init__(self, num_hidden, batch_size):
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
        """
        self.batch_size = batch_size
        self.test_batch_size = 500

        # HyperParameters
        self.num_hidden = num_hidden

        # Graph Related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None

        # 增加一个统计,可以为可视化所用
        self.merged = None

        # 初始化
        self.define_graph()
        self.session = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board', self.graph)

    def define_graph(self):
        """
        定义计算图谱
        """
        with self.graph.as_default():
            # 定义input层————>tensorboard
            with tf.name_scope('input'):
                # 1.定义图谱中的各种变量，使用 tf.placeholder批量喂给
                self.tf_train_samples = tf.placeholder(
                    tf.float32, shape=(self.batch_size, image_size, image_size, num_channels), name='tf_train_samples'
                )
                self.tf_train_labels = tf.placeholder(
                    tf.float32, shape=(self.batch_size, num_labels), name='tf_train_labels'
                )
                self.tf_test_samples = tf.placeholder(
                    tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels), name='tf_test_samples'
                )


            # 定义fc1层————>tensorboard
            with tf.name_scope('fc1'):
                # 2.定义全连接层1,隐藏层
                fc1_weights = tf.Variable(
                    tf.truncated_normal([image_size * image_size, self.num_hidden], stddev = 0.1), name='fc1_weights'
                )
                fc1_biases = tf.Variable(
                    tf.constant(0.1, shape=[self.num_hidden]), name='fc1_biases'
                )

                # 添加 fc1 中的 fc1_weights和 fc1_biases条形图
                tf.summary.histogram('fc1_weights', fc1_weights)
                tf.summary.histogram('fc1_biases', fc1_biases)



            # 定义fc2层————>tensorboard
            with tf.name_scope('fc2'):
                # 3.定义全连接层2,,输出层
                fc2_weights = tf.Variable(
                    tf.truncated_normal([self.num_hidden, num_labels], stddev = 0.1), name='fc2_weights'
                )
                fc2_biases = tf.Variable(
                    tf.constant(0.1, shape=[num_labels]), name='fc2_biases'
                )


                # 添加 fc2 中的 fc2_weights和 fc2_biases条形图
                tf.summary.histogram('fc2_weights', fc2_weights)
                tf.summary.histogram('fc2_biases', fc2_biases)



            # 4.定义图谱的运算
            def model(data):
                # 全连接层1
                shape = data.get_shape().as_list()
                # print(data.get_shape(), shape)
                reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
                #print(reshape.get_shape(), fc1_weights.get_shape(), fc1_biases.get_shape())

                # 定义fc1_model层————>tensorboard
                with tf.name_scope('fc1_model'):
                    fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
                    hidden = tf.nn.relu(fc1_model)

                # 定义fc2_model层————>tensorboard
                # 全连接层2
                with tf.name_scope('fc2_model'):
                    output = tf.matmul(hidden, fc2_weights) + fc2_biases
                    return output

            # 5.训练计算
            logits = model(self.tf_train_samples)

            # 定义loss————>tensorboard
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.tf_train_labels)
                )

                # 添加 loss
                tf.summary.scalar('Loss', self.loss)


            # 6.优化
            # 定义optimizer————>tensorboard
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)



            # 定义train_prediction、test_prediction————>tensorboard
            # 7.训练数据预测、验证和测试
            with tf.name_scope('predictions'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
                self.test_prediction = tf.nn.softmax(model(self.tf_test_samples), name='test_prediction')


            # 综合提取
            self.merged = tf.summary.merge_all()



    def run(self):
        """
        用到session
        """
        # 1.私有函数,打印混淆矩阵
        def print_confusion_matrix(confusionMatrix):
            print('Confusion Matrix:')
            # 计算行占比
            for i, line in enumerate(confusionMatrix):
                print(line, line[i] / np.sum(line))


            a = 0
            # 计算列占比
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
                print(column[i] / np.sum(column), )
            print('\n', np.sum(confusionMatrix), a)


        # 2.运算
        with self.session as session:
            tf.global_variables_initializer().run()

            # 训练集训练模型
            print('Start Training')
            """
            @ batch = 1000
            """
            for i, samples, labels in get_chunk(train_samples, train_labels, chunkSize=self.batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged],
                    feed_dict={self.tf_train_samples:samples, self.tf_train_labels:labels}
                )

                self.writer.add_summary(summary, i)

                # 计算正确标签率
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step {0}, is {1:.4f}'.format(i, l))
                    print('Minibatch accuracy:{0}'.format(accuracy))


            # 测试集验证
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
                result = session.run(
                    self.test_prediction,
                    feed_dict={self.tf_test_samples:samples}
                )
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: {0:.4f}'.format(accuracy))
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        """
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        """
        _predictions = np.argmax(predictions, axis=1)
        _labels = np.argmax(labels, axis=1)

        if need_confusion_matrix is True:
            cm = confusion_matrix(_labels, _predictions)
        else:
            cm = None

        # 简写
        # cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None

        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm


if __name__ == '__main__':
    net = Network(num_hidden=128, batch_size=100)
    net.run()
