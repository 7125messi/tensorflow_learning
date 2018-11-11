# 兼容Python2
from __future__ import print_function,division
import tensorflow as tf
import numpy as np
print('Loaded TF version', tf.__version__)

def tf_basic_operation():
    graph = tf.Graph()
    with graph.as_default():
        # 省内存，通常使用placeholder，喂入数据
        value1 = tf.placeholder(dtype=tf.float64)
        value2 = tf.Variable([3,4], dtype=tf.float64)
        mul = value1 * value2

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        # 假装来了很多数据
        value = load_data()

        # 利用yield关键字：
        # yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器
        # 当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象
        # 当你使用for进行迭代的时候，函数中的代码才会执行
        for partialValue in load_partial(value, 2):
            runResult = sess.run(mul, feed_dict={value1:partialValue})
            print('乘法(value1, value2) = ', runResult)

def load_data():
    return [-x for x in range(1000)]

def load_partial(value, step):
    index = 0
    while index < len(value):
        yield value[index:index + step]
        index += step
    return


if __name__ == '__main__':
    tf_basic_operation()
