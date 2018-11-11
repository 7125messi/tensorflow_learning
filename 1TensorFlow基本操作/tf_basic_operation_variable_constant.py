# 兼容Python2
from __future__ import print_function,division
import tensorflow as tf
import numpy as np
print('Loaded TF version', tf.__version__)

def tf_basic_operation():
    # Creates a new, Empty Graph
    graph = tf.Graph()
    with graph.as_default():
        # constant
        c1 = tf.constant(10)
        c2 = tf.constant(5)
        addc = c1 + c2
        print(addc)
        print(type(addc))
        print(type(c1))

        # Variable
        v1 = tf.Variable(10)
        v2 = tf.Variable(5)
        addv = v1 + v2
        print(addv)
        print(type(addv))
        print(type(v1))

    # 用来运行计算图谱的对象/实例
    # Variable -> 初始化 -> 有值的Tensor
    # 变量是需要初始化的
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print('加法(c1,c2) = ', sess.run(addc))
        print('加法(v1,v2) = ', sess.run(addv))


if __name__ == '__main__':
    tf_basic_operation()
