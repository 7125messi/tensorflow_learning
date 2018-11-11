from __future__ import print_function, division

# 第三方
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

# 自定义
import load

train_samples, train_labels = load._train_samples, load._train_labels
test_samples, test_labels = load._test_samples, load._test_labels

print('Training set', train_samples.shape, train_labels.shape)
print('    Test set', test_samples.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels

# tensorboard --logdir='D:\git\tensorflow\code\Tensorflow1\4Tensorboard数据可视化\board'


def get_chunk(samples, labels, chunkSize):
    """
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for、loop，就像range()函数
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0  # initial step
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

        # Hyper Parameters
        self.num_hidden = num_hidden

        # Graph Related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None

        # 增加一个统计，为可视化所用
        self.merged = None

        # 初始化
        self.define_graph()
        self.session = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board', self.graph)
        """
        In [14]: tf.summary.FileWriter?
        Init signature: tf.summary.FileWriter(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None)
        Docstring:
        Writes `Summary` protocol buffers to event files.

        The `FileWriter` class provides a mechanism to create an event file in a
        given directory and add summaries and events to it. The class updates the
        file contents asynchronously. This allows a training program to call methods
        to add data to the file directly from the training loop, without slowing down
        training.
        Init docstring:
        Creates a `FileWriter` and an event file.

        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions:
        @`add_summary()`,
        @`add_session_log()`,
        @`add_event()`, or
        @`add_graph()`.

        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).

        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:

        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```

        The other arguments to the constructor control the asynchronous writes to
        the event file:

        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.

        Args:
          logdir: A string. Directory where event file will be written.
          graph: A `Graph` object, such as `sess.graph`.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
          graph_def: DEPRECATED: Use the `graph` argument instead.
          filename_suffix: A string. Every event file's name is suffixed with
            `suffix`.
        Raises:
          RuntimeError: If called with eager execution enabled.
        """

    def define_graph(self):
        """
        定义计算图谱
        """
        with self.graph.as_default():
            # 这里只是定义图谱中的各种变量
            """
            In [3]: tf.Graph?
            Init signature: tf.Graph()
            Docstring:
            A TensorFlow computation, represented as a dataflow graph.

            A `Graph` contains a set of
            @{tf.Operation} objects,
            which represent units of computation; and
            @{tf.Tensor} objects, which represent
            the units of data that flow between operations.

            A default `Graph` is always registered, and accessible by calling
            @{tf.get_default_graph}.
            To add an operation to the default graph, simply call one of the functions
            that defines a new `Operation`:

            ```python
            c = tf.constant(4.0)
            assert c.graph is tf.get_default_graph()
            ```

            Another typical usage involves the
            @{tf.Graph.as_default}
            context manager, which overrides the current default graph for the
            lifetime of the context:

            ```python
            g = tf.Graph()
            with g.as_default():
              # Define operations and tensors in `g`.
              c = tf.constant(30.0)
              assert c.graph is g
            ```

            Important note: This class *is not* thread-safe for graph construction. All
            operations should be created from a single thread, or external
            synchronization must be provided. Unless otherwise specified, all methods
            are not thread-safe.
            """


            """
            @tf.name_scope：
            In [2]: tf.name_scope?
            Init signature: tf.name_scope(name, default_name=None, values=None)
            Docstring:
            A context manager for use when defining a Python op.

            This context manager validates that the given `values` are from the
            same graph, makes that graph the default graph, and pushes a
            name scope in that graph (see
            @{tf.Graph.name_scope}
            for more details on that).

            For example, to define a new Python op called `my_op`:

            ```python
            def my_op(a, b, c, name=None):
              with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
                a = tf.convert_to_tensor(a, name="a")
                b = tf.convert_to_tensor(b, name="b")
                c = tf.convert_to_tensor(c, name="c")
                # Define some computation that uses `a`, `b`, and `c`.
                return foo_op(..., name=scope)
            ```
            """
            with tf.name_scope('inputs'):
                self.tf_train_samples = tf.placeholder(
                    tf.float32, shape=(self.batch_size, image_size, image_size, num_channels), name='tf_train_samples'
                )
                self.tf_train_labels = tf.placeholder(
                    tf.float32, shape=(self.batch_size, num_labels), name='tf_train_labels'
                )
                self.tf_test_samples = tf.placeholder(
                    tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels),name='tf_test_samples'
                )

            # fully connected layer 1, fully connected
            with tf.name_scope('fc1'):
                fc1_weights = tf.Variable(
                    tf.truncated_normal([image_size * image_size, self.num_hidden], stddev=0.1), name='fc1_weights'
                )
                fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name='fc1_biases')


                # 添加 fc1 中的 fc1_weights和 fc1_biases的条形图
                """
                In [10]: tf.summary.histogram?
                Signature: tf.summary.histogram(name, values, collections=None, family=None)
                Docstring:
                Outputs a `Summary` protocol buffer with a histogram.

                Adding a histogram summary makes it possible to visualize your data's
                distribution in TensorBoard. You can see a detailed explanation of the
                TensorBoard histogram dashboard
                [here](https://www.tensorflow.org/get_started/tensorboard_histograms).

                The generated
                [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
                has one summary value containing a histogram for `values`.

                This op reports an `InvalidArgument` error if any value is not finite.

                Args:
                  name: A name for the generated node. Will also serve as a series name in
                    TensorBoard.
                  values: A real numeric `Tensor`. Any shape. Values to use to
                    build the histogram.
                  collections: Optional list of graph collections keys. The new summary op is
                    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
                  family: Optional; if provided, used as the prefix of the summary tag name,
                    which controls the tab name used for display on Tensorboard.
                """
                tf.summary.histogram('fc1_weights', fc1_weights)
                tf.summary.histogram('fc1_biases', fc1_biases)

            # fully connected layer 2 --> output layer
            with tf.name_scope('fc2'):
                fc2_weights = tf.Variable(
                    tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1), name='fc2_weights'
                )
                fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]), name='fc2_biases')

                tf.summary.histogram('fc2_weights', fc2_weights)
                tf.summary.histogram('fc2_biases', fc2_biases)

            # 再来定义图谱的运算
            def model(data):
                # fully connected layer 1
                shape = data.get_shape().as_list()
                reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])

                # 添加 fc1_model
                with tf.name_scope('fc1_model'):
                    fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
                    hidden = tf.nn.relu(fc1_model)


                # 添加 fc2_model
                # fully connected layer 2
                with tf.name_scope('fc2_model'):
                    output = tf.matmul(hidden, fc2_weights) + fc2_biases
                    return output

            # Training computation.
            logits = model(self.tf_train_samples)

            # 添加 loss
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.tf_train_labels)
                )

                """
                In [11]: tf.summary.scalar?
                Signature: tf.summary.scalar(name, tensor, collections=None, family=None)
                Docstring:
                Outputs a `Summary` protocol buffer containing a single scalar value.

                The generated Summary has a Tensor.proto containing the input Tensor.

                Args:
                  name: A name for the generated node. Will also serve as the series name in
                    TensorBoard.
                  tensor: A real numeric Tensor containing a single value.
                  collections: Optional list of graph collections keys. The new summary op is
                    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
                  family: Optional; if provided, used as the prefix of the summary tag name,
                    which controls the tab name used for display on Tensorboard.

                Returns:
                  A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

                Raises:
                  ValueError: If tensor has the wrong shape or type.
                File:      c:\anaconda3\lib\site-packages\tensorflow\python\summary\summary.py
                Type:      function
                """
                tf.summary.scalar('Loss', self.loss)

            # Optimizer.
            # 添加 optimizer
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            with tf.name_scope('predictions'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
                self.test_prediction = tf.nn.softmax(model(self.tf_test_samples), name='test_prediction')

            """
            In [13]: tf.summary.merge_all?
            Signature: tf.summary.merge_all(key='summaries', scope=None)
            Docstring:
            Merges all summaries collected in the default graph.

            Args:
              key: `GraphKey` used to collect the summaries.  Defaults to
                `GraphKeys.SUMMARIES`.
              scope: Optional scope used to filter the summary ops, using `re.match`

            Returns:
              If no summaries were collected, returns None.  Otherwise returns a scalar
              `Tensor` of type `string` containing the serialized `Summary` protocol
              buffer resulting from the merging.

            Raises:
              RuntimeError: If called with eager execution enabled.
            """
            self.merged = tf.summary.merge_all()

    def run(self):
        """
        用到Session
        """

        # private function
        def print_confusion_matrix(confusionMatrix):
            print('Confusion    Matrix:')
            for i, line in enumerate(confusionMatrix):
                print(line, line[i] / np.sum(line))
            a = 0
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
                print(column[i] / np.sum(column), )
            print('\n', np.sum(confusionMatrix), a)

        with self.session as session:
            tf.global_variables_initializer().run()

            # 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in get_chunk(train_samples, train_labels, chunkSize=self.batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )

                """
                In [15]: a = tf.summary.FileWriter('./log')

                In [16]: a.
                a.add_event        a.add_run_metadata a.close            a.get_logdir
                a.add_graph        a.add_session_log  a.event_writer     a.reopen
                a.add_meta_graph   a.add_summary      a.flush


                In [19]: a.add_summary?
                Signature: a.add_summary(summary, global_step=None)
                Docstring:
                Adds a `Summary` protocol buffer to the event file.

                This method wraps the provided summary in an `Event` protocol buffer
                and adds it to the event file.

                You can pass the result of evaluating any summary op, using
                @{tf.Session.run} or
                @{tf.Tensor.eval}, to this
                function. Alternatively, you can pass a `tf.Summary` protocol
                buffer that you populate with your own data. The latter is
                commonly done to report evaluation results in event files.

                Args:
                  summary: A `Summary` protocol buffer, optionally serialized as a string.
                  global_step: Number. Optional global step value to record with the
                """
                self.writer.add_summary(summary, i)
                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step {0}, is {1:.1f}'.format(i,l))
                    print('Minibatch accuracy:{0}'.format(accuracy))
            #

            # 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
                result = session.run(self.test_prediction,feed_dict={self.tf_test_samples: samples})
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: {0:.1f}'.format(accuracy))
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))
        #

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        """
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        """
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm


if __name__ == '__main__':
    net = Network(num_hidden=128, batch_size=100)
    net.run()
