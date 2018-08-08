import tensorflow as tf


class ConvolutionalLayer(object):
    """convolutional layer"""""
    def __init__(self, kernel_shape, layer_name, activation="relu"):

        self.kernel_shape = kernel_shape
        self.layer_name = layer_name
        self.activation = activation

        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable("weights", self.kernel_shape,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.biases = tf.get_variable("biases", [self.kernel_shape[3]],
                                          initializer=tf.constant_initializer(0.1))
            tf.summary.histogram("weights", self.weights)
            tf.summary.histogram("biases", self.biases)

    def __call__(self, input_tensor):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            conv_out = tf.nn.conv2d(input_tensor, self.weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
            if self.activation == "relu":
                output_tensor = tf.nn.relu(conv_out + self.biases)
            elif self.activation == "leaky_relu":
                output_tensor = tf.maximum(conv_out + self.biases, 0.01 * (conv_out + self.biases))
            else:
                output_tensor = tf.identity(conv_out + self.biases)
            return output_tensor


class FullyConnectedLayer(object):
    """fully connected layer"""
    def __init__(self, weights_shape, activation, layer_name):

        self.activation = activation
        self.layer_name = layer_name
        self.weights_shape = weights_shape

        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable("weights", self.weights_shape,
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.biases = tf.get_variable("biases", [self.weights_shape[1]],
                                          initializer=tf.constant_initializer(0.1))
            tf.summary.histogram("weights", self.weights)
            tf.summary.histogram("biases", self.biases)

    def __call__(self, input_tensor):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            mult_out = tf.matmul(input_tensor, self.weights)
            if self.activation == "relu":
                output_tensor = tf.nn.relu(mult_out + self.biases)
            elif self.activation == "tanh":
                output_tensor = tf.tanh(mult_out + self.biases)
            else:
                output_tensor = tf.identity(mult_out + self.biases)
            return output_tensor


class MaxPoolingLayer(object):
    """Pooling Layer"""
    def __init__(self, layer_name,
                 ksize=[1, 2, 2, 1],
                 strides=[1, 2, 2, 1]):
        self.ksize = ksize
        self.strides = strides
        self.layer_name = layer_name

    def __call__(self, input_tensor):
        return tf.nn.max_pool(input_tensor,
                              ksize=self.ksize,
                              strides=self.strides,
                              padding='SAME',
                              name=self.layer_name)


class BatchNormalizationLayer(object):

    def __init__(self, output_channels, layer_name):

        self.layer_name = layer_name

        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            self.beta = tf.Variable(tf.constant(0.0, shape=[output_channels]),
                                    name='beta', trainable=True)
            self.gamma = tf.Variable(tf.constant(1.0, shape=[output_channels]),
                                     name='gamma', trainable=True)
            tf.summary.histogram("beta", self.beta)
            tf.summary.histogram("gamma", self.gamma)

            self.ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def __call__(self, input_tensor, phase_train, over_dim=[0,1,2]):

        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):

            def mean_var_with_update():
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            batch_mean, batch_var = tf.nn.moments(input_tensor, over_dim, name='moments')
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (self.ema.average(batch_mean), self.ema.average(batch_var)))
            output_tensor = tf.nn.batch_normalization(input_tensor, mean, var, self.beta, self.gamma, 1e-3)
            return output_tensor


class ResidualLayer(object):

    def __init__(self, kernel_shape, layer_name, activation="relu"):

        self.kernel_shape = kernel_shape
        self.layer_name = layer_name
        self.activation = activation

        with tf.variable_scope("var_"+self.layer_name):
                self.weights = tf.get_variable("weights", self.kernel_shape,
                                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
                self.biases = tf.get_variable("biases", [self.kernel_shape[3]],
                                              initializer=tf.constant_initializer(0.1))
                tf.summary.histogram("weights", self.weights)
                tf.summary.histogram("biases", self.biases)

    def __call__(self, input_tensor):
        with tf.variable_scope("op_"+self.layer_name):
            conv_out = tf.nn.conv2d(input_tensor, self.weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
            if self.activation == "relu":
                output_tensor = tf.nn.relu(conv_out + self.biases)
            elif self.activation == "leaky_relu":
                output_tensor = tf.maximum(conv_out + self.biases, 0.01 * (conv_out + self.biases))
            else:
                output_tensor = tf.identity(conv_out + self.biases)
            return output_tensor + input_tensor


class LSTMLayer(object):

    def __init__(self, hidden_units, layer_name, activation="tanh"):
        self.hidden_units = hidden_units
        self.layer_name = layer_name

        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            if activation == "tanh":
                act_func = tf.nn.tanh
            elif activation == "relu":
                act_func = tf.nn.relu
            else:
                act_func = None
            self.lstm = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units,
                                                     state_is_tuple=False,
                                                     activation=act_func)
            print(self.lstm.variables)
            tf.summary.histogram("weights", self.lstm.trainable_weights)

    def __call__(self, input_tensor, state):
        with tf.variable_scope(self.layer_name, reuse=tf.AUTO_REUSE):
            output_tensor, output_state = self.lstm(input_tensor, state)
            return output_tensor, output_state