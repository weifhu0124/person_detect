import tensorflow as tf

# output more logging message
tf.logging.set_verbosity(tf.logging.INFO)

# define the conv net architecture
def conv_net(features, dropouts, mode, reuse):
    # define and reuse variable scope
    with tf.variable_scope('Conv_net', reuse=reuse):
        # input layer
        input_layer = tf.reshape(features['x'], [-1, 480, 480, 3])

        # first convolution layer
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5, 3],
            padding='same',
            activation=tf.nn.relu
        )
        # first pooling layer
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

        # second convolution layer
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu
        )
        # second pooling layer
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

        # third convolution layer
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=32,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu
        )
        # third pool layer
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

        # first fully-connected layer
        # flatten pool3
        pool3_flat = tf.reshape(pool3, [-1, 60, 60, 32])
        fc1 = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
        # dropouts only happen at training stage
        dropout = tf.layers.dropout(inputs=fc1, rate=dropouts, training= mode==tf.estimator.ModeKeys.TRAIN)

        # second fully-connected layer
        fc2 = tf.layers.dense(inputs=dropout, units=2)
        return fc2