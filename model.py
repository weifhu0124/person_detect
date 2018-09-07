import tensorflow as tf

# output more logging message
tf.logging.set_verbosity(tf.logging.INFO)

# define the conv net architecture
def conv_net(input):
    # first convolutional layer
    weight1 = tf.get_variable('g_weight1', [5, 5, 3, 32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias1 = tf.get_variable('g_bias1', [32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    tensor1 = tf.nn.conv2d(input, weight1, strides=[1, 1, 1, 1], padding='SAME')
    tensor1 = tensor1 + bias1
    tensor1 = tf.contrib.layers.batch_norm(tensor1, epsilon=1e-5, scope='g_bias1')
    # relu activation
    tensor1 = tf.nn.relu(tensor1)
    # first pool layer
    tensor1 = tf.layers.max_pooling2d(tensor1, pool_size=[2,2], strides=2) # output 24x24

    # second convolutional layer
    weight2 = tf.get_variable('g_weight2', [5, 5, 32, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias2 = tf.get_variable('g_bias2', [64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    tensor2 = tf.nn.conv2d(tensor1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    tensor2 = tensor2 + bias2
    tensor2 = tf.contrib.layers.batch_norm(tensor2, epsilon=1e-5, scope='g_bias2')
    tensor2 = tf.nn.relu(tensor2)
    # second pool layer
    tensor2 = tf.layers.max_pooling2d(tensor2, pool_size=[2,2], strides=2) # output 12x12

    # third convolutional layer
    weight3 = tf.get_variable('g_weight3', [5, 5, 64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias3 = tf.get_variable('g_bias3', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    tensor3 = tf.nn.conv2d(tensor2, weight3, strides=[1, 1, 1, 1], padding='SAME')
    tensor3 = tensor3 + bias3
    tensor3 = tf.contrib.layers.batch_norm(tensor3, epsilon=1e-5, scope='g_bias3')
    tensor3 = tf.nn.relu(tensor3)
    # third pool layer
    tensor3 = tf.layers.max_pooling2d(tensor3, pool_size=[2,2], strides=2) # output 6x6

    # first fully-connected layer
    # flatten pool3
    pool3_flat = tf.reshape(tensor3, [-1, 6*6*128])
    fc1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

    # second fully-connected layer
    fc2 = tf.layers.dense(inputs=fc1, units=2)
    return fc2