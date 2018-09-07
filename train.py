import tensorflow as tf
import model
import util
import numpy as np

# hyperparameters
learning_rate = 0.001
dropouts = 0.4
batch_size = 50
num_steps = 200

# generate batch for trainig
def next_batch(batch_size, data, labels):
    # generating random index
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    # get data from the shuffled index
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

print('Loading data ...')
data, labels = util.load_data_label()

# placeholder for the input images
x_placeholder = tf.placeholder(tf.float32, [None, 48, 48, 3])
# placeholder for labels
y_placeholder = tf.placeholder(tf.float32, [None, 2])
# dropout
keep_prob = tf.placeholder(tf.float32)

# pred_classes hold the predicted class from conv net
pred_classes = model.conv_net(x_placeholder)
# define loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_placeholder, logits=pred_classes)
# define optimizer
optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# start a tf session
print('Start training ...')
# initiate variables
init = tf.global_variables_initializer()
loss_track = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, num_steps):
        data_batch, label_batch = next_batch(batch_size=batch_size, data=data, labels=labels)
        sess.run(optimize, feed_dict={x_placeholder:data_batch, y_placeholder:label_batch, keep_prob:0.8})
        loss_v = sess.run(loss, feed_dict={x_placeholder:data_batch, y_placeholder:label_batch, keep_prob:0.8})
        loss_track.append(loss_v)
        if i % 10 == 0:
            print('Round: ' + str(i))
    print('done')

    print(loss_track)
    saver.save(sess, '/cnn_model1.ckpt/')