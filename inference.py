import tensorflow as tf
import model
import util
from sklearn import metrics

# load validation data
data, label = util.load_data_label(train=False)

# restore the session from training
tf.reset_default_graph()
# placeholder for the input images
x_placeholder = tf.placeholder(tf.float32, [None, 48, 48, 3])
pred = model.conv_net(x_placeholder)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'model/cnn_model.ckpt')
    print('model restored')
    pred = sess.run(pred, feed_dict={x_placeholder: data})
    # get the predicted result
    pred_class = sess.run(tf.argmax(pred, axis=1))
    true_class = sess.run(tf.argmax(label, axis=1))

    print(metrics.accuracy_score(y_pred=pred_class, y_true=true_class))