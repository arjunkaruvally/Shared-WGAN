import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# In[2]:

mnist = input_data.read_data_sets("data/", one_hot=True)


learning_rate = 1e-4
epsilon = 1e-4
reg = 0.001

num_steps = 20000
print_every = 100


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def deepnn(x):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv_1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    conv1_out = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    maxpool1_out = max_pool_2x2(conv1_out)

  with tf.name_scope('conv_2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    conv2_out = tf.nn.relu(conv2d(maxpool1_out, W_conv2) + b_conv2)
    maxpool2_out = max_pool_2x2(conv2_out)

  with tf.name_scope('fc_1'):
    keep_prob = tf.placeholder(tf.float32)
    W1 = weight_variable([7 * 7 * 64, 1024])
    b1 = bias_variable([1024])

    maxpool2_flat = tf.reshape(maxpool2_out, [-1, 7*7*64])
    l1_out = tf.nn.relu(tf.matmul(maxpool2_flat, W1) + b1)
    l1_drop = tf.nn.dropout(l1_out, keep_prob)

  with tf.name_scope('fc_2'):
    W2 = weight_variable([1024, 10])
    b2 = bias_variable([10])

    l2_out = tf.matmul(l1_drop, W2) + b2

  return l2_out, keep_prob, W_conv1
  

def run_model():
  x = tf.placeholder(tf.float32, [None, 784], name="input")
  y = tf.placeholder(tf.float32, [None, 10], name="data_output")

  conv_out, keep_prob, W_conv1 = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=conv_out))

  train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('evaluate'):
    correct_prediction = tf.equal(tf.argmax(conv_out, 1), tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # print "initial accuracy: ", acc

  loss_history = []
  validation_accuracy_history = []
  training_accuracy_history = []
  run_loss = 0

  graph_location = "log/"
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(num_steps):

        batch_xs, batch_ys = mnist.train.next_batch(50)

        if step%print_every == 0:
            runv_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0})
            runt_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

            print "Iteration(",step,"/",num_steps,") loss: ", run_loss," train_accuracy: ",runt_acc," validation_accuracy: ",runv_acc
            
            validation_accuracy_history.append(runv_acc)
            training_accuracy_history.append(runt_acc)

        # w_conv1 = sess.run([W_conv1])

        # print w_conv1

        # _ = sess.run([train], { x: batch_xs, y:batch_ys, keep_prob: 0.5 })
        train.run({ x: batch_xs, y:batch_ys, keep_prob: 0.5 })

    save_path = saver.save(sess, "checkpoints/classifier.ckpt")
    print("Model saved in file: %s" % save_path)

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})

    W1_final = sess.run(W_conv1)

    print "test accuracy: ",acc

    print "complete"


  # In[37]:

  # plt.imshow(mnist.train.images[0].reshape([28, 28]), cmap="gray")
  # plt.show()

  # fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')

  # print W1_final.shape

  # for i in range(W1_final.shape[3]):
  #     plt.subplot(8, 8, i+1)
      
  #     plt.imshow(W1_final[:, :, 0, i].reshape([5, 5]), cmap="gray")

  # plt.show()

  # plt.plot(range(num_steps), loss_history)/

  # plt.show()

  plt.plot(range(len(training_accuracy_history)), training_accuracy_history, color='b')
  plt.plot(range(len(validation_accuracy_history)), validation_accuracy_history, color='r')

  plt.show()
  # In[31]: