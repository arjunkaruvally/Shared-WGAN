import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

# estimator = tf.estimator.Estimator(model_fn=model_fn)
# # define our data sets
# x_train = np.array([1., 2., 3., 4.])
# y_train = np.array([0., -1., -2., -3.])
# x_eval = np.array([2., 5., 8., 1.])
# y_eval = np.array([-1.01, -4.1, -7, 0.])
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# # train
# estimator.train(input_fn=input_fn, steps=1000)
# # Here we evaluate how well our model did.
# train_metrics = estimator.evaluate(input_fn=train_input_fn)
# eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
# print("train metrics: %r"% train_metrics)
# print("eval metrics: %r"% eval_metrics)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) #Original y values

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))