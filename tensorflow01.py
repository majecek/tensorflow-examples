# code from http://tensorflow.org/tutorials/mnist/beginners/index.md

import tensorflow as tf
import input_data

# placeholder which describes pixels in image
x = tf.placeholder("float", [None, 784])

#model parameters
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# placeholder for cross-entropy
# correct answer
y_ = tf.placeholder("float", [None,10])

# cross entropy - train our model
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables we created
init = tf.initialize_all_variables()

# run model in session & initialize variables
sess = tf.Session()
sess.run(init)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# let's train and run it 1000x 
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# how well did we go ? 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

