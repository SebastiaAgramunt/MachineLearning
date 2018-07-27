import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.02 #how fast our model learns
training_epochs = 2000 #number of epochs (1 epoch is one pass to all the data)
display_step = 100 

#generate data for linear regression
x_data = np.arange(0,10,0.5)
y_data = np.array([np.random.normal(x) for x in x_data])

#placeholders 
#tf.placeholder(
#    dtype,
#    shape=None,
#    name=None
#)
#Its value must be fed using the feed_dict optional argument to Session.run(), Tensor.eval(), or Operation.run()
#we use placeholders to input the data
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32,name = "X")
    Y = tf.placeholder(tf.float32,name = "Y")

#variables
#tf.Variable(<initial-value>, name=<optional-name>)
#variables will change during the process of backpropagation
W = tf.Variable(np.random.randn(),name = "omega")
b = tf.Variable(np.random.randn(),name = "bias")

#This is the prediction of y
with tf.name_scope('model'):
    Y_pred = tf.add(tf.multiply(W,X),b)

#and the cost
with tf.name_scope('loss'):
    cost = tf.reduce_sum(tf.pow(Y_pred-Y, 2))/(2*x_data.shape[0])

#the trainer
to_optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("tmp", sess.graph)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(x_data, y_data):
            sess.run(to_optimize, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_data, Y: y_data})
            print("Epoch: {}, cost = {:.9f}, w = {}, b = {}".format(epoch+1,c,sess.run(W),sess.run(b)))

    training_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data})
    #Plot
    plt.plot(x_data, y_data, 'ro', label='Data')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Line fitted using gradient descent')
    plt.legend()
    plt.grid()
    plt.show()