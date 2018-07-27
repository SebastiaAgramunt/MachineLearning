import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parameters
learning_rate = 0.02 #how fast our model learns
training_epochs = 2000 #number of epochs (1 epoch is one pass to all the data)
display_step = 100 

#generate data for linear regression
#function to be fitted
a0 = 1 #bias y(x=0) = a0
a1 = 3 #cieffucuebt of x
a2 = 2 #coeffcient of x^2

def polynomial2(x,a0=a0,a1=a1,a2=a2):
    return a2*np.power(x,2.0)+a1*x+a0

x_data = np.arange(-5,5,0.1)
y_data = np.array([np.random.normal(polynomial2(x)) for x in x_data])
x_features = np.array([[1.0,x,np.power(x,2.0)] for x in x_data]).reshape(-1,3,1)
#we will use x_features and y_data to train the model
#have to reshape to (-1,3,1), this is (n_samples,3,1) then when we loop over
#x_features we'll have elements with shape (3,1).

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32,shape = (3,1),name="X")
    Y = tf.placeholder(tf.float32,name="Y")

W = tf.Variable(tf.truncated_normal((1,3)),name="Omega")

with tf.name_scope('model'):
    Y_pred = tf.matmul(W,X) #(1,3)x(3,1) -> scalar

#and the cost
with tf.name_scope('cost'):
    cost = tf.reduce_sum(tf.pow(Y_pred-Y, 2))/(2*x_features.shape[0])
    to_optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Create a summary to monitor cost tensor
summary_cost = tf.summary.scalar(name="loss",tensor=cost)
# Merge all summaries into a single op

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("./tmp", graph=sess.graph)

    for epoch in range(training_epochs):
        for (x,y) in zip(x_features,y_data):
            sess.run(to_optimize,feed_dict={Y : y,X : x})
            #summary = sess.run(summary_cost,feed_dict={Y : y,X : x})
            #writer.add_summary(summary, epoch)

        if (epoch+1) % display_step == 0:
            c = sess.run(cost,feed_dict={Y : y,X : x})
            print("Epoch: {}, cost = {:.9f}, w = {}".format(epoch+1,c,sess.run(W)))


print("\n\nRunning the analytical solution using sklearn package...")

x_features2 = [x[1:] for x in x_features.reshape(-1,3)]
lr = LinearRegression()
lr.fit(x_features2,y_data)
print("Coefficients: a1,a2")
print(lr.coef_)

print("Intercept: a0")
print(lr.intercept_)

'''
Run tensorboard by executing:
tensorboard --logdir=./tmp

then open the url:
http://192.168.0.49:6006/
'''
