import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_batches(X,Y,batch_size):
    '''
    Get batches of data, input is features,target
    and batch_size, output is one minibatch
    '''
    iters = X.shape[0]//batch_size
    for i in range(0,iters):
        yield X[i*batch_size:(i+1)*batch_size],Y[i*batch_size:(i+1)*batch_size]

def new_conv2d(input,n_input_channels,filter_size,n_filters,name = "conv2d",stride=1):
    '''
    Create a convolutional layer
    Input:
    =======
        * input: The image or batch of images [-1,width,height,n_input_channels]
        * n_input_channels: number of colours of our image
        * filter_size: the size of the filter we want to apply
        * n_filters: number of filters for this convolutional layer
        * name: name of the scope
        * stride of the window
    Output:
    =======
        * The convolved layer and the weights (not including the bias)
    '''
    with tf.variable_scope(name):
        shape = [filter_size, filter_size, n_input_channels, n_filters]

        # Weights
        W = tf.Variable(tf.truncated_normal(shape))

        # Biases (one for each filter)
        b = tf.Variable(tf.random_normal([n_filters]),name="b1")

        # Convolution using tensorflow function
        layer = tf.nn.conv2d(input=input,
                             filter=W,
                             strides=[1, stride, stride, 1],
                             padding='SAME')

        # Add bias to the convolution
        layer = tf.nn.bias_add(layer, b)

        return layer
    
def new_pool(input, name,ksize = 2,stride = 2):
    '''
    Create a max-pool layer
    Input:
    ======
        * input: the convolved layer
        * name: name of the scope
        * stride: stride of the window
        * ksize: size of the window
    '''
    with tf.variable_scope(name):
        # Operation for max pool, ksize indicate the size of the window to convovle
        # and stride how we move such window
        layer = tf.nn.max_pool(value=input,
                               ksize=[1, ksize, ksize, 1],
                               strides=[1, stride, stride, 1],
                               padding='SAME')
        
        return layer

def new_relu(input, name):
    '''
    Apply relu function
    '''
    with tf.variable_scope(name):
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer
    
def new_fully_connected(input,flattened_size,n_nodes,name):
    '''
    Create new connected layer
    Input:
    ======
        * input: the already flattened layer
        * flattened_size: the size of the tensor when flatened
        * n_nodes: number of new nodes
        * name: name of the scope
    '''
    with tf.variable_scope(name):
        #input is already flattened
        shape = [flattened_size,n_nodes]
        
        # Weights
        W = tf.Variable(tf.truncated_normal(shape))
        #bias
        b = tf.Variable(tf.truncated_normal([n_nodes]))

        return tf.matmul(input, W) + b


# Training Parameters
learning_rate = 0.001
epochs = 800
batch_size = 512
display_step = 10

# Network Parameters
num_input = mnist.train.images.shape[1] # input = 28*28 = 784
num_classes = mnist.train.labels.shape[1] # number of classes = 10
dropout = 0.75 # Dropout probability

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes]) 
X_image = tf.reshape(X, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)


# I explain shapes assuming one sample at a time [1,28,28,1] instead of [N,28,28,1]
#first layer X_image = tf.shape([1,28,28,1]) --> tf.shape(layer1) = [1,28,28,16]

#Layer 1
#======================================================
layer1 = new_conv2d(X_image,
                    n_input_channels =1,
                    filter_size=5,
                    n_filters = 16,
                    name = "layer1")
#This op preserves the shape
layer1 = new_relu(layer1,"layer1_relu")
# input is tf.shape() = [1,28,28,16] --> tf.shape() = [1,14,14,16]
layer1 = new_pool(layer1, name = "layer1_relu_pooling")

#Layer 2
#======================================================
#input is tf.shape() = [1,14,14,16] --> tf.shape() = [1,14,14,64]
layer2 = new_conv2d(layer1,
                    n_input_channels =16,
                    filter_size=5,
                    n_filters = 64,
                    name = "layer2")
#This op preserves the shape
layer2 = new_relu(layer2,"layer2_relu")
# input is tf.shape() = [1,14,14,64] --> tf.shape() = [1,7,7,64]
layer2 = new_pool(layer2, name = "layer2_relu_pooling")
# input is tf.shape() = [1,7,7,64] --> tf.shape() = [1,7*7*64]
layer2 = tf.reshape(layer2,shape=[-1,7*7*64]) #-1 for all the rest

#Layer 3
#======================================================
# input is tf.shape() = [1,7*7*64] --> [1,1024]
layer3 = new_fully_connected(layer2,7*7*64,1024,"fully_connected")
layer3 = new_relu(layer3,"fully_connected_relu")
layer3 = tf.nn.dropout(layer3, dropout)

#Output layer
#======================================================
output_layer = new_fully_connected(layer3,1024,10,"output_layer")
#output_layer = tf.nn.softmax(output_layer)
    

# Prediction from the output layer
prediction = tf.nn.softmax(output_layer)

# definition of loss and how to minimize
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=output_layer, labels=Y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# metrics
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        data = get_batches(X=mnist.train.images,Y=mnist.train.labels,batch_size=batch_size)
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        for x_,y_ in data:
            sess.run(train_step,feed_dict = {X:x_,Y:y_,keep_prob: dropout})

        if epoch % display_step == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_,
                                                                 Y: y_,
                                                                 keep_prob: 1.0})
            print("Epoch " + str(epoch) + ", loss= " + \
                  "{:.4f}".format(loss) + ", accuracy= " + \
                  "{:.3f}".format(acc))

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[0:256],
                                      Y: mnist.test.labels[0:256],
                                      keep_prob: 1.0}))