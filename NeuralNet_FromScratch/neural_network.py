import numpy as np
import time

def sigmoid(x,deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1 / (1 + np.exp(-x))

def softmax(s):
    exps = np.exp(s)
    return exps/np.sum(exps)

def multiclass_cross_entropy(y_hat, y,epsilon=1e-12):
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    N = y_hat.shape[0]
    ce = -np.sum(np.sum(y*np.log(y_hat)))/N
    return ce

def err(pred, real):
    '''
    Difference btw predicted and real 
    '''
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def get_batches(X,Y,batch_size):
    '''
    Get batches of data, input is features,target
    and batch_size, output is one minibatch
    '''
    iters = X.shape[0]//batch_size
    for i in range(0,iters):
        yield X[i*batch_size:(i+1)*batch_size],Y[i*batch_size:(i+1)*batch_size]

class NeuralNetwork:
    
    def __init__(self, 
                 input_size,
                 output_size, 
                 n_neurons):

        #three layer neural network
        self.w1 = np.random.randn(input_size, n_neurons)
        self.b1 = np.random.randn(1, n_neurons)           
        self.w2 = np.random.randn(n_neurons, n_neurons)
        self.b2 = np.random.randn(1, n_neurons)
        self.w3 = np.random.randn(n_neurons, output_size)
        self.b3 = np.random.randn(1, output_size)

    def ForwardPass(self,x):
        self.x = x
        self.o1 = sigmoid(np.dot(self.x,self.w1) + self.b1)
        self.o2 = sigmoid(np.dot(self.o1,self.w2) + self.b2)
        self.o3 = sigmoid(np.dot(self.o2,self.w3) + self.b3) #output of nn
        return self.o3

    def BackPropagation(self,y,learning_rate,verbose = False):
        if verbose:
            print('Cross_Entropy :{}'.format(multiclass_cross_entropy(self.o3, y)))

        o3_delta = err(self.o3, y)
        z2_delta = np.dot(o3_delta, self.w3.T)
        o2_delta = z2_delta * sigmoid(self.o2,deriv=True)
        z1_delta = np.dot(o2_delta, self.w2.T)
        o1_delta = z1_delta * sigmoid(self.o1,deriv=True)

        self.w3 -= learning_rate * np.dot(self.o2.T, o3_delta)
        self.b3 -= learning_rate * np.sum(o3_delta, axis=0, keepdims=True)
        self.w2 -= learning_rate * np.dot(self.o1.T, o2_delta)
        self.b2 -= learning_rate * np.sum(o2_delta, axis=0)
        self.w1 -= learning_rate * np.dot(self.x.T, o1_delta)
        self.b1 -= learning_rate * np.sum(o1_delta, axis=0)

    def TrainStep(self,x,y,learning_rate = 0.01):
        
        self.ForwardPass(x)
        self.BackPropagation(y, learning_rate)
        
    def Predict(self,x):

        self.ForwardPass(x)
        return self.o3

    def Accuracy(self,X,Y):
        accuracy = 0
        for x,y in zip(X,Y):
            y_hat = self.Predict(x)
            if np.argmax(y_hat) == np.argmax(y):
                accuracy+=1
        return accuracy/X.shape[0]

if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    #get the digits dataframe
    digits = load_digits()
    
    #targets are numbers, we must one-hot encode.
    onehot_target = np.zeros((digits.target.shape[0],10)) #10 is number of classes
    onehot_target[np.arange(digits.target.shape[0]), digits.target] = 1

    #splitting data on training and test
    x_train, x_test, y_train, y_test = train_test_split(digits.data/16.0,
                                                        onehot_target,
                                                        test_size=0.20,
                                                        random_state=50)

    #neural network parameters
    learning_rate = 0.5
    n_neurons = 256
    epochs = 100
    input_size = 64
    output_size = 10

    nn = NeuralNetwork(input_size=input_size, #size of the images 
                       output_size=output_size, #size of the output 10 classes
                       n_neurons=n_neurons) #number of neurons

    loss = list()
    start_time = time.time()
    for i in range(epochs):
        data = get_batches(X=x_train,Y=y_train,batch_size=20)
        for x,y in data:
            y = np.array(y)
            nn.TrainStep(x,y,learning_rate=learning_rate)
            loss.append(multiclass_cross_entropy(nn.ForwardPass(x),y))
    end_time = time.time()

    print("Training time: {0:.2f} seconds".format(end_time-start_time))
    print("Accuracy on the train: {}".format(nn.Accuracy(x_train,y_train)))
    print("Accuracy on the test: {}".format(nn.Accuracy(x_test,y_test)))
    
    plt.plot(loss)
    plt.show()

