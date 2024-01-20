from random import random
import numpy as np
import matplotlib.pyplot as plt
import mnist
import layer
import cnn_functions as f
import optimization as op

x_train, y_train, x_test, y_test = mnist.load()
n = np.size(y_train, 0)
A = np.zeros((n, 10))
for i in range(n):
    A[i, y_train[i]] = 1
y_train = A
x_train = x_train/255

#28x28->14x14->7x7->1x1->1x10
kernel0 = f.circ_conv(np.random.rand(7,7).ravel(), 784)*0.3
kernel1 = f.circ_conv(np.random.rand(3,3).ravel(), 196)*0.3
kernel2 = f.circ_conv(np.random.rand(3,3).ravel(), 196)*0.3
kernel3 = f.circ_conv(np.random.rand(3,3).ravel(), 196)*0.3
kernel4 = f.circ_conv(np.random.rand(3,3).ravel(), 196)*0.3
kernel5 = f.circ_conv(np.random.rand(3,3).ravel(), 196)*0.3
kernel6 = f.circ_conv(np.random.rand(3,3).ravel(), 49)*0.3
kernel7 = f.circ_conv(np.random.rand(3,3).ravel(), 49)*0.3
fc_weight = np.random.rand(10,49)*0.1
weights = [kernel0, kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, fc_weight]
poolings = [2,1,1,1,1,2,1,1]

def mnist_train(x_train, y_train, weights, poolings):
    for i in range(100):
        #pick random sample
        sample_index = np.random.randint(n)
        input = x_train[sample_index]
        label = y_train[sample_index]
        for i in range(10):
            #forward pass
            x, weights = layer.cnn(input, weights, poolings)
            #calculate loss
            errors = op.delta_mse(x, label)
            #backpropagate
            deltas = op.cnn_relu_backprop(weights, errors)
            # for i in range(len(weights)):
            #     print("the weights")
            #     print(weights[i])
            #     print("the deltas")
            #     print(deltas[i])
            #gradient descent
            weights = op.gradient_descent(weights, deltas, 0.2)
            print("The ERROR")
            print(op.mean_square_error(x, label))

        # for i in range(np.size(weights,0)):
        #     print("weights of layer " + str(i))
        #     print(weights[i])
        print("The label")
        print(label)
        print("The prediction")
        print(x)
    return weights

# def mnist_prediction(x_test, y_test, weights):
#     #pick a random sample
#     #run the model with given weights
#     return prediction

mnist_train(x_train, y_train, weights, poolings)