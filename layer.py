#convolutional neural network layer
#R^n -> R^m
import numpy as np
import cnn_functions as f

def cnn_layer(input, weight_matrix, pooling=2, avg_pooling = False):
    n = np.size(input,0)

#    weight_matrix = f.circ_conv(weight_matrix, n)
    V = np.matmul(weight_matrix, input)

    relu_matrix = f.matrix_relu(V)
    weight_matrix = np.matmul(relu_matrix, weight_matrix)
    V = np.matmul(relu_matrix, V)

    # print("the weight matrix before pooling")
    # print(weight_matrix)
    # print("the output")
    # print(V)
    #get the pool matrix and set weights to zero
    #on the weight matrix for pooled entries
    #and pool the V
    if pooling > 1:
        pool_matrix = f.max_pool(V, pooling)
        pool_indices = np.nonzero(pool_matrix)[1]
        pool_nonzeros = np.zeros(n)
        pool_nonzeros[pool_indices] = 1
        weight_matrix = np.matmul(np.diag(pool_nonzeros), weight_matrix)
        V = np.matmul(pool_matrix, V)
    
#    if avg_pooling:
#        weight_matrix

    # print("the weight matrix after pooling")
    # print(weight_matrix)
    # print("the output")
    # print(V)

    return (weight_matrix, V)

#average pool and then fc layer
#7,7 x 7,7,10 -> 10 
def final_fc_layer(input, weights):
    n = np.size(weights,0)
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.sum(input*weights[i])
    relu_matrix = f.matrix_relu(y)
    return np.matmul(relu_matrix, y)

def cnn(input, weights, poolings):
    x = input
    cnn_weights = weights[0:-1]
    fc_weight = weights[-1]
    #loop through the convolution layers
    for i in range(len(cnn_weights)):
        cnn_weights[i], x = cnn_layer(x,cnn_weights[i],poolings[i])
    #fc layer
    x = final_fc_layer(x, fc_weight)
    cnn_weights.append(fc_weight)
    return x, cnn_weights
