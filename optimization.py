#inputs are a prediction vector x
#and true value vector y
import numpy as np
def delta_relu(x):
    n = np.size(x,0)
    r = np.zeros(n)
    for i in range(0,n):
        if x[i] > 0:
            r[i] = 1
    return r

def mean_square_error(x,y):
    # print("the substraction")
    # print(x-y)
    n = np.size(x,0)
    return (1/n)*np.sum((y-x)**2)

def delta_mse(x,y):
    n = np.size(x,0)
    return (-2/n)*(y-x)

def matrix_id_fill(M, n):
    m = np.sqrt(np.size(M))
    if m % 1 == 0:
        m = int(m)
    else:
        print("matrix not square")
        exit()
    A = np.zeros((n,n))
    A[0:m,0:m] = M
    for i in range(m,n):
        A[i,i] = 1
    return A

def cnn_relu_backprop(weights, errors):
    n = np.size(weights, 0)
    deltas = []
    #the deltas for the fc layer
    #10 x 7,7,10 -> 7,7,10
    fc_delta = np.zeros(np.shape(weights[n-1]))
    en = np.size(errors,0)
    for i in range(en):
        fc_delta[i] = weights[n-1][i]*errors[i]
    deltas.append(fc_delta)
    #the layer before the fc layer
    #7,7 x 10,7,7 -> 7,7
    p_delta = weights[n-2]
    for i in range(np.size(fc_delta,0)):
        p_delta = np.dot(p_delta,np.diag(fc_delta[i]))
    deltas.append(p_delta)

    for i in range(1,n-1):
        # print("the output of the current backpropagation layer")
        # print(outputs[n-i])
        # print("matrix product between")
        # print("the weights of the layer")
        # print(weights[n-2-i])
        # print("the deltas of the layer")
        # print(deltas[i])
        m = np.size(weights[n-2-i], 0)
        vi_delta = np.dot(weights[n-2-i], matrix_id_fill(deltas[i], m))
        deltas.append(vi_delta)
    deltas.reverse()
    return deltas

def gradient_descent(weights, deltas, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i]-learning_rate*deltas[i]
    return weights