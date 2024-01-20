import numpy as np
from scipy import sparse
#relu
def matrix_relu(x):
    n = np.size(x, 0) 
    relu_index = np.zeros(n)
    for i in range(0,n):
        if np.max([0, x[i]]):
            relu_index[i] = 1
    return np.diag(relu_index)

#for a single neuron
def last_fc(x, weights):
    y = x*weights
    return y
    
#convolution matrix for network layers
#kernel is a flattened 2d kernel, i.e a vector
def circ_conv(kernel, n):
    k = int(np.sqrt(np.size(kernel, 0)))
    n_sqrt = int(np.sqrt(n))
    #create circular conv matrix according to kernel
    circ_matrix = np.zeros((n,n))
    circ_kernel = np.zeros(n)
    j = 0
    l = 0
    for i in range(0, k):
        circ_kernel[j:j+k] = kernel[l:l+k]
        j = j + n_sqrt
        l = l + k
    for i in range(0,n):
        circ_matrix[i] = circ_kernel
        circ_kernel = np.roll(circ_kernel, 1)
    return circ_matrix

def circ_conv_ds(kernel, n, stride=1):
    k = int(np.sqrt(np.size(kernel, 0)))
    n_sqrt = int(np.sqrt(n))
    #create circular conv matrix according to kernel
    circ_matrix = np.zeros((n,n))
    circ_kernel = np.zeros(n)
    j = 0
    l = 0
    for i in range(0, k):
        circ_kernel[j:j+k] = kernel[l:l+k]
        j = j + n_sqrt
        l = l + k
    for i in range(0,n):
        circ_matrix[stride*i] = circ_kernel
        circ_kernel = np.roll(circ_kernel, stride)
    return circ_matrix

#input is a square
#output dimension is divided by d
def max_pool(x, d=2):
    n = np.size(x,0)
    n_sqrt = int(np.sqrt(n))
    p = int(n_sqrt/d)**2
    pool_matrix = np.zeros([p,n])
    circ_pool = np.zeros([n])

    #create a vector which acts as a circular filter
    #for the values of input which are considered for the pool
    for j in range(0,d):
        circ_pool[j*n_sqrt:j*n_sqrt+d] = 1

    roll = 0
    #for the pooling matrix compare the values of input
    #chosen by the circular filter and filter only
    #the max value. stride is equal to filter dim
    for i in range(0,p):
        pool_matrix[i,np.argmax(np.multiply(x,circ_pool))] = 1
        if np.argmax(circ_pool) % n_sqrt == 0:
            roll = d
        else:
            roll = (d-1)*n_sqrt + d
        circ_pool = np.roll(circ_pool,roll)

    return pool_matrix