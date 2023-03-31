import torch


def Rosenbrock(X):
    a=1
    b=100
    res = torch.sum(torch.abs(b*(X[1:] - torch.pow(X[:-1],2))**2 + (a - torch.pow(X[:-1],2) ) ))
    return res


# y = tf.add(tf.pow(tf.subtract(1.0, x[0]), 2.0), 
#            tf.multiply(100.0, tf.pow(tf.subtract(x[1],tf.pow(x[0], 2.0)), 2.0)), 'y')