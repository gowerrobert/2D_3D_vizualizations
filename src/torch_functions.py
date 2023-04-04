import torch


def Rosenbrock(X):
    a=1
    b=100
    res = torch.sum(b* torch.pow(X[1:] - torch.pow(X[:-1],2) , 2) + torch.pow(a - X[:-1],2)  )
    return res


def Rastrigin(X):
    d = X.shape[0]
    res = 10*d + torch.sum(torch.pow(X,2) - 10 * torch.cos(2*torch.pi*X))
    return res

def IllQuad(X):
    res =  torch.pow(X[1:],2) + 100*torch.pow(X[:-1],2)
    return res
# y = tf.add(tf.pow(tf.subtract(1.0, x[0]), 2.0), 
#            tf.multiply(100.0, tf.pow(tf.subtract(x[1],tf.pow(x[0], 2.0)), 2.0)), 'y')