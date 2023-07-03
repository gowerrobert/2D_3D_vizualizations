import torch


def Rosenbrock(X):
    a = 1
    b = 100
    res = torch.sum(
        b * torch.pow(X[1:] - torch.pow(X[:-1], 2), 2)
        + torch.pow(a - X[:-1], 2)
    )
    return res


def Rastrigin(X):
    d = X.shape[0]
    res = 10 * d + torch.sum(
        torch.pow(X, 2) - 10 * torch.cos(2 * torch.pi * X)
    )
    return res


def IllQuad(X):
    res = torch.pow(X[1:], 2) + 100 * torch.pow(X[:-1], 2)
    return res


# y = tf.add(tf.pow(tf.subtract(1.0, x[0]), 2.0),
#            tf.multiply(100.0, tf.pow(tf.subtract(x[1],tf.pow(x[0], 2.0)), 2.0)), 'y')


def Booth(X):
    res = (X[:-1] + 2 * X[1:] - 7) ** 2 + (2 * X[:-1] + X[1:] - 5) ** 2
    return res


def BukinN6(X):
    res = 100 * torch.sqrt(
        torch.abs(X[1:] - 0.01 * X[:-1] ** 2)
    ) + 0.01 * torch.abs(X[:-1] + 10)

    return res


def GoldsteinPrice(X):
    x, y = X[:-1], X[1:]
    res = 1 + (x + y + 1) ** 2 * (
        19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
    )
    res *= 30 + (2 * x - 3 * y) ** 2 * (
        18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
    )
    return res


def Himmelblau(X):
    x, y = X[:-1], X[1:]
    res = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
    return res
