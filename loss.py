import torch as th


def bell_loss(y_hat, y):
    sigma = 9
    gamma = 300
    error = th.square(y_hat - y)
    scale = 2 * th.square(sigma)
    return th.mean(gamma * (1 - th.exp(-(error / scale))))
