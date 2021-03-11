import torch as th
from torch.nn import functional as F


def bell_loss(y_hat, y):
    sigma = 9
    gamma = 300
    error = th.square(y_hat - y)
    scale = 2 * th.square(sigma)
    return th.mean(gamma * (1 - th.exp(-(error / scale))))


def bell_mse_mae_loss(y_hat, y):
    return bell_loss(y_hat, y) + F.mse_loss(y_hat, y) + F.l1_loss(y_hat, y)
