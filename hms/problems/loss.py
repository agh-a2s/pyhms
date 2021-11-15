import numpy.linalg as nla
import numpy as np

def norm_loss(x, data, norm_ord):
    if len(x) != len(data):
        raise ValueError("Dimension mismatch")

    return nla.norm(np.array(x) - data, ord=norm_ord)**2

def squared_loss(x, data):
    return norm_loss(x, data, norm_ord=2)

def absolute_loss(x, data):
    return norm_loss(x, data, norm_ord=1)
