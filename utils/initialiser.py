from math import floor, sqrt
import random
import torch


def schlichtkrull_std(shape, gain):
    """
    a = \text{gain} \times \frac{3}{\\sqrt{\text{fan\\_in} + \text{fan\\_out}}}
    """
    fan_in, fan_out = shape[0], shape[1]
    return gain * 3.0 / sqrt(float(fan_in + fan_out))

def schlichtkrull_normal_(tensor, shape, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
    std = schlichtkrull_std(shape, gain)
    with torch.no_grad():
        return tensor.normal_(0.0, std)

def schlichtkrull_uniform_(tensor, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a uniform distribution."""
    std = schlichtkrull_std(tensor, gain)
    with torch.no_grad():
        return tensor.uniform_(-std, std)

def select_b_init(init):
    """Return functions for initialising biases"""
    init = init.lower()
    if init in ['zeros', 'zero', 0]:
        return torch.nn.init.zeros_
    elif init in ['ones', 'one', 1]:
        return torch.nn.init.ones_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    elif init == 'normal':
        return torch.nn.init.normal_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')

def select_w_init(init):
    """Return functions for initialising weights"""
    init = init.lower()
    if init in ['glorot-uniform', 'xavier-uniform']:
        return torch.nn.init.xavier_uniform_
    elif init in ['glorot-normal', 'xavier-normal']:
        return torch.nn.init.xavier_normal_
    elif init == 'schlichtkrull-uniform':
        return schlichtkrull_uniform_
    elif init == 'schlichtkrull-normal':
        return schlichtkrull_normal_
    elif init in ['normal', 'standard-normal']:
        return torch.nn.init.normal_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')