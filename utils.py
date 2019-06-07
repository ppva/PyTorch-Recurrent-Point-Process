import torch
import numpy as np

def to_one_hot(a, num_classes):
    o_h = torch.zeros(a.shape[0], a.shape[1], num_classes)
    xx,yy = np.ix_(np.arange(a.shape[0]), np.arange(a.shape[1]))
    o_h[xx,yy,a[:,:,0].long()] = 1
    return o_h

def to_one_hot_uni(a, num_classes):
    return torch.abs(torch.eye(num_classes+1)[a.clamp(min=-1).long()])[:,:-1]
