# from https://github.com/csinva/hierarchical-dnn-interpretations/tree/master/acd 
from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import expit as sigmoid
#from .cd_architecture_specific import *

from torch import tanh

def propagate_conv_linear(relevant, irrelevant, module):
    '''Propagate convolutional or linear layer
    Apply linear part to both pieces
    Split bias based on the ratio of the absolute sums
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel) + 1e-20  # add a small constant so we don't divide by 0
    prop_irrel = torch.abs(irrel) + 1e-20  # add a small constant so we don't divide by 0
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


def propagate_batchnorm2d(relevant, irrelevant, module):
    '''Propagate batchnorm2d operation
    '''
    device = relevant.device
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_rel[torch.isnan(prop_rel)] = 0
    rel = rel + torch.mul(prop_rel, bias)
    irrel = module(relevant + irrelevant) - rel
    return rel, irrel


def propagate_pooling(relevant, irrelevant, pooler, model_type):
    '''propagate pooling operation
    '''
    if model_type == 'mnist':
        unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        window_size = 4
    elif model_type == 'vgg':
        unpool = torch.nn.MaxUnpool2d(kernel_size=pooler.kernel_size, stride=pooler.stride)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=(pooler.kernel_size, pooler.kernel_size),
                                        stride=(pooler.stride, pooler.stride), count_include_pad=False)
        window_size = 4

    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)

    # relevant
    rel = mask_both * relevant
    rel = avg_pooler(rel) * window_size

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = avg_pooler(irrel) * window_size
    return rel, irrel


def propagate_independent(relevant, irrelevant, module):
    '''use for things which operate independently
    ex. avgpool, layer_norm, dropout
    '''
    return module(relevant), module(irrelevant)


def propagate_relu(relevant, irrelevant, activation):
    '''propagate ReLu nonlinearity
    '''
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


def propagate_three(a, b, c, activation):
    '''Propagate a three-part nonlinearity
    '''
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


def propagate_tanh_two(a, b):
    '''propagate tanh nonlinearity
    '''
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))


def cd(im_torch: torch.Tensor, model, mask, model_type='mnist', device='cuda'):
    '''Get contextual decomposition scores for some set of inputs for a specific image
    
    Params
    ------
    im_torch: torch.Tensor
        example to interpret - usually has shape (batch_size, num_channels, height, width)
    model: pytorch model        
    mask: array_like (values in {0, 1})
        required unless transform is supplied
        array with 1s marking the locations of relevant pixels, 0s marking the background
        shape should match the shape of im_torch or just H x W        
    model_type: str, optional
        usually should just leave this blank
        if this is == 'mnist', uses CD for a specific mnist model
        if this is == 'resnet18', uses resnet18 model
    device: str, optional
    transform: function, optional
        transform should be a function which transforms the original image to specify rel
        only used if mask is not passed
        
    Returns
    -------
    relevant: torch.Tensor
        class-wise scores for relevant mask
    irrelevant: torch.Tensor
        class-wise scores for everything but the relevant mask 
    '''
    # set up model
    model.eval()
    im_torch = im_torch.to(device)

    # set up relevant/irrelevant based on mask
    mask = mask.float().to(device)
    relevant = mask * im_torch
    irrelevant = (1 - mask) * im_torch

    # deal with specific architectures which cannot be handled generically
    if model_type == 'mnist':
        mods = list(model.modules())[1:]
        relevant, irrelevant = cd_generic(mods, relevant, irrelevant, model_type)
        # return cd_propagate_mnist(relevant, irrelevant, model)
    elif model_type == 'resnet18':
        pass
        # return cd_propagate_resnet(relevant, irrelevant, model)

    # try the generic case
    else:
        mods = list(model.modules())[1:]
        relevant, irrelevant = cd_generic(mods, relevant, irrelevant, model_type)
    return relevant, irrelevant


def cd_generic(mods, relevant, irrelevant, model_type):
    '''Helper function for cd which loops over modules and propagates them 
    based on the layer name
    '''
    for i, mod in enumerate(mods):
        t = str(type(mod))
        if 'Conv2d' in t:
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'Linear' in t:
            relevant = relevant.view(relevant.size(0), -1)
            irrelevant = irrelevant.view(irrelevant.size(0), -1)
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
        elif 'ReLU' in t:
            relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)
        elif 'AvgPool' in t or 'NormLayer' in t or 'Dropout' in t \
                or 'ReshapeLayer' in t or ('modularize' in t and 'Transform' in t):  # custom layers
            relevant, irrelevant = propagate_independent(relevant, irrelevant, mod)
        elif 'MaxPool2d' in t:
                relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod, model_type)
        elif 'BatchNorm2d' in t:
            relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mod)
    return relevant, irrelevant
