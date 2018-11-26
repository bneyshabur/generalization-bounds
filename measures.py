import torch
import math
import copy
import warnings
import torch.nn as nn
import pdb


# This function reparametrizes the networks with batch normalization in a way that it calculates the same function as the
# original network but without batch normalization. Instead of removing batch norm completely, we set the bias and mean
# to zero, and scaling and variance to one
# Warning: This function only works for convolutional and fully connected networks. It also assumes that
# module.children() returns the children of a module in the forward pass order. Recurssive construction is allowed.
def reparam(model, prev_layer=None):
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer


# This function calculates a measure on the given model
# measure_func is a function that returns a value for a given linear or convolutional layer
# calc_measure calculates the values on individual layers and then calculate the final value based on the given operator
def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    measure_val = 0
    if operator == 'product':
        measure_val = math.exp(calc_measure(model, init_model, measure_func, 'log_product', kwargs, p))
    elif operator == 'norm':
        measure_val = (calc_measure(model, init_model, measure_func, 'sum', kwargs, p=p)) ** (1 / p)
    else:
        measure_val = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
                if operator == 'log_product':
                    measure_val += math.log(measure_func(child, init_child, **kwargs))
                elif operator == 'sum':
                    measure_val += (measure_func(child, init_child, **kwargs)) ** p
                elif operator == 'max':
                    measure_val = max(measure_val, measure_func(child, init_child, **kwargs))
            else:
                measure_val += calc_measure(child, init_child, measure_func, operator, kwargs, p=p)
    return measure_val

# calculates l_pq norm of the parameter matrix of a layer:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def norm(module, init_module, p=2, q=2):
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

# calculates l_p norm of eigen values of a layer
# convolutional tensors are reshaped in a way that all dimensions except the output are together
def op_norm(module, init_module, p=float('Inf')):
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()

# calculates l_pq distance of the parameter matrix of a layer from the random initialization:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def dist(module, init_module, p=2, q=2):
    return (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

# calculates l_pq distance of the parameter matrix of a layer from the random initialization with an extra factor that
# depends on the number of hidden units
def h_dist(module, init_module, p=2, q=2):
    return (n_hidden(module, init_module) ** (1 - 1 / q )) * dist(module, init_module, p=p, q=q)

# ratio of the h_dist to the operator norm
def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float('Inf')):
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)

# number of hidden units
def n_hidden(module, init_module):
    return module.weight.size(0)

# depth --> always 1 for any linear of convolutional layer
def depth(module, init_module):
    return 1

# number of parameters
def n_param(module, init_module):
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(module.weight.size(0),-1).size(1)

# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p )).item()


# This function calculates various measures on the given model and returns two dictionaries:
# 1) measures: different norm based measures on the model
# 2) bounds: different generalization bounds on the model
def calculate(trained_model, init_model, device, train_loader, margin, nchannels, nclasses, img_dim):

    model = copy.deepcopy(trained_model)
    reparam(model)
    reparam(init_model)

    # size of the training set
    m = len(train_loader.dataset)

    # depth
    d = calc_measure(model, init_model, depth, 'sum', {})

    # number of parameters (not including batch norm)
    nparam = calc_measure(model, init_model, n_param, 'sum', {})

    measure, bound = {}, {}
    with torch.no_grad():
        measure['L_{1,inf} norm'] = calc_measure(model, init_model, norm, 'product', {'p':1, 'q':float('Inf')}) / margin
        measure['Frobenious norm'] = calc_measure(model, init_model, norm, 'product', {'p':2, 'q':2}) / margin
        measure['L_{3,1.5} norm'] = calc_measure(model, init_model, norm, 'product', {'p':3, 'q':1.5}) / margin
        measure['Spectral norm'] = calc_measure(model, init_model, op_norm, 'product', {'p':float('Inf')}) / margin
        measure['L_1.5 operator norm'] = calc_measure(model, init_model, op_norm, 'product', {'p':1.5}) / margin
        measure['Trace norm'] = calc_measure(model, init_model, op_norm, 'product', {'p':1}) / margin
        measure['L1_path norm'] = lp_path_norm(model, device, p=1, input_size=[1, nchannels, img_dim, img_dim]) / margin
        measure['L1.5_path norm'] = lp_path_norm(model, device, p=1.5, input_size=[1, nchannels, img_dim, img_dim]) / margin
        measure['L2_path norm'] = lp_path_norm(model, device, p=2, input_size=[1, nchannels, img_dim, img_dim]) / margin


        # Generalization bounds: constants and additive logarithmic factors are not included

        # This value of alpha is based on the improved depth dependency by Golowith et al. 2018
        alpha = math.sqrt(d + math.log(nchannels * img_dim * img_dim))

        bound['L1_max Bound (Bartlett and Mendelson 2002)'] = alpha * measure['L_{1,inf} norm'] / math.sqrt(m)
        bound['Frobenious Bound (Neyshabur et al. 2015)'] = alpha * measure['Frobenious norm'] / math.sqrt(m)
        bound['L_{3,1.5} Bound (Neyshabur et al. 2015)'] = alpha * measure['L_{3,1.5} norm'] / ( m ** (1/3))

        beta = math.log(m) * math.log(nparam)
        ratio = calc_measure(model, init_model, h_dist_op_norm,'norm', {'p':2, 'q':1, 'p_op':float('Inf')}, p=2/3)
        bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'] = beta * measure['Spectral norm'] * ratio / math.sqrt(m)

        ratio = calc_measure(model, init_model, h_dist_op_norm,'norm', {'p':2, 'q':2, 'p_op':float('Inf')}, p=2)
        bound['Spec_Fro Bound (Neyshabur et al. 2018)'] =  d * measure['Spectral norm'] * ratio / math.sqrt(m)

    return measure, bound
