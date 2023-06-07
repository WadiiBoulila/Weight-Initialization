import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import math


# from torch repo
def _to_fan(tensor):
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _custom_weight_initialization(tensor):
    fan_in, fan_out = _to_fan(tensor)
    a = math.sqrt((2)/(fan_in)) + math.sqrt((2)/(fan_out + fan_in))
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def _ZerO_Init(matrix_tensor):
    from scipy.linalg import hadamard
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    return init_matrix


def initialize_weight(model, initialization='xavier'):
    def xavier(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def he(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def custom(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            _custom_weight_initialization(m.weight.data)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def orthogonal(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.orthogonal_(m.weight)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def zerO(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            _ZerO_Init(m.weight.data)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def zeros(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.zeros_(m.weight)
            init.zeros_(m.bias)

    def constant(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.constant_(m.weight, 0.1)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)

    def normal(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.normal_(m.weight)
            init.normal_(m.bias)

    def random(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.uniform_(m.weight)
            init.uniform_(m.bias)

    def sparse(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            sparsity = 0.1 # adjust this value as needed
            std = 0.01 # adjust this value as needed
            init.sparse_(m.weight.data, sparsity=sparsity, std=std)
            if m.bias is not None: # check if bias exists
                init.constant_(m.bias.data, 0)
            

    if initialization == 'xavier':
        model.apply(xavier)
    elif initialization == 'he':
        model.apply(he)
    elif initialization == 'custom':
        model.apply(custom)
    elif initialization == 'orthogonal':
        model.apply(orthogonal)
    elif initialization == 'lecun':
        model.apply(lecun)
    elif initialization == 'zeros':
        model.apply(zeros)
    elif initialization == 'zero':
        model.apply(zerO)
    elif initialization == 'constant':
        model.apply(constant)
    elif initialization == 'normal':
        model.apply(normal)
    elif initialization == 'random':
        model.apply(random)
    elif initialization == 'sparse':
        model.apply(sparse)
    else:
        raise ValueError(f'Invalid initialization argument: {initialization}')





