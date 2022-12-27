import torch
from torch import nn
import torch.nn.init as init
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


def initialize_weight(model, initialization='xavier'):
    def xavier(m):
        if isinstance(m, nn.Linear) and isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def he(m):
        if isinstance(m, nn.Linear) and isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def custom(m):
        if isinstance(m, nn.Linear) and isinstance(m, nn.Conv2d):
            _custom_weight_initialization(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    if initialization == 'xavier':
        model.apply(xavier)
    elif initialization == 'he':
        model.apply(he)
    else:
        model.apply(custom)