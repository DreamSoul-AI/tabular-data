import inspect
import torch
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import repeat


def filter_args(func, arg_dict):
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in arg_dict.items() if k in sig.parameters}
    return valid_args


def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(input, (str, bytes)):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_recursively(fn, input, *args, apply_condition, identity_condition=None, key='root'):
    if apply_condition(input):
        sig = inspect.signature(fn)
        if 'key' in sig.parameters:
            return fn(input, *args, key=key)
        else:
            return fn(input, *args)
    elif identity_condition is not None and identity_condition(input):
        return input
    elif isinstance(input, Mapping):
        return {key: apply_recursively(fn, value, *args, apply_condition=apply_condition,
                                       identity_condition=identity_condition, key='{},{}'.format(key, key_)) for
                key_, value in input.items()}
    elif isinstance(input, Iterable) and not isinstance(input, (str, bytes)):
        return [apply_recursively(fn, item, *args, apply_condition=apply_condition,
                                  identity_condition=identity_condition, key='{},{}'.format(key, i))
                for i, item in enumerate(input)]
    else:
        raise ValueError('Not valid input type: {} with value {}'.format(type(input), input))


def to_device(input, device):
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    fn = lambda x, y: x.to(y)
    output = apply_recursively(fn, input, device,
                               apply_condition=apply_condition, identity_condition=identity_condition)
    return output


def gather_input(data_loader):
    input = {}
    for i, input_i in enumerate(data_loader):
        for key, value in input_i.items():
            if key not in input:
                input[key] = []
            input[key].append(value)
    for key in input:
        input[key] = torch.cat(input[key], dim=0)
    return input


def tree():
    return defaultdict(tree)


def process_input(input):
    processed_input = tree()
    for key in input:
        split_names = key.split('-')
        current = processed_input
        for split_name in split_names[:-1]:
            current = current[split_name]
        current[split_names[-1]] = input[key]
    processed_input = dict(processed_input)
    return processed_input
