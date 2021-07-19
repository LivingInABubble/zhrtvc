import argparse
from pathlib import Path

import numpy as np

_type_priorities = [  # In decreasing order
    Path,
    str,
    int,
    float,
    bool,
]


def _priority(o):
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None)
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None)
    if p is not None:
        return p
    return len(_type_priorities)


def print_args(args: argparse.Namespace, parser=None):
    dt = args2dict(args, parser)
    print("Arguments:")
    for param, value in dt.items():
        print("{0}: {1}".format(param, value))
    print("")


def args2dict(args: argparse.Namespace, parser=None):
    args = vars(args)
    if parser is None:
        priorities = list(map(_priority, args.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args.keys()))

    indices = np.lexsort((list(args.keys()), priorities))
    items = list(args.items())

    out = {items[i][0]: items[i][1] for i in indices}
    return out


def locals2dict(src: dict, maxlen=255):
    outdt = {}
    for key, value in src.items():
        if isinstance(value, Path):
            outdt[key] = str(value)
        if isinstance(value, str):
            outdt[key] = value[:maxlen]
        elif type(value) in _type_priorities:
            outdt[key] = value
        else:
            outdt[key] = str(value)[:maxlen]

        if 'shape' in dir(value):
            outdt['{}_shape'.format(key)] = str(value.shape)
        elif 'size' in dir(value):
            outdt['{}_size'.format(key)] = str(value.size())
        elif '__len__' in dir(value):
            try:
                outdt['{}_len'.format(key)] = len(value)
            except:
                pass
    return outdt
