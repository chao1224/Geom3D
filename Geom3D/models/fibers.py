import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fiber(object):
    """A Handy Data Structure for Fibers"""

    def __init__(
        self,
        num_degrees: int = None,
        num_channels: int = None,
        structure: List[Tuple[int, int]] = None,
        dictionary=None,
    ):
        """
        define fiber structure; use one num_degrees & num_channels OR structure
        OR dictionary

        :param num_degrees: degrees will be [0, ..., num_degrees-1]
        :param num_channels: number of channels, same for each degree
        :param structure: e.g. [(32, 0),(16, 1),(16,2)]
        :param dictionary: e.g. {0:32, 1:16, 2:16}
        """
        if structure:
            self.structure = structure
        elif dictionary:
            self.structure = [(dictionary[o], o) for o in sorted(dictionary.keys())]
        else:
            self.structure = [(num_channels, i) for i in range(num_degrees)]

        self.multiplicities, self.degrees = zip(*self.structure)

        self.structure_dict = {k: v for v, k in self.structure}
        self.dict = self.structure_dict
        self.n_features = np.sum([i[0] * (2 * i[1] + 1) for i in self.structure])
        return

    @staticmethod
    def combine(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] += m
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_max(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] = max(m, new_dict[k])
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    def __repr__(self):
        return f"{self.structure}"


def fiber2head(F, h, structure, squeeze=False):
    if squeeze:
        fibers = [
            F[f"{i}"].view(*F[f"{i}"].shape[:-2], h, -1) for i in structure.degrees
        ]
        fibers = torch.cat(fibers, -1)
    else:
        fibers = [
            F[f"{i}"].view(*F[f"{i}"].shape[:-2], h, -1, 1) for i in structure.degrees
        ]
        fibers = torch.cat(fibers, -2)
    return fibers
