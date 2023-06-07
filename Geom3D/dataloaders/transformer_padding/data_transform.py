# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
from . import algos

import numpy as np
import torch


@torch.jit.script
def convert_to_single_emb(x, offset :int = 16):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x
    # TODO: will clean-ups
    x = x[..., 0:1]
    edge_attr = edge_attr[..., 0:1]
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.idx = 0
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
