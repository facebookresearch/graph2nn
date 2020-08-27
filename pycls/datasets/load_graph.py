#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""load bio neural networks"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.utils import py_random_state
from matplotlib.colors import ListedColormap

import pdb


def compute_stats(G):
    G_cluster = sorted(list(nx.clustering(G).values()))
    cluster = sum(G_cluster) / len(G_cluster)
    path = nx.average_shortest_path_length(G)  # path
    return cluster, path


def plot_graph(graph, name, dpi=200, width=0.5, layout='spring'):
    plt.figure(figsize=(10, 10))
    pos = nx.spiral_layout(graph)
    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    nx.draw(graph, pos=pos, node_size=100, width=width)
    plt.savefig('figs/graph_view_{}.png'.format(name), dpi=dpi, transparent=True)



def load_graph(name, verbose=False, seed=1):
    if 'raw' in name:
        name = name[:-3]
        directed = True
    else:
        directed = False
    filename = '{}.txt'.format(name)
    # filename = 'pycls/datasets/{}.txt'.format(name)
    with open(filename) as f:
        content = f.readlines()
    content = [list(x.strip()) for x in content]
    adj = np.array(content).astype(int)
    if not directed:
        adj = np.logical_or(adj.transpose(), adj).astype(int)

    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    if verbose:
        print(type(graph))
        print(graph.number_of_nodes(), graph.number_of_edges())
        print(compute_stats(graph))
        print(len(graph.edges))
        # plot_graph(graph, 'mc_whole', dpi=60, width=1, layout='circular')
        cmap = ListedColormap(['w', 'k'])
        plt.matshow(nx.to_numpy_matrix(graph), cmap=cmap)
        plt.show()
    return graph


def compute_count(channel, group):
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out


@py_random_state(3)
def ws_graph(n, k, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """
    assert k >= 2 and k <= n
    # compute number of edges:
    edge_num = int(round(k * n / 2))
    count = compute_count(edge_num, n)
    # print(count)
    G = nx.Graph()
    for i in range(n):
        source = [i] * count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
        # print(source, target)
        G.add_edges_from(zip(source, target))
    # rewire edges from each node
    nodes = list(G.nodes())
    for i in range(n):
        u = i
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
        for v in target:
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G


@py_random_state(4)
def connected_ws_graph(n, k, p, tries=100, seed=1):
    """Returns a connected ws-flex graph.
    """
    for i in range(tries):
        # seed is an RNG so should change sequence each call
        G = ws_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError('Maximum number of tries exceeded')


def generate_graph(message_type='ws', n=16, sparsity=0.5, p=0.2,
                   directed=False, seed=123):
    ### for relaxed ws
    degree = n * sparsity
    if message_type == 'ws':
        graph = connected_ws_graph(n=n, k=degree, p=p, seed=seed)
    return graph

# graph = load_graph('mcwhole', True)
# graph = load_graph('mcwholeraw', True)
# graph = load_graph('mcvisual', True)
# graph = load_graph('mcvisualraw', True)
# graph = load_graph('cat', True)
# graph = load_graph('catraw', True)
