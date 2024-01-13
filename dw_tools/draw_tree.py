import daft
import numpy as np


def dh_add_node(pgm, name, label, r, c, scale=1.5, aspect=1.2):
    '''add a node to a PGM, using Doeke's (row, column) coordinates

    Keyword arguments:
    pgm   -- daft PGM object to add a node to
    name  -- name of the node to add
    label -- label that will be displayed in the node
    r     -- horizontal (row) coordinate
    c     -- vertical (column) coordinate
    scale -- scale of each node--bigger is more readable (default: 1.5)
    aspect-- wdith:height ratio (default: 1)
    '''
    _c = 1.8 * c
    pgm.add_node(name, label, _c, 3 - r, scale=scale, aspect=aspect)
    return


def prepare_tree_pgm(
    list_of_nodes, list_of_edges, list_of_node_labels, list_of_edge_labels, root=0
):
    _pgm = daft.PGM()  #

    _n_nodes = len(list_of_nodes)
    _node_assigned = np.zeros((_n_nodes,), dtype=bool)
    _node_assigned[root] = True
    _distance_from_root = {0: [root]}
    _n_walk = 1
    _max_width = 1
    _coordinates = np.zeros((_n_nodes, 2))
    _coordinates[root, 0] = 1
    _width = np.zeros((_n_nodes,))
    _width[root] = 1
    while not _node_assigned.all():
        _tmp_list_assign = []
        _tmp_list_dist = []
        for _edge in list_of_edges:
            if (not _node_assigned[_edge[1]]) and _node_assigned[_edge[0]]:
                _tmp_list_assign.append(_edge[1])
                _tmp_list_dist.append(_edge[1])
                _coordinates[_edge[1], :] = [len(_tmp_list_assign), _n_walk]
        _node_assigned[_tmp_list_assign] = True
        _distance_from_root[_n_walk] = _tmp_list_dist
        _width[_tmp_list_assign] = len(_tmp_list_assign)
        _max_width = np.amax([_max_width, len(_tmp_list_dist)])
        _n_walk = _n_walk + 1
        if _n_walk > _n_nodes:
            print("warning: could not assign all nodes!")
            break
    _coordinates[:, 0] = _coordinates[:, 0] + 0.5 * (_max_width - _width)

    for n in list_of_nodes:
        dh_add_node(
            _pgm, list_of_nodes[n], list_of_node_labels[n], _coordinates[n, 0], _coordinates[n, 1]
        )
    for e in list_of_edges:
        _pgm.add_edge(
            list_of_nodes[e[0]],
            list_of_nodes[e[1]],
            label=f"{list_of_edge_labels[list_of_edges.index(e)]}",
            label_params={'fontsize': 14},
        )
    return _pgm
