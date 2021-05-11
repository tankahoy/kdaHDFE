import networkx as nx
import numpy as np


def cal_df(data_df, fixed_effects):
    """
    This function returns the degree of freedom of category variables(DoF). When there are category variables( fixed
    effects), part of degree of freedom of the model will lose during the demean process. When there is only one
    fixed effect, the loss of DoF is the level of this category. When there are more than one fixed effects,
    we need to calculate their connected components. Even if there are more than two fixed effects, only the first
    two will be used.

    :param data_df: Data with relevant variables
    :param fixed_effects: List of category variables(fixed effect)
    :return: the degree of freedom of category variables
    """
    e = sum([len(np.unique(data_df[fe].values)) for fe in fixed_effects])

    if len(fixed_effects) >= 2:
        g = nx.Graph()
        edge_list = data_df[fixed_effects].values.tolist()
        for ll in edge_list:
            g.add_edge('fix1_' + str(ll[0]), 'fix2_' + str(ll[1]))

        return e - (len(list(nx.connected_components(g))) + len(fixed_effects) - 2)
    else:
        return e
