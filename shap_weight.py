# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:42:44 2023

@author: 84711
"""

import pandas as pd
import networkx as nx
import math

import tools as tls
import distance as ds
import weight as wt


def generate_powersetgraph_nodes(mapped_data,times_index,individual_col, time_col, treatment_col,propensity_col,neighbor,radius,fixed_features_cols=None):
    '''
    Generate different nodes in graph with time subsets slice.

    Parameters
    ----------
    mapped_data : TYPE
        DESCRIPTION.
    times_index : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    neighbor : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    fixed_features_cols : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    powerset : TYPE
        DESCRIPTION.

    '''
    powerset=nx.Graph()
    times=times_index.values()
    subtimes=tls.generate_subsets(times)
    
    three_dim_matrix=ds.transform_to_three_dim(mapped_data, individual_col, time_col, fixed_features_cols)
    inds_treatment=mapped_data.groupby(individual_col)[treatment_col].unique()
    
    node_num=0;
    for time in subtimes:
        if time:
            temp_matrix=three_dim_matrix.loc[(slice(None),time),:]

            distance_matrix=ds.calculate_feature_distances(temp_matrix,treatment_col,propensity_col,fixed_features_cols)
            neighbors_relationship=wt.neighbors(distance_matrix,neighbor,radius)

            weighted_series=wt.generate_weight(neighbors_relationship,inds_treatment,neighbor)
        else:
            weighted_series=pd.Series(0,index=inds_treatment.index)
            
        powerset.add_node(node_num, match_series=weighted_series)
        powerset.nodes[node_num]['layer']=len(time)
        powerset.nodes[node_num]['included_times']=time
        node_num+=1
        
    return powerset

def connect_powersetgraph_nodes(powerset_graph):
    '''
    Connect the edges of a full powerset graph.

    Parameters
    ----------
    powerset_graph : networkx.graph
        Only nodes.

    Returns
    -------
    None.

    '''
    # Note the layer of nodes.
    max_layer = max(nx.get_node_attributes(powerset_graph, 'layer').values())
    
    for layer in range(max_layer,1,-1):
        
        # Select the adjacent layers
        nodes_in_layer=[node for node, data in powerset_graph.nodes(data=True) if data['layer']==layer]
        lower_layer=[node for node, data in powerset_graph.nodes(data=True) if data['layer']==(layer-1)]
        
        # Add the edge attributed by additional value
        for node in nodes_in_layer:
            for lower_node in lower_layer:
                complement_set=set(powerset_graph.nodes[node]['included_times']) - set(powerset_graph.nodes[lower_node]['included_times'])
                complement_list = list(complement_set)
                addition=complement_list[0]
                
                margin_effect_series=powerset_graph.nodes[node]['match_series'] - powerset_graph.nodes[lower_node]['match_series']
                
                powerset_graph.add_edge(lower_node,node,add=addition)
                powerset_graph[lower_node][node]['margin_effect']=margin_effect_series
                powerset_graph[lower_node][node]['weight']=math.factorial(max_layer-layer)*math.factorial(layer-1)/math.factorial(max_layer)
                
    return powerset_graph
    
def cal_shapley_value(powerset_graph,shap_abs=False):
    '''
    Calculate the shapley value for each paticipant.

    Parameters
    ----------
    powerset_graph : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    result_series_dict={}
    
    for edge in powerset_graph.edges(data=True):
        addition=edge[2]['add']
        series=edge[2]['margin_effect']
        weight=edge[2]['weight']
        
        series=weight*series
        
        # Generate the addition dict
        if addition in result_series_dict:
            result_series_dict[addition] = result_series_dict[addition]+series
        else:
            result_series_dict[addition] = series
    
    result_df = pd.DataFrame(result_series_dict)
    
    if shap_abs:
        result_df=result_df.abs()
    # return a dataframe with shapley value (individual).
    return result_df