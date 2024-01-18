# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:42:28 2023

@author: 84711
"""

import numpy as np
import pandas as pd

import networkx as nx
import math

import tools as tls
import distance as ds

def neighbors(distance_matrix ,n, radius):
    """
    找到距离矩阵中每个个体最近的 n 个邻居，记录其索引。

    参数：
    distance_matrix: 特征值距离矩阵，二维数组。
    n: 邻居数。
    radius: 半径范围。

    返回：
    neighbors_list: 列表，每个个体最多有 n 个有效的近邻索引。
    """
    num_individuals = distance_matrix.shape[0]

    # 找到每个个体最近的 n 个邻居的索引
    min_indices = np.argpartition(distance_matrix, n, axis=1)[:, :n]

    # 初始化列表
    neighbors_list = []

    # 对于每个个体，判断距离是否在半径范围内，将有效邻居索引加入列表
    for i in range(num_individuals):
        distances_to_neighbors = distance_matrix[i, min_indices[i]]
        valid_neighbors = min_indices[i][np.abs(distances_to_neighbors) < radius]
        neighbors_list.append(valid_neighbors)
    
    array=np.full((num_individuals,n),np.nan)
    
    for i,j in enumerate(neighbors_list):
        array[i][0:len(j)]=j
    cols_list=[f'neighbor_{i+1}' for i in range(n)]
    
    df=pd.DataFrame(array,columns=cols_list)
    df.index.name='PAC'
    
    return df

def generate_weight(neighbor_relationship,treatment_series,num_neighbors):
    '''
    根据传入的邻接关系和干预列，生成权重序列（以映射后的id为索引）

    Parameters
    ----------
    neighbor_relationship : TYPE
        DESCRIPTION.
    treatment_series : TYPE
        DESCRIPTION.

    Returns
    -------
    result_series : TYPE
        DESCRIPTION.

    '''
    stacked_series=neighbor_relationship.stack()
    # Groupby stacked_series means calculate the times of unique value.
    result_series=stacked_series.groupby(stacked_series).size()
    
    result_series.index=result_series.index.astype(int)
    result_series.index.name='PAC'
    
    return_series=pd.Series(0,index=neighbor_relationship.index)
            
    for index,value in result_series.items():
        if treatment_series[index]==1 and value!=0:
            result_series[index]=1
        else:
            result_series[index]=value/num_neighbors
    
    common_index=return_series.index.intersection(result_series.index)
    return_series.loc[common_index]=result_series.loc[common_index]
    
    return return_series


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
            neighbors_relationship=neighbors(distance_matrix,neighbor,radius)

            weighted_series=generate_weight(neighbors_relationship,inds_treatment,neighbor)
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