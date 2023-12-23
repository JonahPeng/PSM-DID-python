# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:42:28 2023

@author: 84711
"""

import numpy as np
import pandas as pd

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
    
    result_series=stacked_series.groupby(stacked_series).size()
    
    result_series.index=result_series.index.astype(int)
    result_series.index.name='PAC'
    
    return_series=pd.Series(0,index=neighbor_relationship.index)
            
    for i,value in enumerate(result_series):
        if treatment_series[i]==1 and value!=0:
            result_series[i]=1
        else:
            result_series[i]=value/num_neighbors
    
    common_index=return_series.index.intersection(result_series.index)
    return_series.loc[common_index]=result_series.loc[common_index]
    
    return return_series