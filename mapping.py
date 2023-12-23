# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:44:04 2023

@author: 84711
"""

import numpy as np

def map_entities_and_times(dataframe,individual_col,time_col):
    """
    将具体的个体编码和时间编码映射为自然数，返回映射后的数组和映射关系。

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    mapped_data : TYPE
        DESCRIPTION.
    individual_to_index : TYPE
        DESCRIPTION.
    time_to_index : TYPE
        DESCRIPTION.

    """
    # 获取个体列和时间列
    individuals = dataframe[individual_col]
    times = dataframe[time_col]

    # 获取唯一的个体和时间
    unique_individuals = np.unique(individuals.values)
    unique_times = np.unique(times.values)

    # 构建个体和时间的映射关系
    individual_to_index = {ind: idx for idx, ind in enumerate(unique_individuals)}
    time_to_index = {time: idx for idx, time in enumerate(unique_times)}
    
    dataframe[individual_col]=dataframe[individual_col].map(individual_to_index)
    dataframe[time_col]=dataframe[time_col].map(time_to_index) 

    return dataframe, individual_to_index, time_to_index


def unique_individuals(dataframe,individual_col,treatment_col):
    """
    提取独立的个体编号，返回独特个体编号和处理状态的二维数组。

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    ind_index : TYPE
        DESCRIPTION.

    Returns
    -------
    unique_individuals : TYPE
        DESCRIPTION.

    """
    temp_array=dataframe[[individual_col,treatment_col]].values
    # 初始化一个空列表来存储唯一的行
    unique_rows_list = []

    # 使用循环逐行构建唯一的行列表
    for row in temp_array:
        if row.tolist() not in unique_rows_list:
            unique_rows_list.append(row.tolist())
    
    unique_rows=dataframe(unique_rows_list,columns=[individual_col,treatment_col])
    return unique_rows

def join_arrays_with_index(array1, array2, index_column):
    """
    将第二个数组根据第一个数组中的行索引数据拼接到第一个数组上。

    参数：
    - array1: 第一个二维数组
    - array2: 第二个二维数组
    - index_column: 一维数组，表示要使用的行索引数据

    返回值：
    - 拼接后的二维数组
    """
    # 使用索引数组从第二个数组中选择相应的行
    selected_rows = array2[index_column]

    # 将第一个数组和选择的行连接在一起
    result_array = np.concatenate((array1, selected_rows), axis=1)

    return result_array

def remampping_index(dataframe, remap_col ,map_dict):
    remap_relationship= {v:k for k,v in map_dict.items()}
    dataframe[remap_col]=dataframe[remap_col].map(remap_relationship)
    return dataframe