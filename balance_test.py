# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:46:13 2023

@author: 84711
"""

import pandas as pd
import numpy as np
import math

def balance_check_means(data, treatment_col, covariate_cols, weights_col):
    """
    平衡性检验函数，计算匹配前后两组在协变量上的均值差异。

    参数：
    data: 包含数据的DataFrame。
    treatment_col: 二元变量，表示干预效应的列。
    covariate_cols: 协变量列名的列表。
    weights_col: 权重列名。

    返回：
    balance_results: DataFrame，包含匹配前后均值差异的统计信息。
    """
    # 根据干预效应划分两组
    group_1 = data[data[treatment_col] == 1]
    group_0 = data[data[treatment_col] == 0]
    
    group_1_var=group_1[covariate_cols].var().values
    group_0_var=group_0[covariate_cols].var().values
    
    std=np.sqrt(group_0_var/2+group_1_var/2)
    
    weighted_1_var=weighted_variance(group_1, covariate_cols, weights_col)
    weighted_0_var=weighted_variance(group_0, covariate_cols, weights_col)
    
    weighted_std=np.sqrt(weighted_0_var/2+weighted_1_var/2)
    # 计算匹配前的均值差异
    pre_match_diff = group_1[covariate_cols].mean() - group_0[covariate_cols].mean()
    
    tw=np.mean(group_1[covariate_cols].to_numpy()*group_1[weights_col].to_numpy()[:,None]/group_1[weights_col].sum(),axis=0) -np.mean( group_0[covariate_cols].to_numpy()*group_0[weights_col].to_numpy()[:,None]/group_0[weights_col].sum(),axis=0)
    
    # 计算匹配后的加权均值差异
    weighted_diff = pd.DataFrame(tw)[0]

    # 统计信息
    balance_results = pd.DataFrame({
        'Pre-Match Mean Difference': pre_match_diff.values,
        'Weighted Mean Difference': weighted_diff.values,
        'Pre B-stats':(pre_match_diff/std).values,
        'After B-stats':(weighted_diff/weighted_std).values
    }, index=covariate_cols)

    return balance_results

def balance_check_cofficient(data, treatment_col, covariate_cols, weights_col):
    '''
    Balance check function, return a table of regression coefficients.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    covariate_cols : TYPE
        DESCRIPTION.
    weights_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    

def weighted_variance(data, columns, weight):
    """
    计算加权方差的函数。

    参数：
    data: 包含数据的DataFrame。
    columns: 需要计算方差的列名列表。
    weight: 权重列名。

    返回：
    var: 加权方差。
    """
    cov_cols=data[columns]
    weight_col=data[weight]
    
    # 计算加权均值
    weighted_mean = pd.DataFrame(cov_cols.to_numpy()*weight_col.to_numpy()[:,None],columns=cov_cols.columns).sum() / data[weight].sum()

    # 计算每个观测值和加权均值的差的平方
    squared_diff = (data[columns] - weighted_mean) ** 2

    # 对差的平方进行加权求和
    weighted_sum_squared_diff = pd.DataFrame(squared_diff.to_numpy() * data[weight].to_numpy()[:,None]).sum()

    # 计算加权方差
    num_nonzero_weights = (data[weight] != 0).sum()
    var = weighted_sum_squared_diff / ((num_nonzero_weights - 1) / num_nonzero_weights *data[weight].sum() )

    return var

