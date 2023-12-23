import numpy as np
import pandas as pd

def transform_to_three_dim(mapped_data, individual_col,time_col,fixed_features_cols):
    '''
    Dataframe转为Multiindex，如果提供了特殊类变量fixed features，则确保他们在所有期均非空。

    Parameters
    ----------
    mapped_data : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    fixed_features_cols : TYPE
        DESCRIPTION.

    Returns
    -------
    three_dim_matrix : TYPE
        DESCRIPTION.

    '''
    # 初始化三维数组
    three_dim_matrix=mapped_data.set_index([individual_col,time_col])
    
    # 填充精确匹配
    three_dim_matrix=fill_fixed_features(three_dim_matrix, fixed_features_cols)
    
    return three_dim_matrix

def fill_fixed_features(multi_index, fixed_features_cols):
    '''
    填充精确匹配值，用以确保每一期都存在特殊类别变量。

    Parameters
    ----------
    multi_index : pd.MultiIndex
        包含多层级索引的 DataFrame 的 MultiIndex。
    fixed_features_cols : list
        包含要填充的特殊类别变量的列名的列表。

    Returns
    -------
    result_mi : pd.MultiIndex
        经过填充的 MultiIndex。

    '''
    result_mi = multi_index.copy()
    if fixed_features_cols!=None:
    # 依次填充
        for feature in fixed_features_cols:
            for idi in result_mi.index.levels[0]:
                feature_series = result_mi.loc[idi, feature].copy()
    
                # 前向填充
                st = feature_series.first_valid_index()
                first_value = feature_series[st]
                feature_series.loc[:st] = first_value
    
                # 后向填充
                et = feature_series.last_valid_index()
                last_value = feature_series[et]
                feature_series.loc[et:] = last_value
                
                result_mi.loc[idi,feature]=feature_series.values

    return result_mi

def calculate_feature_distances(three_dim, treatment_col, propensity_col, fixed_features_cols=None):
    # 初始化距离矩阵
    index_levels = three_dim.index.levels
    inds, times = index_levels[0], index_levels[1]
    cols = three_dim.columns
    
    arrays = np.full((inds.size, times.size, cols.size), np.NaN)
    distance_matrix = np.full((inds.size, inds.size), np.inf, dtype=float)
    
    for i, ind in enumerate(inds):
        for j, time in enumerate(times):
            if (ind, time) in three_dim.index:
                arrays[i, j, :] = three_dim.loc[(ind, time)].values
    
    bool_array = np.full((inds.size , inds.size,times.size), False, dtype=bool)
    
    if fixed_features_cols:
        features_array_index = [cols.get_loc(key) for key in fixed_features_cols]
        treatment_col_index = cols.get_loc(treatment_col)
        propensity_col_index = cols.get_loc(propensity_col)
        
        for i in range(inds.size):
            for j in range(inds.size):
                if i != j:
                    k2 = arrays[i, :, treatment_col_index] != arrays[j, :, treatment_col_index]
                    k1 = arrays[i, :, features_array_index] == arrays[j, :, features_array_index]
                    bool_array[i, j, :] = k1 * k2
    else:
        treatment_col_index = cols.get_loc(treatment_col)
        propensity_col_index = cols.get_loc(propensity_col)
        for i in range(inds.size):
            for j in range(inds.size):
                if i != j:
                    k2 = arrays[i, :, treatment_col_index] != arrays[j, :, treatment_col_index]
                    bool_array[i, j, :] = k2
    
    
    
    propensity_distance = (arrays[:, :, propensity_col_index][:, None,:] -
                           arrays[:, :, propensity_col_index][None,:, :]) ** 2
    propensity_distance[~bool_array] = np.nan
    propensity_distance = np.nanmean(propensity_distance,axis=2)
    
    distance_matrix=propensity_distance
    distance_matrix[propensity_distance==np.nan] = np.inf
    
    return distance_matrix

if __name__=="__main__":
    # 示例映射数据
    data = {'id': [1, 1, 2, 2, 3],
        'time': [10, 20, 10, 20, 10],
        'feature1': [1, 1, 0, 0, 1],
        'feature2': [10, 20, 30, 40, 50],
        'fixed_feature1': [1, 2,1, 2, 4],
        'fixed_feature2': [10, 20, 10, 20, 40]}
    df = pd.DataFrame(data)
    # 转换为三维矩阵
    three_dimensional_array = transform_to_three_dim(df,'id','time',['fixed_feature1','fixed_feature2'])
    
    dm=calculate_feature_distances(three_dimensional_array,'feature1','feature2',['fixed_feature1','fixed_feature2'])
    
    # 打印结果
    print("三维矩阵:")
    print(three_dimensional_array)
    
    print(dm)