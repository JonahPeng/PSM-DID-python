import numpy as np
import pandas as pd

import mapping as mp

import distance as ds
import weight as wt

import tools as tls

import PSM_test as pt

import shap_weight as sw

# 匹配全局变量定义区
radius = 0.05
neighbor = 4

# 函数定义区
def weight_cal(dataframe,individual_col,time_col,treatment_col,propensity_col,fixed_features_cols=None,glob=False):
    '''
    Function to calculate the weight for DID and balance check.

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    fixed_features_cols : TYPE, optional
        DESCRIPTION. The default is None.
    glob : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    weighted_data : TYPE
        DESCRIPTION.

    '''
    usingdata=dataframe.loc[:,[individual_col,time_col,treatment_col,propensity_col,*fixed_features_cols]]
    
    mapped_data,individuals_index,times_index=mp.map_entities_and_times(usingdata,individual_col,time_col)
    
    
    # 提取个体干预效应值（观测期内不变），这是个脚手架
    inds_treatment=mapped_data.groupby(individual_col)[treatment_col].unique()
    
    # Transfrom the two-dim data into three-dim data.
    three_dim_matrix=ds.transform_to_three_dim(mapped_data,individual_col,time_col,fixed_features_cols)
    
    # Calculation the distance matrix.
    distance_matrix=ds.calculate_feature_distances(three_dim_matrix,treatment_col,propensity_col,fixed_features_cols)
    
    # Confirm the neighbors.
    neighbors_relationship=wt.neighbors(distance_matrix,neighbor,radius)
    
    if glob:
    # Generate the global weights of observations. 
        weighted_series=wt.generate_weight(neighbors_relationship,inds_treatment,neighbor)
    
        temp=mp.remampping_index(pd.DataFrame(weighted_series).reset_index(), individual_col, individuals_index)
        weighted_data=pd.merge(dataframe,temp,left_on='PAC',right_on='PAC',how='left')
        weighted_data.rename(columns={0:'weights'},inplace=True)        
    
    else:
    # Shapley weight decomposition. Still under constrcution. For logic misunderstanding of decomposition.
        powerset=sw.generate_powersetgraph_nodes(mapped_data,times_index,individual_col, time_col, treatment_col,propensity_col,neighbor,radius,fixed_features_cols)
        powerset=sw.connect_powersetgraph_nodes(powerset)
        time_decomposed_weight=sw.cal_shapley_value(powerset,shap_abs=True)
    
        time_decomposed_weight=time_decomposed_weight.stack().reset_index()
    
        time_decomposed_weight.rename(columns={'level_1':time_col},inplace=True)
        
        time_decomposed_weight=mp.remampping_index(time_decomposed_weight, individual_col, individuals_index)
        time_decomposed_weight=mp.remampping_index(time_decomposed_weight, time_col, times_index)
        
        weighted_data=pd.merge(dataframe,time_decomposed_weight,on=['PAC','year'],how='left')
    
    return weighted_data
    
def DID(data,end_v,exo_vs,time_col=None,individual_col=None,weights_col=None):
    '''
    Difference in difference model.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    end_v : TYPE
        DESCRIPTION.
    exo_vs : TYPE
        DESCRIPTION.
    time_col : TYPE, optional
        DESCRIPTION. The default is None.
    individual_col : TYPE, optional
        DESCRIPTION. The default is None.
    weights_col : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    
    return 

# 示例用法
if __name__ == "__main__":
    # 生成示例数组
    
    df= pd.read_csv(r"D:\OneDrive\【S05】组内事宜\主体功能区规划评估\Part2.csv")
    
    individual_col='PAC'
    time_col='year'
    
    treatment_col='function1'
    propensity_col='p1'
    
    fixed_features_cols=['ecoregion']
    
    data_test=df.head(500)
    
    # Matching and weighting.(PSM)
    weighted_data=weight_cal(data_test,individual_col,time_col,treatment_col,propensity_col,fixed_features_cols,glob=True)
    
    weights_col='weights'
    
    # Define the group of variables used in regression.
    ecology_columns=["PWL","LAI","NPP","PM25","HFP"] 
    agriculture_columns=["GP","PAM","PAL","IRR","PIR"]
    development_columns=["GDP","PAP","PR","DCE","PMP","PIeS","POP"]
    obs_columns=["PAC","year","function1","function2","function3"]
    all_variances= obs_columns+ecology_columns+agriculture_columns+development_columns
    all_covariances=ecology_columns+agriculture_columns+development_columns
    
    # Dual-track balance check.
    unique_times=weighted_data['year'].unique()
    COF_check_results=[]
    SMD_check_results=[]
    
    for time in unique_times:
    # COF balance check.
        COF_check_result=pt.balance_check_cofficient(weighted_data.loc[weighted_data['year']==time], treatment_col, all_covariances, weights_col)
        COF_check_results.append(COF_check_result)
    
    # SMD balance check.
        SMD_check_result=pt.balance_check_means(weighted_data.loc[weighted_data['year']==time], treatment_col, all_covariances, weights_col)
        SMD_check_results.append(SMD_check_result)
    
    # Common support check
    pt.propensity_hist_check(weighted_data, treatment_col, propensity_col, weights_col)
    
    common_support_check=pt.common_support_check(weighted_data, treatment_col, propensity_col)
    
    
    
    