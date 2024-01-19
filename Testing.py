# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:46:13 2023

@author: 84711
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import linearmodels as ls
import patsy as pt
import matplotlib.pyplot as plt

import To_docx as tdx
import tools as tls

# Complex Functions
def balance_check_multi_times(data,treatment_col, covariate_cols, weights_col, time_col,check_method='SMD'):
    '''
    Balance check for multi times data. Generate check

    Parameters
    ----------
    data : pandas.dataframe
        Panel data.
    treatment_col : str
        Name of treatment col.
    covariate_cols : list of str
        Names of covariables columns.
    weights_col : str
        Names of weights col.
    time_col : str
        Name of times col.
    check_method: str
        Used method to evaluate the balance situation. Implemented: SMD(Standard Means Difference, default), COF(Coefficients Significance)

    Returns
    -------
    List of dataframe of check results.

    '''
    check_methods={'SMD':balance_check_means,'COF':balance_check_cofficient}
    
    unique_times=data[time_col].unique()
    check_results=[]
    check_bools=[]
    
    for time in unique_times:
        check_result,check_bool=check_methods[check_method](data.loc[data[time_col]==time],treatment_col,covariate_cols,weights_col)
        check_results.append(check_result)
        check_bools.append(check_bool)
    
    return check_results,check_bools
    

# Basic Functions
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
    is_balanced=False
    
    # 根据干预效应划分两组
    group_1 = data[data[treatment_col] == 1]
    group_0 = data[data[treatment_col] == 0]
    
    group_1_var=group_1[covariate_cols].var().values
    group_0_var=group_0[covariate_cols].var().values
    
    std=np.sqrt(group_0_var/2+group_1_var/2)
    
    weighted_1_var=__weighted_variance__(group_1, covariate_cols, weights_col)
    weighted_0_var=__weighted_variance__(group_0, covariate_cols, weights_col)
    
    weighted_std=np.sqrt(weighted_0_var/2+weighted_1_var/2)
    # 计算匹配前的均值差异
    pre_match_diff = group_1[covariate_cols].mean() - group_0[covariate_cols].mean()
    
    k1=np.mean(group_1[covariate_cols].to_numpy()*group_1[weights_col].to_numpy()[:,None]/group_1[weights_col].sum(),axis=0)
    k21=group_0[covariate_cols].to_numpy()
    k22=group_0[weights_col].to_numpy()
    k23=group_0[weights_col].sum()
    k2=np.mean(k21*k22[:,None]/k23,axis=0)
    tw=k1-k2
    
    # 计算匹配后的加权均值差异
    weighted_diff = pd.DataFrame(tw)[0]

    # Summary
    balance_result = pd.DataFrame({
        'Pre-Match Mean Difference': pre_match_diff.values,
        'Weighted Mean Difference': weighted_diff.values,
        'Pre B-stats':(pre_match_diff/std).values,
        'After B-stats':(weighted_diff/weighted_std).values
    }, index=covariate_cols)
    
    if (balance_result['After B-stats'].abs()<0.05).all():
        is_balanced=True
    
    balanced_check_tb=tdx.df_to_docx(balance_result)
    
    return balanced_check_tb,is_balanced

def balance_check_cofficient(dataframe, treatment_col, covariate_cols, weights_col):
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
    - ------
    None.

    '''
    is_balanced=False
    
    # Realization of Xie(2021)`s balance check.
    end_var=treatment_col
    exo_vars=covariate_cols
    
    formula=f"{end_var} ~ 1 + {'+'.join(exo_vars)}"
    
    y,X=pt.dmatrices(formula,data=dataframe,return_type='dataframe')
    
    probit_model=sm.Probit(y,X)
    probit_result=probit_model.fit()
    
    w_probit_model=sm.Probit(y,X)
    w_probit_result=w_probit_model.fit(cov_type='HC1',method='bfgs',weights=dataframe['weights'])
    
    regs=[probit_result,w_probit_result]
    
    if (regs[1].pvalues.values[1:]>0.05).all():
        is_balanced=True
    
    return tdx.regs_to_docx(regs, ['Pre-matching','After-matching']),is_balanced
 
def common_support_check(data, treatment_col, propensity_col,weights_col):
    '''
    Summary the common support observations.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    weights_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df=data.copy()
    
    propensity_hist(df, treatment_col, propensity_col, weights_col)
    
    max_propensity=df[df[treatment_col]==0][propensity_col].max()
    min_propensity=df[df[treatment_col]==0][propensity_col].min()
    
    max_propensity=min(max_propensity,df[df[treatment_col]==1][propensity_col].max())
    min_propensity=max(min_propensity,df[df[treatment_col]==1][propensity_col].min())
    
    usingdata=data.loc[:,[treatment_col,propensity_col]]
    commondata=usingdata.loc[(df[propensity_col]<=max_propensity) & (df[propensity_col]>=min_propensity)]
    
    count1=usingdata.groupby(treatment_col)[propensity_col].count().rename('all_obs')
    count2=commondata.groupby(treatment_col)[propensity_col].count().rename('common_obs')
    
    common_support_result=pd.concat([count1,count2],axis=1)
    common_support_result['%Common support']=common_support_result['common_obs']/common_support_result['all_obs']
    
    tdx.df_to_docx(common_support_result)
    
    return common_support_result


def propensity_hist(data,treatment_col,propensity_col,weights_col):
    '''
    Common support checking for Propensity Matching. Return a graph of common supoport situation. 

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df=data.copy()
    
    plt.figure(figsize=(10,6))
    
    plt.hist(df[df[treatment_col]==0][propensity_col],bins=30,alpha=0.5,label='Control Group',weights=df[df[treatment_col]==0][weights_col])
    plt.hist(df[df[treatment_col]==1][propensity_col],bins=30,alpha=0.5,label='Treatment Group',weights=df[df[treatment_col]==1][weights_col])
    
    plt.title('Propensity Score Distribution for Treatment and Control Groups')
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(rf'PythonData\PythonFigures\Propensity_Histogram_of_{treatment_col}.png')

def __weighted_variance__(data, columns, weight):
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

def parallel_test(data,end_v,time_col,treatment_col, individual_col, covariate_cols,time_startpoint,weights_col=None):
    '''
    Check for parallel tendency. After passing which, DID model fits.
    
    Using the method from Luo(2015). Event Study Method.
    
    Linked paper:罗知,赵奇伟,严兵.约束机制和激励机制对国有企业长期投资的影响[J].中国工业经济,2015(10):69-84.

    Parameters
    ----------
    data : Dataframe
        DESCRIPTION.
    time_col : Column name of Time property.
        DESCRIPTION.
    individual_col : Column name of Individual property.
        DESCRIPTION.
    covariate_cols : Column names of Co-variables.
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Generate the interactive dubbing variables of all observed times.
    if weights_col:
        df=data.loc[data[weights_col]>0].copy()
    else:
        df=data.copy()
    
    exo_vs=[]
    
    times=df[time_col].unique()
    
    for time in times:
        temp_name=str(time)
        name_col='t'+str(time)
        df[name_col]=0
        df.loc[df[time_col]==time,[name_col]]=1
        
        df[name_col]=df[name_col]*df[treatment_col]
        
        exo_vs.append(temp_name)
    
    exo_vs_int=[int(time) for time in exo_vs]
    # Drop the last period before the treatment as the control group.
    last_period=tls.max_element_less_than_x(exo_vs_int, time_startpoint)
    exo_vs_int.remove(last_period)
    exo_vs.remove(str(last_period))
    
    exo_vs=['t'+time for time in exo_vs]
    
    exo_vs.sort()
    
    # Two-way fixed effects panel OLS model (Event Study)
    df=df.set_index([individual_col,time_col])
    
    formula_str=f"{end_v} ~ 1 + {'+'.join(exo_vs)} + {'+'.join(covariate_cols)} + EntityEffects + TimeEffects"
    
    if weights_col:
        parallel_model=ls.PanelOLS.from_formula(formula_str,  data=df,weights=df[weights_col])
    else:
        parallel_model=ls.PanelOLS.from_formula(formula_str, data=df)
    
    model_result=parallel_model.fit()
    
    # Get the params and the CI95 of regression.
    params=model_result.params
    CI95=model_result.conf_int(0.95)
    
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    
    # Points to plot.
    x=range(len(exo_vs))
    y=params[1:1+len(exo_vs)].values
    y_ci=CI95.iloc[1:1+len(exo_vs),:]
    
    ax.plot(x,y,label='Estimated Effect',marker='o',color='#202020',linestyle='-',linewidth=2.5)
    ax.fill_between(x,y_ci['lower'],y_ci['upper'],color="#A0A0A0",alpha=0.75,label='95% CI')
    ax.plot(x,y_ci['lower'],color='#606060',linestyle='-.',linewidth=1.5)
    ax.plot(x,y_ci['upper'],color='#606060',linestyle='-.',linewidth=1.5)
    ax.axhline(y=0,color='#000000',linestyle='-',linewidth=1)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Estimated Effect')
    
    ax.set_xticks(x,[t.lstrip('t') for t in exo_vs])
    
    ax.set_title(f'Event Study Figure of {end_v}')
    ax.legend()
    
    fig.savefig(rf'PythonData\PythonFigures\Event_Study_Figure_of_{end_v}.png')
    
    plt.show()
    
    return model_result
