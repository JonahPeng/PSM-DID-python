# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:04:29 2023

Functions for parallel trends test of DID model.

@author: 84711
"""

import linearmodels as ls
import tools as tls

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
        df=data.loc[data[weights_col>0]].copy()
    else:
        df=data.copy()
    
    exo_vs=[]
    
    times=df[time_col].unique()
    
    for time in times:
        temp_name=time.type(str)
        df[temp_name]=0
        df.loc[df[time_col]==time,[temp_name]]=1
        
        df[temp_name]=df[temp_name]*df[treatment_col]
        
        exo_vs.append(temp_name)
    
    # Drop the last period before the treatment as the control group.
    last_period=tls.max_element_less_than_x(exo_vs, time_startpoint)
    exo_vs.remove(last_period)
    
    # Two-way fixed effects panel OLS model (Event Study)
    df.set_index([individual_col,time_col])
    
    formula_str=f"{end_v} ~ {'+'.join(exo_vs)}"
    
    if weights_col:
        parallel_model=ls.PanelOLS.from_formula(formula_str, data=df,weights=df[weights_col])
    else:
        parallel_model=ls.PanelOLS.from_formula(formula_str, data=df)
    
    parallel_model.entity_effects=True
    parallel_model.time_effects=True
    parallel_model.has_constant=True
    
    model_result=parallel_model.fit()
    
    print(model_result)
    
    # under constructing…… show the plot of event study……
    return model_result
