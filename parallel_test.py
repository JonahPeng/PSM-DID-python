# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:04:29 2023

Functions for parallel trends test of DID model.

@author: 84711
"""

import mapping as mp
import linearmodels as ls

def parallel_test(data,end_v,time_col,treatment_col, individual_col, covariate_cols,time_startpoint,weights_col=None):
    '''
    Check for parallel tendency. After passing which, DID model fits.
    
    Using the method from Luo(2015). Event Study Method.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    covariate_cols : TYPE
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
    
    parallel_model.fit()
    
    # Under constructing……
    return 
