# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:04:29 2023

Functions for parallel trends test of DID model.

@author: 84711
"""

import linearmodels as ls
import tools as tls
import matplotlib.pyplot as plt

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
    
    ax.set_title('Event Study Figure')
    ax.legend()
    
    plt.show()
    
    return model_result
