# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:55:42 2024

For output the regression result into docx tables.

@author: 84711
"""

import tabulate as tb
import numpy as np

def reg_to_docx(regs,end_vs,exo_vs):
    '''
    Transfrom given results of regression into table.

    Parameters
    ----------
    regs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    params=[]
    pvalues=[]
    nobs=[]
    rss=[]
    
    # Get the information from results of regression.
    for reg in regs:
        param=reg.params.values
        pvalue=reg.pvalues.values
        obs=reg.nobs
        rsquared=reg.rsquared
        
        params.append([round(i,3) for i in param])
        pvalues.append([round(i,3) for i in pvalue])
        nobs.append(obs)
        rss.append(round(rsquared,3))
        
    output_tb=np.zeros((len(params[0])*2+4,len(rss)+1),dtype='U25')
    
    for i in range(len(end_vs)+1):
        if i==0:
            for j in range(len(exo_vs)):
                output_tb[2*j+1][i]=exo_vs[j]
            
            output_tb[-5][i]='Treatment×Time'
            output_tb[0][i]=' '
            output_tb[-2][i]="Observations"
            output_tb[-1][i]="R-squared"
            
        else:
            output_tb[0][i]=end_vs[i-1]
            for j in range(len(params[0])):
                output_tb[2*j+1][i]=params[i-1][j]
                output_tb[2*j+2][i]=pvalues[i-1][j]
                
                # Foolish stars
                if pvalues[i-1][j]<0.1:
                    output_tb[2*j+2][i]=output_tb[2*j+2][i]+'*'
                    if pvalues[i-1][j]<0.05:
                        output_tb[2*j+2][i]=output_tb[2*j+2][i]+'*'
                        if pvalues[i-1][j]<0.01:
                            output_tb[2*j+2][i]=output_tb[2*j+2][i]+'*'
            
            output_tb[-2][i]=nobs[i-1]
            output_tb[-1][i]=rss[i-1]
    
    table=tb.tabulate(output_tb,headers='firstrow',tablefmt='fancy_grid')    
    
    print(table)
    return table
    

