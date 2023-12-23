# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:23:47 2023

@author: 84711
"""

def generate_subsets(my_list):
    '''
    计算一个列表的全部子集。

    Parameters
    ----------
    list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    subsets=[[]]
    
    for element in my_list:
        current_subsets=[]
        for subset in subsets:
            current_subsets.append(subset+[element])
        subsets.extend(current_subsets)
        
    return subsets

