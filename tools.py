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

def max_element_less_than_x(lst, target):
    '''
    Find the max element in given list which is smaller than target number. If there is no such element, return None.

    Parameters
    ----------
    lst : List
        Given list.
    target : Double
        Target number.

    Returns
        The max element (value).
    -------
    None.

    '''
    flitered_elements=filter(lambda x: x < target,lst)
    
    return max(flitered_elements, default=None)
