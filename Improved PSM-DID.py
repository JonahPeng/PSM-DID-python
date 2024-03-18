import os

import pandas as pd

import mapping as mp

import distance as ds
import weight as wt

import linearmodels as ls
import statsmodels.api as sm
import patsy as pt

import Testing as ts

import To_docx as tdx


import warnings # ignore warnings
warnings.filterwarnings("ignore")

# 匹配全局变量定义区
radius = 0.05
neighbor = 4

# 函数定义区
def matching(dataframe,individual_col,time_col,treatment_col,propensity_col,fixed_features_cols=None,method=False,new_matching=False,new_distance=False):
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
    if os.path.exists(r"PythonData\WeightedData.csv") and not new_matching:
        weighted_data=pd.read_csv(r"PythonData\WeightedData.csv")
        return weighted_data        
    
    matching_methods={'global':global_matching,'phased':phased_matching}
    
    usingdata=dataframe.loc[:,[individual_col,time_col,treatment_col,propensity_col,*fixed_features_cols]]
    
    mapped_data,individuals_index,times_index=mp.map_entities_and_times(usingdata,individual_col,time_col)
    
    # Transfrom the two-dim data into three-dim data.
    three_dim_matrix=ds.transform_to_three_dim(mapped_data,individual_col,time_col,fixed_features_cols)
    
    weight=matching_methods[method](three_dim_matrix, individual_col, individuals_index, time_col, times_index, treatment_col, propensity_col, fixed_features_cols,refresh=new_distance)
    
    weighted_data=pd.merge(dataframe,weight,left_on=[individual_col,time_col],right_on=[individual_col,time_col],how='left')
    weighted_data.rename(columns={0:'weights'},inplace=True)
    
    '''
    else:
    # Shapley weight decomposition. Still under constrcution. For logic misunderstanding of decomposition.
        powerset=wt.generate_powersetgraph_nodes(mapped_data,times_index,individual_col, time_col, treatment_col,propensity_col,neighbor,radius,fixed_features_cols)
        powerset=wt.connect_powersetgraph_nodes(powerset)
        time_decomposed_weight=wt.cal_shapley_value(powerset,shap_abs=True)
    
        time_decomposed_weight=time_decomposed_weight.stack().reset_index()
    
        time_decomposed_weight.rename(columns={'level_1':time_col},inplace=True)
        
        time_decomposed_weight=mp.remampping_index(time_decomposed_weight, individual_col, individuals_index)
        time_decomposed_weight=mp.remampping_index(time_decomposed_weight, time_col, times_index)
        
        weighted_data=pd.merge(dataframe,time_decomposed_weight,on=['PAC','year'],how='left')
    '''
    weighted_data.to_csv(r"PythonData\WeightedData.csv",index=True)
    
    return weighted_data
    

def global_matching(three_dim_matrix,individual_col,individuals_index,time_col,times_index,treatment_col,propensity_col,fixed_features_cols,refresh=False):
    '''
    Matching using fixed relationship of individuals of all the observed times.

    Parameters
    ----------
    three_dim_matrix : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    fixed_features_cols : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Calculate the global distance matrix.(2-dim)
    distance_matrix=ds.calculate_global_feature_distances(three_dim_matrix,treatment_col,propensity_col,fixed_features_cols,new_matrix=refresh)
    
    inds_treatment=three_dim_matrix[treatment_col]
    
    # Confirm the neighbors.
    neighbors_relationship=wt.neighbors(distance_matrix,neighbor,radius)
    
    temps=[]
    # Generate the global weights of observations. 
    for i in range(len(times_index)):
        weight_series=wt.generate_weight(neighbors_relationship,inds_treatment.loc[(slice(None),i),:],neighbor)
    
        new_index=pd.MultiIndex.from_product([weight_series.index,[i]],names=[individual_col,time_col])        
        temp=weight_series.copy().reset_index().set_index(new_index)
        temp.rename(columns={0:'weights'},inplace=True)
        temp=temp['weights']
        
        temps.append(temp)
    
    merged_series=pd.concat(temps,axis=0)
    
    merged_series=mp.remampping_index(pd.DataFrame(merged_series).reset_index(), individual_col, individuals_index)
    merged_series=mp.remampping_index(merged_series, time_col, times_index)
    
    return merged_series
    
def phased_matching(three_dim_matrix,individual_col,individuals_index,time_col,times_index,treatment_col,propensity_col,fixed_features_cols,refresh=False):
    '''
    Matching using each observed time as a unique matching process.

    Parameters
    ----------
    three_dim_matrix : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    individuals_index : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    times_index : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    propensity_col : TYPE
        DESCRIPTION.
    fixed_features_cols : TYPE
        DESCRIPTION.
    inds_treatment : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Calculate the phased distance matrix.(3-dim)
    distance_matrix=ds.calculate_time_seperated_feature_distance(three_dim_matrix, treatment_col, propensity_col,new_matrix=refresh)
    
    inds_treatment=three_dim_matrix[treatment_col]
    
    weight_series_list=[]
    
    # Confirm the neighbors.
    for i in range(distance_matrix.shape[2]):    
        neighbors_relationship=wt.neighbors(distance_matrix[:,:,i], neighbor,radius)
        
        # Generate the weights
        weight_series=wt.generate_weight(neighbors_relationship, inds_treatment.loc[(slice(None),i)], neighbor)
        new_index=pd.MultiIndex.from_product([weight_series.index,[i]],names=[individual_col,time_col])
        weight_series=weight_series.copy().reset_index().set_index(new_index)
        weight_series.rename(columns={0:'weights'},inplace=True)
        
        weight_series=weight_series['weights']
        
        weight_series_list.append(weight_series)
        
    # Restore index values.
    merged_series=pd.concat(weight_series_list,axis=0)
    
    merged_series=mp.remampping_index(pd.DataFrame(merged_series).reset_index(), individual_col, individuals_index)
    merged_series=mp.remampping_index(merged_series, time_col, times_index)
    
    return merged_series

def PanelPSM(data,end_v,exo_vs,time_col,individual_col,phased=False,multi_category=False):
    '''
    Panel probit regression. 

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    end_v : TYPE
        DESCRIPTION.
    exo_vs : TYPE
        DESCRIPTION.
    treatment_col : TYPE
        DESCRIPTION.
    time_col : TYPE
        DESCRIPTION.
    individual_col : TYPE
        DESCRIPTION.
    phased : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    A dataframe with prediction results of propensity.

    '''
    # This function has not yet been tested.
    
    all_vars=exo_vs+[end_v]
    # Make data balanced.
    df=data.dropna(subset=all_vars)
    
    times=df[time_col].unique()
    
    propensity_cols=[]
    
    if multi_category:
        dummy_df=pd.get_dummies(df[end_v],prefix=end_v)
        end_vs=dummy_df.columns.tolist()
        df.drop(end_v,axis=1)
    
        df=pd.concat([df,dummy_df],axis=1)
        for time in times:
            for end_v in end_vs:
                formula=f"{end_v} ~ 1 + {'+'.join(exo_vs)}"
                using_data=df[time_col==time]
                y,X=pt.dmatrices(formula,data=using_data,return_type='dataframe')
            
                probit_model=sm.Probit(y,X)
                probit_result=probit_model.fit()
                
                propensity_scores=probit_result.predict()
                
                propensity_col=f'p_{end_v}'
                propensity_cols.append(propensity_col)
                
                df.loc[df[time_col]==time,propensity_col]=propensity_scores
    
    else:
        for time in times:
            formula=f"{end_v} ~ 1 + {'+'.join(exo_vs)}"
            using_data=df[time_col==time]
            y,X=pt.dmatrices(formula,data=using_data,return_type='dataframe')
                
            probit_model=sm.Probit(y,X)
            probit_result=probit_model.fit()
                
            propensity_scores=probit_result.predict()
                
            propensity_col=f'p_{end_v}'
            propensity_cols.append(propensity_col)
                
            df.loc[df[time_col]==time,propensity_col]=propensity_scores
    
    return df
    

def PanelDID(data,end_v,exo_vs,treatment_col,time_col,individual_col,applied_col,weights_col=None):
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
    df=data.copy()
    
    df['interactive']=df[applied_col]*df[treatment_col]
    
    df=df.set_index([individual_col,time_col])
    
    formula_str=f"{end_v} ~ 1 + {'+'.join(exo_vs)} + interactive + EntityEffects +TimeEffects"
    
    DID_model = ls.PanelOLS.from_formula(formula_str,data=df,weights=df[weights_col])
    
    result=DID_model.fit()
    
    return result

def log(text,log_file):
    with open(log_file,'a') as file:
        file.write(text)
        file.write("\n")

# 示例用法
if __name__ == "__main__":
    # Set working path
    dir_path=r"D:\OneDrive\【S05】组内事宜\主体功能区规划评估"
    os.chdir(dir_path)
    
    # 日志文件路径
    log_file=r"PythonData\log.txt"
    
    # Read the original dataset.
    df_= pd.read_csv(r"D:\OneDrive\【S05】组内事宜\主体功能区规划评估\ExcelData\Part2_Normal_Comparison.csv")
    
    with open(log_file,'w') as file:
        file.write("# StartPoint\n")
    
    individual_col='PAC'
    time_col='year'
    
    treatment_cols=['Agri2Urb','Agri2Eco','Urb2Eco']
    propensity_cols=['Agri2Urbpvalue','Agri2Ecopvalue','Urb2Ecopvalue']
    filters_cols=['f1','f2','f3']
    applied_col='Time'
    
    fixed_features_cols=['ecoregion']
    
    df_=df_.loc[df_[time_col]>=2009]
    df_=df_.loc[df_[time_col]<=2017]
    
    
    for i in range(len(treatment_cols)):
        log(f"\n## Variables Group {i}",log_file)
        df=df_.loc[df_[filters_cols[i]]==0]
        
        treatment_col=treatment_cols[i]
        propensity_col=propensity_cols[i]
        # Matching and weighting.(PSM)
        weighted_data=matching(df,individual_col,time_col,treatment_col,propensity_col,fixed_features_cols,method='phased',new_distance=True,new_matching=True)
    
        weights_col='weights'
        
        # Define the group of variables used in regression.
        ecology_columns=["PWL","NPP","PM25","HFP"] 
        agriculture_columns=["GP","PAM","PAL","IRR"]
        development_columns=["GDP","PAP","PR","DCE","PIeS","POP"]
        variables_groups=[agriculture_columns,development_columns,ecology_columns]
        all_covariances=ecology_columns+agriculture_columns+development_columns
        
        # Balance check.
        balance_check_results,balance_check_bools=ts.balance_check_multi_times(weighted_data, treatment_col, all_covariances, weights_col, time_col)
        
        
        if all(balance_check_bools):
            log("The balance check of Propensity Matching passed.",log_file)
            # Common support check.
            common_support_check=ts.common_support_check(weighted_data, treatment_col, propensity_col,weights_col)
            
            log("Results of common support check:",log_file)
            log(tdx.df_to_docx(common_support_check),log_file)
            
            results=[]
            
            end_vs=variables_groups.pop(i)
            exo_vs=[]
            for item in variables_groups:
                exo_vs=exo_vs+item
            
            # Drop the unmatched observations.
            weighted_data=weighted_data.loc[weighted_data[weights_col]>0]
            
            # Define related variables of econometrics model.
            for end_v in end_vs:
                result=PanelDID(weighted_data,end_v,exo_vs,treatment_col,time_col,individual_col,applied_col,weights_col=weights_col)
                results.append(result)
            
            time_startpoint=2012
            
            for end_v in end_vs:
                parallel_test_check=ts.parallel_test(weighted_data, end_v, time_col, treatment_col, individual_col, exo_vs, time_startpoint,weights_col=weights_col)
            
            temp_df,table=tdx.regs_to_docx(results,end_vs)
            log("Results of DID Regressions:",log_file)
            log(table,log_file)
        
        else:
            log("The balance check of Propensity Matching not passed. Failed Times:",log_file)
            false_times= [index for index,value in enumerate(balance_check_bools) if not value]
            log(",".join([str(t) for t in false_times]),log_file)
            log('\n',log_file)