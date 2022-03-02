import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def quantile(df_col):
    new_feature = pd.DataFrame(index=df_col.index)
    new_feature[f"{df_col.name}_quantile"] = pd.qcut(df_col, [0, 0.1, 0.3, 0.7, 0.9, 1], labels=False)
    return new_feature

def polynomial(df_col, expo=2):
    new_feature = pd.DataFrame(index=df_col.index)
    new_feature[f"{df_col.name}^{expo}"] = df_col**expo
    return new_feature
    
    
def unary(df, func, col_list=None):
    if col_list is None:
        col_list = df.columns
    new_features = pd.DataFrame(index=df.index)
    for col in col_list:
        new_features = pd.merge(new_features, func(df[col]), left_index=True, right_index=True)
    return new_features
        
    
def interact(colA, colB):
    new_feature = pd.DataFrame(index=colA.index)
    new_feature[f"{colA.name}*{colB.name}"] = colA * colB
    return new_feature
    
def divide(colA, colB):
    new_feature = pd.DataFrame(index=colA.index)
    new_feature[f"{colA.name}/{colB.name}"] = colA / colB
    return new_feature
    
def binary(df, func, cols_list=None):
    if cols_list is None:
        cols_list = [(colA, colB) for colA in df.columns for colB in df.columns if colA != colB]
        cols_list = set(tuple(sorted(item)) for item in cols_list) # drop reverse duplicte columns
    new_features = pd.DataFrame(index=df.index)
    for colA, colB in cols_list:
        new_features = pd.merge(new_features, func(df[colA], df[colB]), left_index=True, right_index=True)
    return new_features

def get_unary_features(df):
    """
    calculate unary function on the columns, and upddate config
    """
    config = {}
    new_features = unary(df, func=quantile, col_list=df.columns)
    config["quantile"] = df.columns
    new_features = pd.merge(new_features, unary(df, func=polynomial, col_list=df.columns), left_index=True, right_index=True)
    config["polynomial"] = df.columns
    return new_features, config


def get_binary_features(df):
    """
    calculate binary function on the columns, and upddate config
    """
    config = {}
    cols_list = [(colA, colB) for colA in df.columns for colB in df.columns if colA != colB]
    cols_list = list(set(tuple(sorted(item)) for item in cols_list)) # drop reverse duplicte columns
    # print(cols_list)
    # new_features = pd.DataFrame(index=df.index)
    new_features = binary(df, func=interact, cols_list=cols_list)
    config["interact"] = cols_list
    new_features = pd.merge(new_features, binary(df, func=divide, cols_list=cols_list), left_index=True, right_index=True)
    config["divide"] = cols_list
    return new_features, config
    
    
def create_features_by_config(df, config):
    new_features = pd.DataFrame(index=df.index)
    for key, value in config.items():
        if isinstance(key, str):
            if key == 'quantile':
                new_features = pd.merge(new_features, unary(df, func=quantile, col_list=value), left_index=True, right_index=True)
            elif key == 'polynomial':
                new_features = pd.merge(new_features, unary(df, func=polynomial, col_list=value), left_index=True, right_index=True)
            elif key == 'interact':
                new_features = pd.merge(new_features, binary(df, func=interact, cols_list=value), left_index=True, right_index=True)
            elif key == 'divide':
                new_features = pd.merge(new_features, binary(df, func=divide, cols_list=value), left_index=True, right_index=True)
            else:
                raise("Support only quantile, polynomial, interact, divide")
        else:
            raise("Support only string keys")
    return new_features, config            


def get_df_after_arithmetic_fe(df, config=None, model=None, drop=True):
    """
    get df, caculate arithmetic features, return new df with 'good' new feature (i.e more 'normalize' featruers)
    
    :arith_dict: dictionary contains wich aritmetic to do on each row
                {'quantile': ['columnA', 'columnB'], 'division':[('columnA', 'columnB'), ('columnC', 'columnD')]}
    :model: model to get feature importance
    :drop: drop features that don't distributed noramly
    
    :return: new_df with new features, config with 
    """
    if config:
        new_features, config = create_features_by_config(df, config)
        
    else:
        # we don't want to calculate all of the options. we cacluate only on the important numeric columns 
        imp_thresh = 0.2 # can get as an argument
        numeric_df = df.select_dtypes('number')
        imp_feat = [feat for feat, imp in zip(df.columns, model.feature_importances_) if imp > imp_thresh]
        filtered_df = numeric_df[imp_feat]
        
        unary_features, un_config = get_unary_features(filtered_df)
        binary_features, bin_config = get_binary_features(filtered_df)
        
        new_features = pd.merge(unary_features, binary_features, left_index=True, right_index=True)
        un_config.update(bin_config)
        config = un_config
        
    if drop:
        pass
        # new_features, config = filter_feaetures(....) # Ariel implementation. according new_features, and update config according
        
    new_df = pd.merge(df, new_features, left_index=True, right_index=True) # maybe we want to return only the new_features, maybe according a flag

    return new_df, config