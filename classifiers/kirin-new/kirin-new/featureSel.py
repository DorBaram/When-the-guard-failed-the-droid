import pandas as pd
import numpy as np
    
def RFC(data):
    data.drop(columns=["name"], inplace=True)   # drop the name column and separate the target variable
    y = data.loc[:, "type"].values
    X = data.drop(columns=["type"])
    features = X.columns.to_list()
    X = X.values  
    # train the classifier 
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=110)
    rf.fit(X, y)
    importances = rf.feature_importances_   # get the feature importances
    df = pd.DataFrame(data={"features": features, "ranking": importances})  # combine features names and importances
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)   # sort the features by importance and return the top features
    top_features = df.features.to_list()
    return top_features

def DTC(data):
    data.drop(columns=["name"], inplace=True)   # drop the name column and separate the target variable
    y = data.loc[:, "type"].values
    X = data.drop(columns=["type"])
    features = X.columns.to_list()
    X = X.values  
    # train the classifier 
    from sklearn.tree import DecisionTreeClassifier
    rf = DecisionTreeClassifier()  
    rf.fit(X, y)
    importances = rf.feature_importances_   # get the feature importances
    df = pd.DataFrame(data={"features": features, "ranking": importances})  # combine features names and importances
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)   # sort the features by importance and return the top features
    top_features = df.features.to_list()
    return top_features

def ABC(data):
    data.drop(columns=["name"], inplace=True)   # drop the name column and separate the target variable
    y = data.loc[:, "type"].values
    X = data.drop(columns=["type"])
    features = X.columns.to_list()
    X = X.values  
    # train the classifier 
    from sklearn.ensemble import AdaBoostClassifier
    rf = AdaBoostClassifier()
    rf.fit(X, y)
    importances = rf.feature_importances_   # get the feature importances
    df = pd.DataFrame(data={"features": features, "ranking": importances})  # combine features names and importances
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)   # sort the features by importance and return the top features
    top_features = df.features.to_list()
    return top_features

def ETC(data):
    data.drop(columns=["name"], inplace=True)   # drop the name column and separate the target variable
    y = data.loc[:, "type"].values
    X = data.drop(columns=["type"])
    features = X.columns.to_list()
    X = X.values  
    # train the classifier 
    from sklearn.ensemble import ExtraTreesClassifier
    rf = ExtraTreesClassifier()
    rf.fit(X, y)
    importances = rf.feature_importances_   # get the feature importances
    df = pd.DataFrame(data={"features": features, "ranking": importances})  # combine features names and importances
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)   # sort the features by importance and return the top features
    top_features = df.features.to_list()
    return top_features

def GBC(data):
    data.drop(columns=["name"], inplace=True)   # drop the name column and separate the target variable
    y = data.loc[:, "type"].values
    X = data.drop(columns=["type"])
    features = X.columns.to_list()
    X = X.values  
    # train the classifier 
    from sklearn.ensemble import GradientBoostingClassifier
    rf = GradientBoostingClassifier()
    rf.fit(X, y)
    importances = rf.feature_importances_   # get the feature importances
    df = pd.DataFrame(data={"features": features, "ranking": importances})  # combine features names and importances
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)   # sort the features by importance and return the top features
    top_features = df.features.to_list()
    return top_features


