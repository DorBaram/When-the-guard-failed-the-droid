import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

def dividedataset():
    print("Dividing dataset into 80 20") 
    df = pd.read_csv('data.csv')
    df = df[df.category != 2] # Removes all rows where the 'category' column is manipulated
    df = df.drop_duplicates(subset=['name']) # Removes duplicate rows based on the 'name' column.
    #Splits 'df' into two dataframes, 'benign' and 'mallicius', based on the value of the 'type' column.
    df_b = df[df.type == 0] #benign
    df_m = df[df.type == 1] #mallicius
    df_b.group_num = 0
    df_m.group_num = 0
     # Randomly samples 20% of the rows from 'df_m' and assigns the value 0 to the 'category' column for those rows
    df_m = df_m.sample(frac = 0.2)
    df_m.category = 0
    df_b.category = 0
    print(len(df_b))    #should be 60k +-
    print(len(df_m))    #should be 6k +- 
    # split the benign into train and test
    train_b = df_b.sample(frac = 0.6)
    train_b.category = 0
    test_b = df_b.drop(train_b.index)
    test_b.category = 1
    # shuffle the train and test set
    shuffled_train_b = train_b.sample(frac=1)
    result_train_b = np.array_split(shuffled_train_b, 5)
    shuffled_test_b = test_b.sample(frac=1)
    result_test_b = np.array_split(shuffled_test_b, 5)
    # split the malicious into train and test
    train_m = df_m.sample(frac = 0.6)
    train_m.category = 0
    test_m = df_m.drop(train_m.index)
    test_m.category = 1
    # shuffle the train and test set
    shuffled_train_m = train_m.sample(frac=1)
    result_train_m = np.array_split(shuffled_train_m, 5)
    shuffled_test_m = test_m.sample(frac=1)
    result_test_m = np.array_split(shuffled_test_m, 5)

    # Assign the group number (0-4) to the 'group_num' column for each group -> cross validation to not be overfitted 
    for i in range(5):
        result_train_b[i].group_num = i
        result_test_b[i].group_num = i
        result_train_m[i].group_num = i
        result_test_m[i].group_num = i
        # Combining all the groups into single dataframe
        if i == 0:
            train_b = result_train_b[i]
            test_b = result_test_b[i]
            train_m = result_train_m[i]
            test_m = result_test_m[i]
        else:
            train_b = train_b.combine_first(result_train_b[i])
            test_b = test_b.combine_first(result_test_b[i])
            train_m = train_m.combine_first(result_train_m[i])
            test_m = test_m.combine_first(result_test_m[i])
    #Combining all the dataframe into a single dataframe
    df_dataframe = train_b.combine_first(train_m.combine_first(test_b.combine_first(test_m)))
    df_dataframe.to_csv("dataset.csv") #Exports the data to the csv
    return df

dividedataset()
