import concurrent.futures
import math

# XGBOOST###############################################################################################
from xgboost import XGBClassifier


def xgboost(X_train, y_train, X_test, X_test_manipulated, num_estimators, max_depth, learning_rate,):
    clf = XGBClassifier(
        n_estimators=num_estimators, max_depth=max_depth, learning_rate=learning_rate
    )
    clf.fit(X_train, y_train)
    return (
        clf.predict(X_test),
        clf.predict(X_test_manipulated),
        num_estimators,
        max_depth,
        learning_rate,
    )

def xg_boost( num_of_features, features, group, group_mani, report, X_train, X_test, X_test_manipulated, y_train, y_test, y_test_manipulated,):
    print(group + ": xgboost")
    dict1 = {
        "num of trees": 0,
        "max_depth": 0,
        "learning_rate": 0,
        "recall": 0,
        "confusion matrix": 0,
    }
    dict2 = {"recall": 0, "confusion matrix": 0}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            executor.submit(
                xgboost,
                X_train,
                y_train,
                X_test,
                X_test_manipulated,
                n,
                max_depth,
                learning_rate,
            )
            for n in [110]
            for max_depth in [3, 5, 10, 20, 30]
            for learning_rate in [0.1, 0.01, 0.001]
        ]
    max1, max2 = 0, 0
    for f_ in concurrent.futures.as_completed(results):
        result = f_.result()
        get_perfomances(y_test, result[0], dict1)
        get_perfomances(y_test_manipulated, result[1], dict2)
        dict1["num of trees"] = result[2]
        dict1["max_depth"] = result[3]
        dict1["learning_rate"] = result[4]
        x = [
            group,
            group_mani,
            " ",
            num_of_features,
            features,
            dict1["num of trees"],
            dict1["max_depth"],
            dict1["learning_rate"],
            " ",
            dict1["recall"],
            dict1["confusion matrix"],
            " ",
            dict2["recall"],
            dict2["confusion matrix"],
            " ",
            X_test_manipulated.shape[0],
        ]
        r1, r2 = dict1["recall"], dict2["recall"]
        if r1 > max1:
            max1 = r1
            x1 = x

        if r2 > max2:
            max2 = r2
            x2 = x

    if max1 != 0:
        write_to_csv(report, x1)
    if max2 != 0:
        write_to_csv(report, x2)


# reports.py###########################################################################################
def randomForest(
    X_train,
    y_train,
    X_test,
    X_test_manipulated,
    num_estimators,
    criterion,
    min_samples_split,
):
    # def xgboost(X_train, y_train, X_test, X_test_manipulated, num_estimators, max_depth, learning_rate):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=num_estimators,
        criterion=criterion,
        min_samples_split=min_samples_split,
    )
    clf.fit(X_train, y_train)
    return (
        clf.predict(X_test),
        clf.predict(X_test),
        num_estimators,
        criterion,
        min_samples_split,
    )


# X_train, X_test,X_test_manipulated, y_train, y_test,y_test_manipulated
def random_forest(
    num_of_features,
    features,
    group,
    group_mani,
    report,
    X_train,
    X_test,
    X_test_manipulated,
    y_train,
    y_test,
    y_test_manipulated,
):
    print(group + ": rf")
    dict1 = {
        "num of trees": 0,
        "criterion": 0,
        "min_samples_split": 0,
        "recall": 0,
        "confusion matrix": 0,
    }
    dict2 = {"recall": 0, "confusion matrix": 0}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            executor.submit(
                randomForest,
                X_train,
                y_train,
                X_test,
                X_test_manipulated,
                n,
                criteria,
                min_samples_split,
            )
            for n in [100]
            for criteria in ["entropy", "gini"]
            for min_samples_split in [3]
        ]
    max1, max2 = 0, 0
    for f_ in concurrent.futures.as_completed(results):
        result = f_.result()
        get_perfomances(y_test, result[0], dict1)
        get_perfomances(y_test_manipulated, result[1], dict2)
        dict1["num of trees"] = result[2]
        dict1["criterion"] = result[3]
        dict1["min_samples_split"] = result[4]
        x = [
            group,
            group_mani,
            " ",
            num_of_features,
            features,
            dict1["num of trees"],
            dict1["criterion"],
            dict1["min_samples_split"],
            " ",
            dict1["recall"],
            dict1["confusion matrix"],
            " ",
            dict2["recall"],
            dict2["confusion matrix"],
            " ",
            X_test_manipulated.shape[0],
        ]
        r1, r2 = dict1["recall"], dict2["recall"]
        if r1 > max1:
            max1 = r1
            x1 = x

        if r2 > max2:
            max2 = r2
            x2 = x

    if max1 != 0:
        write_to_csv(report, x1)
    if max2 != 0:
        write_to_csv(report, x2)


def get_perfomances(y, y_pred, dict):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        recall_score,
        precision_score,
    )

    import numpy

    recall = recall_score(y, y_pred, zero_division=1)
    dict["confusion matrix"] = numpy.array_str(confusion_matrix(y, y_pred))
    dict["recall"] = round(recall_score(y, y_pred, zero_division=1), 2)


def write_to_csv(file_name, list_):
    import csv

    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(list_)


# machine.py###########################################################################################
def build_reports(reports):
    # the function builds a new report with titles
    import csv

    for r in reports:
        with open(r[0], "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(r[1])


def get_data(data, list_):
    # the function returns a dataframe with the needed features for the machine
    all_columns = data.columns.to_list()
    data_columns = ["name", "type", "group_num", "group_mani", "category"]
    for l in list_:
        data_columns += [c for c in all_columns if l in c]
    # print(len(data_columns))
    return data.loc[:, data_columns]


def divide_data(data):
    # the dunction divides the data to 5 groups
    import pandas as pd

    pd.set_option("mode.chained_assignment", None)
    train_data, test_data, test_manipulated_data = (
        data[data.category == 0],
        data[data.category == 1],
        data[data.category == 2],
    )
    train_data.drop(columns=["category"], inplace=True)
    test_data.drop(columns=["category"], inplace=True)
    test_manipulated_data.drop(columns=["category"], inplace=True)
    train, test, mani = [], [], []

    for i in range(5):
        train_ = train_data[(train_data.type == 0) | (train_data.group_num == i)]
        train_.reset_index(drop=True, inplace=True)
        train_.drop(columns=["group_num", "group_mani"], inplace=True)
        test_ = test_data[(test_data.group_num == i)]
        test_.drop(columns=["group_num", "group_mani"], inplace=True)
        test_.reset_index(drop=True, inplace=True)
        mani_ = test_manipulated_data[(test_manipulated_data.group_num == i)]
        mani_.drop(columns=["group_num"], inplace=True)
        m = []
        t = 0
        for j in range(1, 7, 1):
            m.append(mani_[mani_.group_mani == j])
            m[t].reset_index(drop=True, inplace=True)
            m[t].drop(columns=["group_mani"], inplace=True)
            t += 1
        train.append(train_)
        test.append(test_)
        mani.append(m)
        # print(train[i].shape,test[i].shape,mani[i][0].shape,mani[i][1].shape,mani[i][2].shape,mani[i][3].shape,mani[i][4].shape,mani[i][5].shape)
    return train, test, mani


def div_data(data, i):
    # the dunction divides the data to 5 groups
    import pandas as pd

    pd.set_option("mode.chained_assignment", None)
    train_data, test_data, test_manipulated_data = (
        data[data.category == 0],
        data[data.category == 1],
        data[data.category == 2],
    )
    train_data.drop(columns=["category"], inplace=True)
    test_data.drop(columns=["category"], inplace=True)
    test_manipulated_data.drop(columns=["category"], inplace=True)
    train_data = train_data[(train_data.type == 0) | (train_data.group_num == i)]
    train_data.reset_index(drop=True, inplace=True)
    train_data.drop(columns=["group_num", "group_mani"], inplace=True)
    test_data = test_data[test_data.group_num == i]
    test_data.drop(columns=["group_num", "group_mani"], inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_manipulated_data.drop(columns=["group_num"], inplace=True)
    m = []
    t = 0

    for j in range(1, 7, 1):
        m.append(test_manipulated_data[test_manipulated_data.group_mani == j])
        m[t].reset_index(drop=True, inplace=True)
        m[t].drop(columns=["group_mani"], inplace=True)
        t += 1
    # print(train[i].shape,test[i].shape,mani[i][0].shape,mani[i][1].shape,mani[i][2].shape,mani[i][3].shape,mani[i][4].shape,mani[i][5].shape)
    return train_data, test_data, m


def get_X_y_features(num_of_features, features, group, test, train, mani):
    import pandas as pd

    j = group
    top_features = features[j][0:num_of_features]
    train_ = train[j].loc[:, top_features + ["type"]]
    test_ = test.loc[:, top_features + ["type"]]
    mani_ = test.loc[:, top_features + ["type"]]
    X_train = train_.loc[:, ~train_.columns.isin(["type"])]
    X_test = test_.loc[:, ~test_.columns.isin(["type"])]
    X_mani = test_.loc[:, ~test_.columns.isin(["type"])]
    y_train = train_.type
    y_test = test_.type
    y_mani = mani_.type
    return X_train, X_test, X_mani, y_train, y_test, y_mani, top_features


def keep_same_apks(test, mani):
    test["cat"] = 1
    mani["cat"] = 2
    x = test
    test = x[x.cat == 1]
    test.drop(columns=["cat"], inplace=True)
    return test, mani

