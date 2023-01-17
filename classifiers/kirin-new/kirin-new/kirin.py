import helper
import featureSel
import pandas as pd
import numpy as np
import time


def run(data, reports_):
    print(
        "Creating 'kirin.csv' and starting..."
    )  # If already created, it will replace it
    reports_path = [
        reports_ + "kirin.csv"
    ]  # Define the path to the reports file and the report items
    reports_items = [
        [
            "group",
            "attack",
            " ",
            "test_recall",
            "test_confusion_matrix",
            " ",
            "test_manipulated_recall",
            "test_manipulatedconfusion_matrix",
            " ",
            "num of observations",
        ]
    ]
    reports_ = tuple(
        zip(reports_path, reports_items)
    )  # Zip the reports path and items into a tuple
    helper.build_reports(reports_)  # Build the reports
    data = pd.read_csv(data)  # Read the data file into a pandas DataFrame
    data = helper.get_data(
        data,
        [
            "perm: ",
        ],
    )  # Preprocess the data

    train, test, mani = helper.divide_data(
        data
    )  # Split the data into train, test, and mani sets (manipulated test is not used)
    features = []  # Initialize an empty list to store the features
    NumOfTimes = (
        5  # Set the number of times to run the feature selection, should remain at 5
    )
    print("Running feature selection")
    for i in range(NumOfTimes):  # Run the feature selection numOfTimes times
        features.append(
            featureSel.RFC(train[i])
        )  # Select the features using the decision tree method

    print("Running kirin")
    # Set the number of features to use
    numOfFeatures = 110  # can change
    # in DecisionTreeClassifier:
    # 80->65, 85->64.2, 90->65, 95->64.5,
    # 105->65.2,
    # 110->65.6, 120->65, 130->64.4, 140->64.4,
    # 100->65.2, 150->64.4, 200->64.4, 250->64, 300->63.8

    for group in range(NumOfTimes):
        mani_group = 0
        m, t = helper.keep_same_apks(mani[group][mani_group], test[group])
        (
            X_train,
            X_test,
            X_mani,
            y_train,
            y_test,
            y_mani,
            top_features,
        ) = helper.get_X_y_features(
            numOfFeatures, features, group, t, train, m
        )  # Get the X and y arrays and the top features

        #if (NumOfTimes % 2) == 1:
        if (1) == 1:
            helper.xg_boost(  # Run the xgboost classifier
                numOfFeatures,
                top_features,
                str(group),
                str(mani_group + 1),
                reports_[0][0],
                X_train,
                X_test,
                X_mani,  #
                y_train,
                y_test,
                y_mani,
            )
        else:
            helper.random_forest(  # Run the random forest classifier
                numOfFeatures,
                top_features,
                str(group),
                str(mani_group + 1),
                reports_[0][0],
                X_train,
                X_test,
                X_mani,  #
                y_train,
                y_test,
                y_mani,
            )


if __name__ == "__main__":
    start_time = time.time()
    data = "dataset.csv"
    reports_ = ""  # "reports\\"
    run(data, reports_)
    print("Done, check csv")
    end_time = time.time()
    print("Runtime: ", "{:.2f}".format(end_time - start_time))

