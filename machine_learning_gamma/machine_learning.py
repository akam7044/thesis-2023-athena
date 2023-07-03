import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import sys

from evaluation_tools import evaluate_model, validate_model
from grid_search import (
    get_best_param_KNN,
    get_best_param_LR,
    get_best_param_RF,
    get_best_param_SVC,
)
from feature_reduction import feature_reduction_mrmr


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


"""
Reducing the features, splitting the data into test and train, oversample the training data, train the model and validate and evaluate it
"""


def read_and_split(filename: str, reduce: str, random_state: int):
    df = pd.read_csv(filename)

    # Remove Personal Questions
    df = df[df["personalQ"] != 1].reset_index(drop=True)

    headers = df.columns
    non_embeddings_headers = []
    for header in headers:
        if header.find("embbedings") < 0:
            non_embeddings_headers.append(header)

    X = df.drop(columns=non_embeddings_headers)
    Y = df["classification"]
    X = MinMaxScaler().fit_transform(X)

    # Train-test split
    x_index = range(len(X))
    x_train_index, x_test_index, y_train, y_test = train_test_split(
        x_index, Y, test_size=0.30, random_state=random_state
    )
    x_train = pd.DataFrame(X).iloc[x_train_index]
    x_test = pd.DataFrame(X).iloc[x_test_index]

    # sc = MinMaxScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)
    # Oversample minority group
    sm = SMOTE(random_state=12)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    if reduce == "mrmr":
        x_train, x_test = feature_reduction_mrmr(
            pd.DataFrame(x_train), pd.DataFrame(x_test), pd.DataFrame(y_train), 30
        )

    return x_train, x_test, y_train, y_test, x_test_index, df


def train_model(
    model_name: str, grid_search: bool, model_weights, x_train, y_train, reducer
):
    if model_name == "svc":
        if grid_search:
            grid = get_best_param_SVC(
                x_train=x_train, y_train=y_train, reducer_name=reducer
            )
            model = grid.best_estimator_
        else:
            model = SVC(
                C=model_weights["C"],
                gamma=model_weights["gamma"],
                kernel=model_weights["kernel"],
            )
    elif model_name == "lr":
        if grid_search:
            grid = get_best_param_LR(
                x_train=x_train, y_train=y_train, reducer_name=reducer
            )
            model = grid.best_estimator_
        else:
            model = LogisticRegression(
                C=model_weights["C"],
                solver=model_weights["solver"],
                penalty=model_weights["penalty"],
            )
    elif model_name == "knn":
        if grid_search:
            grid = get_best_param_KNN(
                x_train=x_train, y_train=y_train, reducer_name=reducer
            )
            model = grid.best_estimator_
        else:
            model = KNeighborsClassifier(
                n_neighbors=model_weights["n_neighbors"],
                weights=model_weights["weights"],
                metric=model_weights["metric"],
            )
    elif model_name == "rf":
        if grid_search:
            grid = get_best_param_RF(
                x_train=x_train, y_train=y_train, reducer_name=reducer
            )
            model = grid.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=model_weights["n_estimators"],
                max_features=model_weights["max_features"],
                max_depth=model_weights["max_depth"],
                criterion=model_weights["criterion"],
            )

    return model


def train_test_model(
    filename: str,
    model_name: str,
    reduce: str = "mrmr",
    grid_search: bool = True,
    model_weights: dict = {},
    random_state: int = 0,
):
    x_train, x_test, y_train, y_test, x_test_index, df = read_and_split(
        filename=filename, reduce=reduce, random_state=random_state
    )

    # Train ML model
    model = train_model(
        model_name=model_name,
        grid_search=grid_search,
        model_weights=model_weights,
        x_train=x_train,
        y_train=y_train,
        reducer=reduce,
    )

    # Validate with training data
    accuracy, specificiy, recall, precision, f1_score = validate_model(
        model, pd.DataFrame(x_train), pd.DataFrame(y_train)
    )

    extra_print(
        f"\tAverage Accuracy: {accuracy} \n\
      Average Specificity: {specificiy} \n\
      Average Recall: {recall}\n\
      Average Precision:{precision}\n\
      Average F1 score {f1_score}\n\
      "
    )

    # Test with test data
    accuracy, specificiy, recall, precision, f1_score = evaluate_model(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_test_index=x_test_index,
        df=df,
    )
    extra_print("___________________")
    extra_print("Evaluate model")
    extra_print(
        f"\tAccuracy: {accuracy} \n\
    Specificity: {specificiy} \n\
    Recall: {recall}\n\
    Precision:{precision}\n\
    F1 score {f1_score}\n\
    "
    )

    return accuracy, specificiy, recall, precision, f1_score, model
