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
from evatluation_tools_transcripts import validate_model, evaluate_model
from feature_reduction import (
    feature_reduction_lda,
    feature_reduction_mrmr,
    feature_reduction_pca,
)
from grid_search import (
    get_best_param_KNN,
    get_best_param_LR,
    get_best_param_RF,
    get_best_param_SVC,
)
import sys

path = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena"
os.chdir(path)


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


"""
Reducing the features, splitting the data into test and train, oversample the training data, train the model and validate and evaluate it
"""


def read_and_split(
    filename: str, isTranscript: bool, reduce: str, random_state: int, chunked: bool
):
    df = pd.read_csv(filename)

    if isTranscript:
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

        # Oversample minority group
        sm = SMOTE(random_state=12)
        x_train, y_train = sm.fit_resample(x_train, y_train)

        # Feature Reduction
        if reduce == "pca":
            x_train, x_test = feature_reduction_pca(x_train, x_test, 0.9)
        elif reduce == "lda":
            x_train, x_test = feature_reduction_lda(x_train, x_test, y_train)
        elif reduce == "mrmr":
            x_train, x_test = feature_reduction_mrmr(
                pd.DataFrame(x_train), pd.DataFrame(x_test), pd.DataFrame(y_train), 30
            )

    else:
        if chunked:
            df.drop(["voiceID", "label_x"], inplace=True, axis=1)
            df.rename(columns={"label_y": "label"}, inplace=True)
        else:
            df.drop(["voiceID"], inplace=True, axis=1)
        df["label"].value_counts()
        df = df.dropna()

        df_X = df.iloc[:, :-1]
        df_Y = df.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(
            df_X, df_Y, test_size=0.3, random_state=random_state
        )

        sc = MinMaxScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        pd.DataFrame(x_train)

    return x_train, x_test, y_train, y_test, x_test_index, df


def train_model(model_name: str, grid_search: bool, model_weights, x_train, y_train):
    if model_name == "svc":
        if grid_search:
            grid = get_best_param_SVC(x_train=x_train, y_train=y_train)
            model = grid.best_estimator_
        else:
            model = SVC(
                C=model_weights["C"],
                gamma=model_weights["gamma"],
                kernel=model_weights["kernel"],
            )
    elif model_name == "lr":
        if grid_search:
            grid = get_best_param_LR(x_train=x_train, y_train=y_train)
            model = grid.best_estimator_
        else:
            model = LogisticRegression(
                C=model_weights["C"],
                solver=model_weights["solver"],
                penalty=model_weights["penalty"],
            )
    elif model_name == "knn":
        if grid_search:
            grid = get_best_param_KNN(x_train=x_train, y_train=y_train)
            model = grid.best_estimator_
        else:
            model = KNeighborsClassifier(
                n_neighbors=model_weights["n_neighbors"],
                weights=model_weights["weights"],
                metric=model_weights["metric"],
            )
    elif model_name == "rf":
        if grid_search:
            grid = get_best_param_RF(x_train=x_train, y_train=y_train)
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
    chunked: bool = False,
    reduce: str = "mrmr",
    isTranscript: bool = True,
    grid_search: bool = True,
    model_weights: dict = {},
    random_state: int = 42,
):
    x_train, x_test, y_train, y_test, x_test_index, df = read_and_split(
        filename=filename,
        isTranscript=isTranscript,
        reduce=reduce,
        random_state=random_state,
        chunked=chunked,
    )

    # Train ML model
    model = train_model(
        model_name=model_name,
        grid_search=grid_search,
        model_weights=model_weights,
        x_train=x_train,
        y_train=y_train,
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

    return accuracy, specificiy, recall, precision, f1_score
