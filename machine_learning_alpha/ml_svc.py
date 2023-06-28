import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from feature_reduction import (
    feature_reduction_lda,
    feature_reduction_mrmr,
    feature_reduction_pca,
)
from evaluation_tools import validate_model, evaluate_model

path = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena"
os.chdir(path)

CV_SPLIT = 5


def get_best_param(x_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "poly", "sigmoid"],
    }
    grid = GridSearchCV(
        SVC(), param_grid, refit=True, verbose=0, return_train_score=True, cv=CV_SPLIT
    )
    # print(grid.cv_results_)
    grid.fit(x_train, y_train)
    print(grid.best_estimator_.get_params())
    return grid


def train_test_SVC(
    filename: str,
    hold_out: bool,
    include_personal_q: bool,
    grid_search: bool,
    reduce: bool,
    model_weights: dict = {},
    random_state: int = 0,
):
    df = pd.read_csv(filename)

    if include_personal_q:
        df = df[df["noPersonalQ"] != 1].reset_index(drop=True)
    else:
        df = df[df["personalQ"] != 1].reset_index(drop=True)

    headers = df.columns
    non_embeddings_headers = []
    for header in headers:
        if header.find("embbedings") < 0:
            non_embeddings_headers.append(header)

    X = df.drop(columns=non_embeddings_headers)
    Y = df["classification"]

    # Feature Reduction
    if reduce:
        # x_train = feature_reduction_pca(x_train,0.9).values
        # x_train = feature_reduction_lda(x_train,y_train)
        X = feature_reduction_mrmr(X, Y, 20)

    else:
        x_val = X.values
        X = StandardScaler().fit_transform(x_val)

    # Test Train split
    if hold_out:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=random_state
        )
    else:
        x_train = X
        y_train = Y

    if grid_search:
        grid = get_best_param(x_train=x_train, y_train=y_train)
        model_svc = grid.best_estimator_
    else:
        model_svc = SVC(
            C=model_weights["C"],
            gamma=model_weights["gamma"],
            kernel=model_weights["kernel"],
        )

    accuracy, specificiy, recall, precision, f1_score = validate_model(
        model_svc, pd.DataFrame(x_train), pd.DataFrame(y_train)
    )
    print(
        f"\tAverage Accuracy: {accuracy} \n\
      Average Specificity: {specificiy} \n\
      Average Recall: {recall}\n\
      Average Precision:{precision}\n\
      Average F1 score {f1_score}\n\
      "
    )

    if hold_out:
        accuracy, specificiy, recall, precision, f1_score = evaluate_model(
            model_svc, x_train, x_test, y_train, y_test
        )
        print("____________________________________________")
        print("Evaluate model")
        print(
            f"\tAccuracy: {accuracy} \n\
        Specificity: {specificiy} \n\
        Recall: {recall}\n\
        Precision:{precision}\n\
        F1 score {f1_score}\n\
        "
        )
        return accuracy, specificiy, recall, precision, f1_score
    else:
        return accuracy, specificiy, recall, precision, f1_score
