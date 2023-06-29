import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from feature_reduction import (
    feature_reduction_lda,
    feature_reduction_mrmr,
    feature_reduction_pca,
)
from evaluation_tools import validate_model, evaluate_model
from imblearn.over_sampling import SMOTE

CV_SPLIT = 5

"""
GridSearch for parameter optimisation
"""


def get_best_param_RF(x_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        refit=True,
        verbose=0,
        return_train_score=True,
        cv=CV_SPLIT,
    )
    grid.fit(x_train, y_train)
    print(grid.best_estimator_.get_params())
    return grid


def get_best_param_KNN(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        refit=True,
        verbose=0,
        return_train_score=True,
        cv=CV_SPLIT,
    )
    grid.fit(x_train, y_train)
    print(grid.best_estimator_.get_params())
    return grid


def get_best_param_LR(x_train, y_train):
    param_grid = {
        "C": [100, 10, 1.0, 0.1, 0.01],
        "solvers": ["newton-cg", "lbfgs", "liblinear"],
        "penalty": ["l1", "l2"],
    }
    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        refit=True,
        verbose=0,
        return_train_score=True,
        cv=CV_SPLIT,
    )
    grid.fit(x_train, y_train)
    print(grid.best_estimator_.get_params())
    return grid


def get_best_param_SVC(x_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "poly", "sigmoid"],
    }
    grid = GridSearchCV(
        SVC(), param_grid, refit=True, verbose=0, return_train_score=True, cv=CV_SPLIT
    )
    grid.fit(x_train, y_train)
    print(grid.best_estimator_.get_params())
    return grid


"""
Reducing the features, splitting the data into test and train, oversample the training data, train the model and validate and evaluate it
"""


def train_test_model(
    filename: str,
    reduce: str,
    model_name: str,
    grid_search: bool = True,
    model_weights: dict = {},
    random_state: int = 0,
):
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

    # Feature Reduction
    if reduce == "pca":
        X = feature_reduction_pca(X, 0.9).values
    elif reduce == "lda":
        X = feature_reduction_lda(X, Y)
    elif reduce == "mrmr":
        X = feature_reduction_mrmr(X, Y, 20)
    else:
        x_val = X.values
        X = StandardScaler().fit_transform(x_val)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.30, random_state=random_state
    )

    # Oversample minority group
    sm = SMOTE(random_state=12)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    # Train ML model
    if model_name == "svc":
        if grid_search:
            grid = get_best_param_SVC(x_train=x_train, y_train=y_train)
            model_svc = grid.best_estimator_
        else:
            model_svc = SVC(
                C=model_weights["C"],
                gamma=model_weights["gamma"],
                kernel=model_weights["kernel"],
            )
    elif model_name == "lr":
        if grid_search:
            grid = get_best_param_LR(x_train=x_train, y_train=y_train)
            model_svc = grid.best_estimator_
        else:
            model_svc = LogisticRegression(
                C=model_weights["C"],
                solver=model_weights["solver"],
                penalty=model_weights["penalty"],
            )
    elif model_name == "knn":
        if grid_search:
            grid = get_best_param_KNN(x_train=x_train, y_train=y_train)
            model_svc = grid.best_estimator_
        else:
            model_svc = KNeighborsClassifier(
                n_neighbors=model_weights["n_neighbors"],
                weights=model_weights["weights"],
                metric=model_weights["metric"],
            )
    elif model_name == "rf":
        if grid_search:
            grid = get_best_param_RF(x_train=x_train, y_train=y_train)
            model_svc = grid.best_estimator_
        else:
            model_svc = RandomForestClassifier(
                n_estimators=model_weights["n_estimators"],
                max_features=model_weights["max_features"],
                max_depth=model_weights["max_depth"],
                criterion=model_weights["criterion"],
            )

    # Validate with training data
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

    # Test with test data
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


"""
As the dataset is small, different random states for train_test_split will be used
"""


def train_test_SVC_TEST_avg_seeds(
    filename: str,
    reduce: str,
    model_name: str,
    grid_search: bool = True,
    model_weights: dict = {},
):
    random_states = [0, 5, 13, 27, 36, 42]

    acc_list = []
    specificity_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    for random_state in random_states:
        print(f"Random State: {random_state}")
        accuracy, specificiy, recall, precision, f1_score = train_test_model(
            filename,
            reduce,
            model_name,
            grid_search,
            model_weights,
            random_state,
        )
        acc_list.append(accuracy)
        specificity_list.append(specificiy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1_score)

    accuracy_mean = np.mean(acc_list)
    recall_mean = np.mean(recall_list)
    specificity_mean = np.mean(specificity_list)
    precision_mean = np.mean(precision_list)
    f1_mean = np.mean(f1_list)

    print(
        "Accuracy list: ", acc_list
    )  # mean accuracy acros the 6 folds for each iteration
    print(
        "Recall list: ", recall_list
    )  # mean accuracy acros the 6 folds for each iteration
    print("Specificity list: ", specificity_list)
    print("Precision list: ", precision_list)
    print("F1  score list: ", f1_list)

    print(
        f"\tAverage Accuracy: {accuracy_mean} \n\
      Average Specificity: {specificity_mean} \n\
      Average Recall: {recall_mean}\n\
      Average Precision:{precision_mean}\n\
      Average F1 score {f1_mean}\n\
      "
    )
