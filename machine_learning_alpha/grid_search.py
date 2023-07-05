from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


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
    extra_print(str(grid.best_estimator_.get_params()))
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
    extra_print(str(grid.best_estimator_.get_params()))
    return grid


def get_best_param_LR(x_train, y_train):
    param_grid = {
        "C": [100, 10, 1.0, 0.1, 0.01],
        "solver": ["newton-cg", "lbfgs", "liblinear"],
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
    extra_print(str(grid.best_estimator_.get_params()))
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
    extra_print(str(grid.best_estimator_.get_params()))
    return grid
