from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    StandardScaler,
    LinearDiscriminantAnalysis as LDA,
)
import sys


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


CV_SPLIT = 5

"""
GridSearch for parameter optimisation
"""


def get_reducer_variables(reducer_name: str):
    if reducer_name == "pca":
        return {"reducer": PCA(), "variables": [0.8, 0.9, 0.95]}
    elif reducer_name == "lda":
        return {"reducer": LDA(), "variables": [1]}


def get_best_param_RF(x_train, y_train, reducer_name):
    if reducer_name != "mrmr":
        reducer = get_reducer_variables(reducer_name=reducer_name)
        reducer_var = reducer["variables"]
        reducer_name = reducer["reducer"]

        pipe = Pipeline(
            steps=[("reducer", reducer_name), ("rf", RandomForestClassifier())]
        )

        param_grid = {
            "rf__n_estimators": [100, 200, 500],
            "rf__max_features": ["auto", "sqrt", "log2"],
            "rf__max_depth": [4, 5, 6, 7, 8],
            "rf__criterion": ["gini", "entropy"],
            "reducer__n_components": reducer_var,
        }
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            refit=True,
            verbose=0,
            return_train_score=True,
            cv=CV_SPLIT,
        )
        grid.fit(x_train, y_train)
        extra_print(str(grid.best_estimator_.get_params()))

        return grid
    else:
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


def get_best_param_KNN(x_train, y_train, reducer_name):
    if reducer_name != "mrmr":
        reducer = get_reducer_variables(reducer_name=reducer_name)
        reducer_var = reducer["variables"]
        reducer_name = reducer["reducer"]

        pipe = Pipeline(
            steps=[("reducer", reducer_name), ("knn", KNeighborsClassifier())]
        )

        param_grid = {
            "knn__n_neighbors": range(1, 21, 2),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan", "minkowski"],
            "reducer__n_components": reducer_var,
        }
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            refit=True,
            verbose=0,
            return_train_score=True,
            cv=CV_SPLIT,
        )
        grid.fit(x_train, y_train)
        extra_print(str(grid.best_estimator_.get_params()))

        return grid
    else:
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


def get_best_param_LR(x_train, y_train, reducer_name):
    if reducer_name != "mrmr":
        reducer = get_reducer_variables(reducer_name=reducer_name)
        reducer_var = reducer["variables"]
        reducer_name = reducer["reducer"]

        pipe = Pipeline(steps=[("reducer", reducer_name), ("lr", LogisticRegression())])

        param_grid = {
            "lr__C": [100, 10, 1.0, 0.1, 0.01],
            "lr__solver": ["newton-cg", "lbfgs", "liblinear"],
            "lr__penalty": ["l1", "l2"],
            "reducer__n_components": reducer_var,
        }
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            refit=True,
            verbose=0,
            return_train_score=True,
            cv=CV_SPLIT,
        )
        grid.fit(x_train, y_train)
        extra_print(str(grid.best_estimator_.get_params()))

        return grid
    else:
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


def get_best_param_SVC(x_train, y_train, reducer_name):
    if reducer_name != "mrmr":
        reducer = get_reducer_variables(reducer_name=reducer_name)
        reducer_var = reducer["variables"]
        reducer_name = reducer["reducer"]

        pipe = Pipeline(steps=[("reducer", reducer_name), ("svc", SVC())])

        param_grid = {
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": [1, 0.1, 0.01, 0.001],
            "svc__kernel": ["rbf", "poly", "sigmoid"],
            "reducer__n_components": reducer_var,
        }
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            refit=True,
            verbose=0,
            return_train_score=True,
            cv=CV_SPLIT,
        )
        grid.fit(x_train, y_train)
        extra_print(str(grid.best_estimator_.get_params()))

        return grid
    else:
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
            "kernel": ["rbf", "poly", "sigmoid"],
        }
        grid = GridSearchCV(
            SVC(),
            param_grid,
            refit=True,
            verbose=0,
            return_train_score=True,
            cv=CV_SPLIT,
        )
        grid.fit(x_train, y_train)
        extra_print(str(grid.best_estimator_.get_params()))
        return grid
