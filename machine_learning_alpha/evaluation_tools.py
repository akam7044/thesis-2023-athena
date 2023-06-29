from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def division_function(n, d):
    if d:
        return n / d
    elif n == 0 and d == 0:
        return 0
    else:
        return None


def validate_model(model, X, Y):
    """
    validates the model with a k-fold validation which is iterated
    returns the mean accuracy, specificiy, recall, precision, f1 score and auc score
    """

    splits = 5
    iteration = 10

    acc_list = []
    specificity_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    folds = StratifiedKFold(n_splits=splits)

    # Iterate "interation" times of k-fold
    for i in range(1, iteration):
        # print(f'Iteration {i}/{iteration}')

        acc_total = 0
        specificity_total = 0
        recall_total = 0
        precision_total = 0
        f1_total = 0

        for train_index, test_index in folds.split(X, Y):
            x_train = X.iloc[train_index, :]
            x_test = X.iloc[test_index, :]
            y_train = Y.iloc[train_index, :]
            y_test = Y.iloc[test_index, :]

            # scale
            sc = MinMaxScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            # fit model and predict
            model.fit(x_train, np.ravel(y_train))
            y_pred = model.predict(x_test)

            conf_matrix = confusion_matrix(y_test, y_pred)
            TN = conf_matrix[0][0]
            FP = conf_matrix[0][1]
            FN = conf_matrix[1][0]
            TP = conf_matrix[1][1]

            accuracy = (division_function((TP + TN), (TP + TN + FP + FN))) * 100
            recall = division_function(TP, (TP + FN)) * 100  # recall
            specificity = division_function(TN, (TN + FP)) * 100
            precision = division_function(TP, (TP + FP)) * 100
            f1_score = division_function(2 * (recall * precision), (recall + precision))

            # sum it up
            acc_total += accuracy
            recall_total += recall
            specificity_total += specificity
            precision_total += precision
            f1_total += f1_score

        # avg
        accuracy_mean = acc_total / splits
        recall_mean = recall_total / splits
        specificity_mean = specificity_total / splits
        precision_mean = precision_total / splits
        f1_mean = f1_total / splits

        acc_list.append(accuracy_mean)
        recall_list.append(recall_mean)
        specificity_list.append(specificity_mean)
        precision_list.append(precision_mean)
        f1_list.append(f1_mean)

    return (
        np.mean(acc_list),
        np.mean(specificity_list),
        np.mean(recall_list),
        np.mean(precision_list),
        np.mean(f1_list),
    )


def evaluate_model(model, x_train, x_test, y_train, y_test):
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    recall = (TP / (TP + FN)) * 100  # recall
    specificity = (TN / (TN + FP)) * 100
    precision = (TP / (TP + FP)) * 100
    f1_score = 2 * (recall * precision) / (recall + precision)

    return accuracy, recall, specificity, precision, f1_score
