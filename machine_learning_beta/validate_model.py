from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def weird_division(n, d):
    return n / d if d else 0


def validate_model(model, X, Y, fold):
    """
    validates the model with a k-fold validation which is iterated
    returns the mean accuracy, specificiy, recall, precision, f1 score and auc score
    """
    decision_thresholds = []  # some list
    splits = 6
    iteration = 10

    acc_list = []
    specificity_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    if fold == "Strat":
        folds = StratifiedKFold(n_splits=splits)
    elif fold == "K":
        folds = KFold(splits, shuffle=True)

    # Iterate "interation" times of k-fold
    for i in range(1, iteration):
        # print(f'Iteration {i}/{iteration}')

        acc_total = 0
        specificity_total = 0
        recall_total = 0
        precision_total = 0
        f1_total = 0
        # auc_total = 0

        for train_index, test_index in folds.split(X, Y):
            # x_train,y_train,x_test,y_test = X.iloc[train_index,:], Y.iloc[train_index,:], X.iloc[test_index,:],Y.iloc[test_index,:]
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
            # y_pred = model.predict(x_test)

            y_proba = model.predict_proba(x_test)
            y_pred = []

            for decision_threshold in decision_thresholds:
                for y_prob in y_proba:
                    if y_prob > decision_threshold:
                        y_pred.append(1)  # same format as y_pred
                    else:
                        y_pred.append(0)

            conf_matrix = confusion_matrix(y_test, y_pred)

            print(conf_matrix)
            TN = conf_matrix[0][0]
            FP = conf_matrix[0][1]
            FN = conf_matrix[1][0]
            TP = conf_matrix[1][1]

            accuracy = (weird_division((TP + TN), (TP + TN + FP + FN))) * 100
            recall = weird_division(TP, (TP + FN)) * 100  # recall
            specificity = weird_division(TN, (TN + FP)) * 100
            precision = weird_division(TP, (TP + FP)) * 100
            f1_score = weird_division(2 * (recall * precision), (recall + precision))

            # sum it up
            acc_total += accuracy
            recall_total += recall
            specificity_total += specificity
            precision_total += precision
            f1_total += f1_score
            # auc_total += roc_auc_score(y_test, y_pred)

        # avg
        accuracy_mean = acc_total / splits
        recall_mean = recall_total / splits
        specificity_mean = specificity_total / splits
        precision_mean = precision_total / splits
        f1_mean = f1_total / splits
        # auc_mean = auc_total / splits

        acc_list.append(accuracy_mean)
        recall_list.append(recall_mean)
        specificity_list.append(specificity_mean)
        precision_list.append(precision_mean)
        f1_list.append(f1_mean)
        # auc_list.append(auc_mean)

    # print("Accuracy for the 10 iterations: ",  acc_list) #mean accuracy acros the 6 folds for each iteration
    # print("Recall for the 10 iterations: ",  recall_list) #mean accuracy acros the 6 folds for each iteration
    # print("Specificity for the 10 iterations: ",  specificity_list)
    # print("Precision for the 10 iterations: ",  precision_list)
    # print("F1  score for the 10 iterations: ",  f1_list)

    return (
        np.mean(acc_list),
        np.mean(specificity_list),
        np.mean(recall_list),
        np.mean(precision_list),
        np.mean(f1_list),
        # np.mean(auc_list),
    )
