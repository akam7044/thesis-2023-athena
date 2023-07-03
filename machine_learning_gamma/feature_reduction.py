import pandas as pd
from mrmr import mrmr_classif


def feature_reduction_mrmr(x_train, x_test, y_train, n_components):
    selected_components = mrmr_classif(X=x_train, y=y_train, K=n_components)
    x_train = pd.DataFrame(x_train).loc[:, selected_components]
    x_test = pd.DataFrame(x_test).loc[:, selected_components]
    return x_train, x_test
