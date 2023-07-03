from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    StandardScaler,
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from mrmr import mrmr_classif
from sklearn.preprocessing import MinMaxScaler


def feature_reduction_pca(x_train, x_test, variance: float):
    pca = PCA(n_components=variance)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test


def feature_reduction_lda(x_train, x_test, y_train):
    """
    LDA is supervised so we need a test and train split
    """

    # LDA
    lda = LDA(n_components=1)
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)

    return x_train, x_test


def feature_reduction_mrmr(x_train, x_test, y_train, n_components):
    selected_components = mrmr_classif(X=x_train, y=y_train, K=n_components)
    x_train = pd.DataFrame(x_train).loc[:, selected_components]
    x_test = pd.DataFrame(x_test).loc[:, selected_components]
    return x_train, x_test
