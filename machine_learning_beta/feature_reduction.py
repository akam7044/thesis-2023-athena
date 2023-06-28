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


def feature_reduction_pca(X, variance: float):
    # Normalise:
    x_val = X.values
    x_val = StandardScaler().fit_transform(x_val)

    feat_cols = ["feature" + str(i) for i in range(x_val.shape[1])]

    pca = PCA(n_components=variance)

    principal_components = pca.fit_transform(x_val)

    n_components = principal_components.shape[1]
    print(f"For {variance} variance  ->  {n_components} components were computed")

    col_names = []
    for i in range(n_components):
        col_names.append("PC" + str(i + 1))

    principal_components_df = pd.DataFrame(data=principal_components, columns=col_names)

    return principal_components_df


def feature_reduction_lda(X_train, y_train):
    """
    LDA is supervised so we need a test and train split
    """

    # Normalise:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    # LDA
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)

    return X_train


def feature_reduction_mrmr(X, Y, n_components):
    selected_components = mrmr_classif(X=X, y=Y, K=n_components)
    x_train = pd.DataFrame(X).loc[:, selected_components].values
    return x_train
