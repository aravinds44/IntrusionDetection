from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features_chi2(X, y, k=10):
    """
    Select top k features using the Chi-Square test.
    """
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

def select_features_mi(X, y, k=10):
    """
    Select top k features using Mutual Information.
    """
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

def select_features_rf(X, y, k=10):
    """
    Select top k features using feature importance from a RandomForestClassifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = feature_importances.nlargest(k).index
    return X[selected_features], selected_features

def feature_selection_pipeline(X, y, method='chi2', k=10):
    """
    Apply the chosen feature selection method.
    """
    if method == 'chi2':
        return select_features_chi2(X, y, k)
    elif method == 'mutual_info':
        return select_features_mi(X, y, k)
    elif method == 'random_forest':
        return select_features_rf(X, y, k)
    else:
        raise ValueError("Invalid feature selection method. Choose from 'chi2', 'mutual_info', or 'random_forest'.")
