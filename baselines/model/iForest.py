from sklearn.ensemble import IsolationForest


def _normalize(test_scores):
    """Map anomaly scores onto [0, 1] using this test file's own range."""
    return (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())


def iForest(X_train, X_test):
    '''
    Input:
        X_train: training data
        X_test: one test matrix, or a list of test matrices that share this fit
    Output:
        y_pred: predicted labels, or a list of them when X_test is a list
    '''
    iso_forest = IsolationForest(
        n_estimators=1000,  # Number of trees in the forest
        contamination='auto',  # Automatically estimate the fraction of outliers
        random_state=42  # For reproducibility
    )
    iso_forest.fit(X_train)  # Fit on the training data

    several = isinstance(X_test, (list, tuple))
    tests = list(X_test) if several else [X_test]
    predictions = [_normalize(iso_forest.decision_function(features)) for features in tests]
    if several:
        return predictions
    return predictions[0]