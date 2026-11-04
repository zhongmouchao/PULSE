from sklearn.svm import OneClassSVM


def _normalize(test_scores):
    """Map decision scores onto [0, 1] using this test file's own range."""
    return (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())


def OCSVM(X_train, X_test):
    '''
    Input:
        X_train: training data
        X_test: one test matrix, or a list of test matrices that share this fit
    Output:
        y_pred: predicted labels, or a list of them when X_test is a list
    '''

    occ_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)  # Adjust hyperparameters as needed
    occ_svm.fit(X_train)  # Fit on the training data

    several = isinstance(X_test, (list, tuple))
    tests = list(X_test) if several else [X_test]
    predictions = [_normalize(occ_svm.decision_function(features)) for features in tests]
    if several:
        return predictions
    return predictions[0]
