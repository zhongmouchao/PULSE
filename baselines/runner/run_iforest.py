import logging
from model.iForest import iForest
def run_iforest(observed_feature, testing_feature):
    """Run iForest method.
    (Checked 04/10/2026)
    Run iForest method
    Args:
        observed_feature (torch.Tensor): Observed feature
        testing_feature (torch.Tensor): Testing feature
    Output:
        y_pred (numpy.ndarray): Predicted labels
    """
    logging.info('Running iForest')
    y_pred = iForest(X_train=observed_feature, X_test=testing_feature)
    return y_pred