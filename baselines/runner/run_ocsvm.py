import logging
from model.OCSVM import OCSVM
def run_ocsvm(observed_feature, testing_feature):
    """Run OCSVM method.
    (Checked 04/10/2026)
    Run OCSVM method
    Args:
        observed_feature (torch.Tensor): Observed feature
        testing_feature (torch.Tensor): Testing feature
    Output:
        y_pred (numpy.ndarray): Predicted labels
    """
    logging.info('Running OCSVM')
    y_pred = OCSVM(X_train=observed_feature, X_test=testing_feature)
    return y_pred