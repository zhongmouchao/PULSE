import logging
from model.twoStep import twoStep
from model.classifier.classifier_models import LogisticRegression, SimpleMultiLayerPerceptron, WideDeepNetwork, WideDeepTransformer, TransformerClassifier, NAM
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import math
def run_twostep(observed_feature, testing_feature, test_activity_binary, test_freq,unobserved_feature,classifier_model, device,spy_seed, penalty_coeff):
    """Run 2Step method.
    (NOT CHECKED YET)
    Run 2Step method
    Args:
        observed_feature (torch.Tensor): Observed feature
        testing_feature (torch.Tensor): Testing feature
        test_activity_binary (list): Testing activity binary
        test_freq (list): Test frequency
        unobserved_feature (torch.Tensor): Unobserved feature
        classifier_model (str): Classifier model
        device (str): Device
        spy_seed (int): Spy seed
        penalty_coeff (float): Penalty coefficient
    """
    logging.info('Running 2Step')
    
    if classifier_model == 'LR':
        classifier = LogisticRegression(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = LogisticRegression(input_dim=observed_feature.shape[1], device=device)
    elif classifier_model == 'MLP':
        classifier = SimpleMultiLayerPerceptron(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = SimpleMultiLayerPerceptron(input_dim=observed_feature.shape[1], device=device)
    elif classifier_model == 'WD':
        classifier = WideDeepNetwork(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = WideDeepNetwork(input_dim=observed_feature.shape[1], device=device)
    elif classifier_model == 'WDT':
        classifier = WideDeepTransformer(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = WideDeepTransformer(input_dim=observed_feature.shape[1], device=device)
    elif classifier_model == 'Transformer':
        classifier = TransformerClassifier(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = TransformerClassifier(input_dim=observed_feature.shape[1], device=device)
    elif classifier_model == 'NAM':
        classifier = NAM(input_dim=observed_feature.shape[1], device=device)
        reliable_classifier = NAM(input_dim=observed_feature.shape[1], device=device)
    else:
        raise ValueError(f'Unknown classifier model: {classifier_model}')
    final_predictions = twoStep(
        X_observed=observed_feature,
        X_unobserved=unobserved_feature,
        X_test=testing_feature,
        final_classifier=classifier,
        reliable_classifier=reliable_classifier,
        spy_seed=spy_seed,
        device=device,
        penalty_coeff=penalty_coeff
    )
    if isinstance(testing_feature, (list, tuple)):
        return final_predictions, None, None, None

    roc_auc = ap = rho = None
    if test_activity_binary is not None:
        fpr_temp, tpr_temp, _ = roc_curve(test_activity_binary, final_predictions)
        roc_auc = auc(fpr_temp, tpr_temp)
        ap = average_precision_score(test_activity_binary, final_predictions)
    return final_predictions, roc_auc, ap, rho
 
