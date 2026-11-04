import logging
import math

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from utils.neighborhood_sampler import UniformNeighborhoodSampler
from utils.gen_representation import chem
from utils.gen_unlabeled import sample_near_observed, unobserved_neighborhood_sizes

from model.classifier.classifier_models import (
    LogisticRegression,
    SimpleMultiLayerPerceptron,
    NAM,
    TransformerClassifier,
    WideDeepNetwork,
    WideDeepTransformer,
)
from model.const_pu import ConstPU


def _make_classifier(classifier_model, input_dim, device):
    if classifier_model == 'LR':
        return LogisticRegression(input_dim=input_dim, device=device)
    if classifier_model == 'MLP':
        return SimpleMultiLayerPerceptron(input_dim=input_dim, device=device)
    if classifier_model == 'WD':
        return WideDeepNetwork(input_dim=input_dim, device=device)
    if classifier_model == 'WDT':
        return WideDeepTransformer(input_dim=input_dim, device=device)
    if classifier_model == 'Transformer':
        return TransformerClassifier(input_dim=input_dim, device=device)
    if classifier_model == 'NAM':
        return NAM(input_dim=input_dim, device=device)
    raise ValueError(f'Unknown classifier model: {classifier_model}')


def _split_sets(features, labels, device):
    """Positive and unlabeled rows, in the same order boolean indexing would produce."""
    positive = torch.tensor(features[labels == 1], dtype=torch.float32, device=device)
    unlabeled = torch.tensor(features[labels == 0], dtype=torch.float32, device=device)
    return positive, unlabeled


def _classifier_penalty(classifier):
    penalty = None
    for parameter in classifier.parameters():
        term = torch.sum(torch.square(parameter))
        penalty = term if penalty is None else penalty + term
    return penalty


def _clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _fit_constpu(
    x_positive,
    x_unlabeled,
    classifier_model,
    input_dim,
    device,
    likelihood_q,
    penalty_coeff,
    base_lr,
    max_lr,
    step_size_up,
    max_epochs,
    patience,
    min_delta,
    max_grad_norm,
    log_prefix,
):
    """Train one ConstPU model and restore the weights from the lowest loss."""
    classifier = _make_classifier(classifier_model, input_dim, device)
    model = ConstPU(classifier_model=classifier, q=likelihood_q, device=device, exact_opt=True)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        mode='triangular',
        cycle_momentum=False,
    )
    model.train()
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_likelihood(x_positive, x_unlabeled)
        loss = loss + penalty_coeff * _classifier_penalty(model.classifier)
        if not torch.isfinite(loss):
            logging.info(f'Non-finite loss at epoch {epoch} during {log_prefix}.')
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_state = _clone_state(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            logging.info(
                f'{log_prefix}, Epoch {epoch}, Loss: {current_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}'
            )
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch} during {log_prefix}')
            break

    if best_state is None:
        return None
    model.load_state_dict(best_state)
    return model


def run_constpu(observed_feature, unobserved_feature, testing_feature, classifier_model, device, penalty_coeff):
    """
    Run ConstPU and select q by 10-fold cross validation using corrected AUC.

    Returns:
        y_pred: prediction on testing_feature
        best_q: q with the highest average corrected AUC
        best_model: ConstPU model retrained on the full data using best_q
    """
    logging.info('Running ConstPU with 10-fold CV for q selection')

    # =========================
    # Prepare full dataset
    # =========================
    X_test = torch.tensor(testing_feature, dtype=torch.float32, device=device)

    X_all_np = np.concatenate([observed_feature, unobserved_feature], axis=0)
    y_all_np = np.concatenate([
        np.ones(len(observed_feature), dtype=np.float32),
        np.zeros(len(unobserved_feature), dtype=np.float32)
    ], axis=0)

    n_obs = len(observed_feature)
    n_unobs = len(unobserved_feature)

    # =========================
    # Candidate q list
    # =========================
    q_lower = min(2 * (n_obs / (n_obs + n_unobs)), 0.5)
    q_upper = 1.0
    h = 0.1
    q_lower_temp = math.ceil(q_lower / h) * h

    if abs(q_lower_temp - q_lower) < 1e-8:
        q_list = list(np.arange(q_lower, q_upper + h, h))
    else:
        q_list = [q_lower] + list(np.arange(q_lower_temp, q_upper + h, h))
    if q_list[-1] > 1.0:
        q_list = q_list[:-1]
    if q_list[0] < 2 * (n_obs / (n_obs + n_unobs)):
        q_list = q_list[1:]

    logging.info(f'q candidates: {q_list}')

    # =========================
    # Shared optimization setup
    # =========================
    base_lr = 1e-3
    max_lr = 1e-1
    step_size_up = 50
    max_epochs = 2000
    patience = 100
    min_delta = 1e-6
    max_grad_norm = 1.0

    # The fold split does not depend on q, so the feature rows are moved once.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    input_dim = observed_feature.shape[1]
    fit_kwargs = dict(
        classifier_model=classifier_model,
        input_dim=input_dim,
        device=device,
        penalty_coeff=penalty_coeff,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        max_grad_norm=max_grad_norm,
    )
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all_np, y_all_np), start=1):
        x_positive, x_unlabeled = _split_sets(X_all_np[train_idx], y_all_np[train_idx], device)
        x_val = torch.tensor(X_all_np[val_idx], dtype=torch.float32, device=device)
        folds.append((fold_idx, x_positive, x_unlabeled, x_val, y_all_np[val_idx]))

    best_q = None
    best_avg_corrected_auc = -float('inf')

    for q in q_list:
        logging.info('==============================================================')
        logging.info(f'Evaluating q = {q:.6f} by 10-fold CV')

        fold_corrected_auc_list = []
        pi = n_obs / (q * (n_obs + n_unobs))
        logging.info(f'pi for q={q:.6f}: {pi:.6f}')

        for fold_idx, x_positive, x_unlabeled, x_val, y_val_np in folds:
            logging.info(f'----- Fold {fold_idx}/10 for q={q:.6f} -----')
            fitted = _fit_constpu(
                x_positive,
                x_unlabeled,
                likelihood_q=pi,
                log_prefix=f'Fold {fold_idx}, q={q:.6f}',
                **fit_kwargs,
            )
            if fitted is None:
                logging.info(f'Fold {fold_idx}, q={q:.6f}: training failed, skip this fold')
                continue

            fitted.eval()
            with torch.no_grad():
                y_val_pred = fitted(x_val).cpu().detach().numpy().reshape(-1)
            raw_auc = roc_auc_score(y_val_np, y_val_pred)
            corrected_auc = (raw_auc - q / 2.0) / (1.0 - q)
            fold_corrected_auc_list.append(corrected_auc)
            logging.info(
                f'Fold {fold_idx}, q={q:.6f}, raw_auc={raw_auc:.6f}, corrected_auc={corrected_auc:.6f}'
            )

        if len(fold_corrected_auc_list) == 0:
            avg_corrected_auc = -float('inf')
        else:
            avg_corrected_auc = float(np.mean(fold_corrected_auc_list))

        logging.info(f'q={q:.6f}, average corrected AUC over folds = {avg_corrected_auc:.6f}')
        if avg_corrected_auc > best_avg_corrected_auc:
            best_avg_corrected_auc = avg_corrected_auc
            best_q = q

    if best_q is None:
        raise RuntimeError("Cross-validation failed: no valid q was found.")

    logging.info('==============================================================')
    logging.info(f'Best q selected by CV: {best_q:.6f}')
    logging.info(f'Best average corrected AUC: {best_avg_corrected_auc:.6f}')

    x_positive, x_unlabeled = _split_sets(X_all_np, y_all_np, device)
    best_model = _fit_constpu(
        x_positive,
        x_unlabeled,
        likelihood_q=best_q,
        log_prefix='final training',
        **fit_kwargs,
    )
    if best_model is None:
        raise RuntimeError("Final training failed: no valid ConstPU model was found.")

    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test).cpu().detach().numpy()

    return y_pred, best_model, best_q


def _sample_unobserved(observed_peptides, batch_size, excluded, rng, neighborhood_sizes):
    """Draw peptides one or two mutations from an observed peptide.

    Results that are already observed are rejected. Each draw is paired with
    the unobserved-neighborhood size of the parent it was mutated from.

    Inputs:
        observed_peptides (list[str]): mutation parents.
        batch_size (int): number of accepted peptides.
        excluded (set[str]): peptides left out of the unlabeled sum.
        rng (numpy Generator): random source.
        neighborhood_sizes (ndarray): |N(parent)| for each observed peptide.
    Output:
        peptides (list[str]): batch_size unobserved peptides.
        sizes (ndarray int64): neighborhood size of each peptide's parent.
    """
    peptides, parents = sample_near_observed(
        observed_peptides,
        batch_size,
        rng,
        max_mutations=2,
        excluded=excluded,
        return_parents=True,
    )
    return peptides, neighborhood_sizes[parents]


def _chem_matrix(peptides):
    """Chemical feature rows for peptides drawn during an SGD step.

    Inputs:
        peptides (list[str]): one common length.
    Output:
        ndarray float32: (len(peptides), 3L).
    """
    rows = [np.asarray(chem(peptide), dtype=np.float32).reshape(-1) for peptide in peptides]
    return np.stack(rows, axis=0)


def _fit_constpu_sgd(
    x_positive,
    universe_peptides,
    classifier_model,
    input_dim,
    device,
    likelihood_q,
    penalty_coeff,
    base_lr,
    max_lr,
    step_size_up,
    max_epochs,
    patience,
    min_delta,
    max_grad_norm,
        unobs_batch,
        rng,
        neighborhood_sizes,
        log_prefix,
        sampler=None,
):
    """Train ConstPU on the sum of the observed mean and the unlabeled mean.

    The observed mean divides log(q s(x)) by the number of peptides in the
    labeled sum. Without `sampler`, each step draws peptides one or two mutations
    from an observed peptide and the unlabeled term is the weighted mean of
    log(1 - q s(u)), with weight 1 / |N(parent)|. With `sampler`, the batch is
    uniform on the unobserved 3-mutation neighborhood and the unlabeled term is
    the plain mean of log(1 - q s(u)). Held-out observed peptides stay out of
    the draw. The saved weights are those with the lowest sampled loss.

    Inputs:
        x_positive (ndarray): (n_labeled, feature_dim) observed features in the labeled sum.
        universe_peptides (list[str]): every observed peptide. These are excluded from the draw.
        unobs_batch (int): unlabeled peptides per step.
        rng (numpy Generator): random source for the unlabeled draws.
        Other arguments match _fit_constpu.
    Output:
        ConstPU or None: fitted model, or None when every loss was non-finite.
    """
    classifier = _make_classifier(classifier_model, input_dim, device)
    model = ConstPU(classifier_model=classifier, q=likelihood_q, device=device, exact_opt=True)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        mode='triangular',
        cycle_momentum=False,
    )
    positive = torch.tensor(x_positive, dtype=torch.float32, device=device)
    known = {peptide.upper() for peptide in universe_peptides}
    model.train()
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        attempts = None
        if sampler is None:
            peptides, parent_sizes = _sample_unobserved(
                universe_peptides, unobs_batch, known, rng, neighborhood_sizes,
            )
        else:
            peptides, attempts = sampler.sample(unobs_batch)
            parent_sizes = None
        unlabeled_np = _chem_matrix(peptides)
        if unlabeled_np.shape[1] != positive.shape[1]:
            raise ValueError(
                f'CHEM features have width {unlabeled_np.shape[1]}, '
                f'but the observed features have width {positive.shape[1]}'
            )
        unlabeled = torch.tensor(unlabeled_np, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        if sampler is None:
            log_likelihood, labeled_mean, unlabeled_mean = model.sgd_log_likelihood(
                positive, unlabeled, parent_sizes,
            )
        else:
            log_likelihood, labeled_mean, unlabeled_mean = model.uniform_sgd_log_likelihood(
                positive, unlabeled,
            )
        loss = -log_likelihood + penalty_coeff * _classifier_penalty(model.classifier)
        if not torch.isfinite(loss):
            logging.info(
                '%s non-finite loss at epoch %d. Labeled mean %.6e, unlabeled mean %.6e.',
                log_prefix,
                epoch,
                float(labeled_mean),
                float(unlabeled_mean),
            )
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        current_loss = float(loss.item())
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_state = _clone_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 50 == 0:
            proposal_note = ''
            if attempts is not None:
                proposal_note = f', proposals per accepted peptide: {attempts / unobs_batch:.1f}'
            logging.info(
                '%s, Epoch %d, Loss: %.6f, labeled mean: %.6f, unlabeled mean: %.6f%s, LR: %.6f',
                log_prefix,
                epoch,
                current_loss,
                float(labeled_mean),
                float(unlabeled_mean),
                proposal_note,
                scheduler.get_last_lr()[0],
            )
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch} during {log_prefix}')
            break

    if best_state is None:
        return None
    model.load_state_dict(best_state)
    return model


def run_constpu_sgd(
    observed_feature,
    observed_peptides,
    testing_feature,
    classifier_model,
    device,
    penalty_coeff,
    unobs_batch=128,
    seed=0,
    q=None,
    final_only=False,
    uniform=False,
):
    """ConstPU with on-the-fly unlabeled samples near the observed peptides.

    The objective is the mean observed log term plus the mean unobserved log
    term. q is chosen from 0.1, 0.2, ..., 1.0 by corrected AUC. Fold training
    holds out observed peptides and scores them against one fixed unlabeled sample.
    Passing q runs that value only. final_only skips the folds and fits the test model.
    uniform draws the unlabeled batch uniformly from the unobserved 3-mutation
    neighborhood. Otherwise the batch is one or two mutations from a parent and
    each draw is weighted by 1 / |N(parent)|.

    Inputs:
        observed_feature (ndarray): (n_observed, feature_dim).
        observed_peptides (list[str]): same peptides, one common length.
        testing_feature (ndarray): rows scored after training.
        classifier_model (str): LR, MLP, WD, WDT, Transformer, or NAM.
        device (torch.device): training device.
        penalty_coeff (float): weight on the squared classifier weights.
        unobs_batch (int): unlabeled peptides drawn per step and for each fold's validation sample.
        seed (int): random source. Fold splits use random_state 0.
        q (float or None): one q in (0, 1]. None runs 0.1 through 1.0 and then the final fit.
        final_only (bool): fit every observed peptide at q and score the test set. Requires q.
        uniform (bool): sample unlabeled peptides uniformly from the 3-mutation neighborhood.
    Output:
        y_pred (ndarray or None): test scores. None when only the cross-validation AUC is computed.
        best_model (ConstPU or None): final model. None for a cross-validation-only q.
        best_q (float): q used for the final fit, or the q whose AUC was computed.
        cv_auc (float or None): mean corrected AUC. None for a final-only fit.
    """
    if final_only and q is None:
        raise ValueError('final_only requires q')
    if q is not None and not 0 < float(q) <= 1:
        raise ValueError(f'q must be in (0, 1], got {q}')
    if len(observed_peptides) < 10:
        raise ValueError(f'ConstPU SGD needs at least 10 observed peptides, got {len(observed_peptides)}')
    peptide_length = len(observed_peptides[0])
    if any(len(peptide) != peptide_length for peptide in observed_peptides):
        raise ValueError('observed peptides do not share one length')
    if uniform:
        logging.info(
            'ConstPU uniform averages log(q s(x)) over %d observed peptides and '
            'log(1 - q s(u)) over %d peptides drawn uniformly from the unobserved '
            '3-mutation neighborhood',
            len(observed_peptides),
            unobs_batch,
        )
    else:
        logging.info(
            'ConstPU SGD averages %d observed log terms and a batch of %d unobserved log terms',
            len(observed_peptides),
            unobs_batch,
        )
    q_list = [round(0.1 * step, 1) for step in range(1, 11)] if q is None else [float(q)]
    logging.info(f'q candidates: {q_list}')

    base_lr = 1e-3
    max_lr = 1e-1
    step_size_up = 50
    max_epochs = 2000
    patience = 100
    min_delta = 1e-6
    max_grad_norm = 1.0
    input_dim = observed_feature.shape[1]
    rng = np.random.default_rng(seed)
    known = {peptide.upper() for peptide in observed_peptides}
    sampler = None
    neighborhood_sizes = None
    if uniform:
        sampler = UniformNeighborhoodSampler(observed_peptides, radius=3, rng=rng)
        logging.info(
            'Raw 3-mutation neighborhood sizes range from %d to %d',
            int(sampler.neighborhood_size.min()),
            int(sampler.neighborhood_size.max()),
        )
    else:
        neighborhood_sizes = unobserved_neighborhood_sizes(observed_peptides, max_mutations=2)
        logging.info(
            'Unobserved neighborhood sizes range from %d to %d',
            int(neighborhood_sizes.min()),
            int(neighborhood_sizes.max()),
        )
    folds = []
    if not final_only:
        # Same seed and split for every q job, so the ten AUCs are comparable.
        splitter = KFold(n_splits=10, shuffle=True, random_state=0)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(observed_feature), start=1):
            if sampler is None:
                val_peptides, _ = _sample_unobserved(
                    observed_peptides, unobs_batch, known, rng, neighborhood_sizes,
                )
            else:
                val_peptides, _ = sampler.sample(unobs_batch)
            x_val_unlabeled = _chem_matrix(val_peptides)
            x_val = np.concatenate([observed_feature[val_idx], x_val_unlabeled], axis=0)
            y_val = np.concatenate([
                np.ones(len(val_idx), dtype=np.float32),
                np.zeros(len(val_peptides), dtype=np.float32),
            ])
            folds.append((fold_idx, observed_feature[train_idx], x_val, y_val))

    fit_kwargs = dict(
        classifier_model=classifier_model,
        input_dim=input_dim,
        device=device,
        penalty_coeff=penalty_coeff,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        max_grad_norm=max_grad_norm,
        unobs_batch=unobs_batch,
        rng=rng,
        neighborhood_sizes=neighborhood_sizes,
        sampler=sampler,
    )
    best_q = None
    best_avg_corrected_auc = -float('inf')
    if not final_only:
        for q_value in q_list:
            logging.info('==============================================================')
            logging.info(f'Evaluating q = {q_value:.6e} by 10-fold CV')
            fold_corrected_auc_list = []
            for fold_idx, x_train, x_val, y_val in folds:
                logging.info(f'----- Fold {fold_idx}/10 for q={q_value:.6e} -----')
                fitted = _fit_constpu_sgd(
                    x_train,
                    observed_peptides,
                    likelihood_q=float(q_value),
                    log_prefix=f'Fold {fold_idx}, q={q_value:.6e}',
                    **fit_kwargs,
                )
                if fitted is None:
                    logging.info(f'Fold {fold_idx}, q={q_value:.6e}: training failed, skip this fold')
                    continue
                fitted.eval()
                with torch.no_grad():
                    scores = fitted(torch.tensor(x_val, dtype=torch.float32, device=device))
                    y_val_pred = scores.cpu().detach().numpy().reshape(-1)
                raw_auc = roc_auc_score(y_val, y_val_pred)
                corrected_auc = (raw_auc - q_value / 2.0) / (1.0 - q_value) if q_value < 1.0 else raw_auc
                fold_corrected_auc_list.append(corrected_auc)
                logging.info(
                    f'Fold {fold_idx}, q={q_value:.6e}, raw_auc={raw_auc:.6f}, corrected_auc={corrected_auc:.6f}'
                )
            if len(fold_corrected_auc_list) == 0:
                raise RuntimeError(f'Cross-validation failed for q={q_value}: every fold failed')
            avg_corrected_auc = float(np.mean(fold_corrected_auc_list))
            logging.info(f'q={q_value:.6e}, average corrected AUC over folds = {avg_corrected_auc:.6f}')
            if q is not None:
                logging.info('CV only. The final model is not fit until this q is selected.')
                return None, None, float(q_value), avg_corrected_auc
            if avg_corrected_auc > best_avg_corrected_auc:
                best_avg_corrected_auc = avg_corrected_auc
                best_q = q_value
        if best_q is None:
            raise RuntimeError('Cross-validation failed: no valid q was found.')
        logging.info('==============================================================')
        logging.info(f'Best q selected by CV: {best_q:.6e}')
        logging.info(f'Best average corrected AUC: {best_avg_corrected_auc:.6f}')
    else:
        best_q = float(q)
        logging.info(f'Final fit only at q = {best_q:.6e}. Cross-validation is skipped.')
    best_model = _fit_constpu_sgd(
        observed_feature,
        observed_peptides,
        likelihood_q=float(best_q),
        log_prefix='final training',
        **fit_kwargs,
    )
    if best_model is None:
        raise RuntimeError('Final training failed: no valid ConstPU SGD model was found.')
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(torch.tensor(testing_feature, dtype=torch.float32, device=device))
        y_pred = y_pred.cpu().detach().numpy()
    return y_pred, best_model, best_q, None if final_only else best_avg_corrected_auc