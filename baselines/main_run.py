#!/usr/bin/env python
"""
Run the RBD baselines: ConstPU, 2Step, OCSVM, and iForest.

ESM-1v is a separate script, esm1v_model.py. PULSE is not in this package.
"""

import argparse
import copy
import logging
import os
import pickle
import random
from types import NoneType

# Configure logging before importing local modules that may call basicConfig.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use('Agg')  # Use non-interactive backend for SLURM

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from Bio.Seq import Seq
from sklearn.metrics import auc, average_precision_score, roc_curve

from runner.run_constpu import run_constpu, run_constpu_sgd
from runner.run_iforest import run_iforest
from runner.run_ocsvm import run_ocsvm
from runner.run_twostep import run_twostep
from utils.pulse_support import canonical_parent_years, select_parent_years
from utils.gen_representation import featurize_peptides, load_cached_features, load_random_features
from utils.train_test_unlabel_loader import (
    load_train_test_data,
    load_unlabeled_data,
)


def score_predictions(y_pred, test_activity_binary, test_freq):
    """Score predictions from whichever test columns are present.

    activity -> ROC AUC and average precision
    freq -> Spearman correlation
    both columns -> all three metrics

    A constant prediction makes Spearman undefined. In that case each test
    sequence is assigned a fresh uniform draw from (0, 1) before scoring.
    Output:
        y_pred: scores used for the metrics, resampled when the input was constant.
        scores: dict with roc_auc, ap, and rho.
    """
    if test_activity_binary is None and test_freq is None:
        raise ValueError('Test data has neither an activity column nor a freq column')

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_pred.size > 0 and np.all(y_pred == y_pred[0]):
        logging.info(
            'all test sequences received same prediction -- resampling predicted values uniformly from (0,1)'
        )
        y_pred = np.random.uniform(0.0, 1.0, size=y_pred.shape)

    scores = {'roc_auc': None, 'ap': None, 'rho': None}
    if test_activity_binary is not None:
        fpr, tpr, _ = roc_curve(test_activity_binary, y_pred)
        scores['roc_auc'] = auc(fpr, tpr)
        scores['ap'] = average_precision_score(test_activity_binary, y_pred)
    if test_freq is not None:
        rho, _ = spearmanr(test_freq, y_pred)
        scores['rho'] = rho
    return y_pred, scores


def log_scores(scores):
    if scores['roc_auc'] is not None:
        logging.info(f'AUC: {scores["roc_auc"]:.4f}')
        logging.info(f'Average Precision: {scores["ap"]:.4f}')
    if scores['rho'] is not None:
        logging.info(f'Spearman rho: {scores["rho"]:.4f}')


def log_score_summary(score_list):
    labels = (
        ('roc_auc', 'AUC'),
        ('ap', 'Average Precision'),
        ('rho', 'Spearman rho'),
    )
    for key, label in labels:
        values = [scores[key] for scores in score_list if scores[key] is not None]
        if values:
            logging.info(f'{label}: {np.mean(values)} +/- {np.std(values)}')


def prediction_record(y_pred, scores, **extra):
    record = {'y_pred': y_pred, 'roc_auc': scores['roc_auc'], 'ap': scores['ap'], 'rho': scores['rho']}
    record.update(extra)
    return record


def mutation_label(obs_dist_max):
    """Readable mutation-distance label for result filenames."""
    if obs_dist_max == 1:
        return 'one_point_mutation'
    if obs_dist_max == 2:
        return 'two_point_mutation'
    if obs_dist_max == 3:
        return 'three_point_mutation'
    return f'{obs_dist_max}_point_mutation'


def constpu_uses_sgd(args):
    """True when ConstPU estimates the full unlabeled sum by sampling."""
    return args.method == 'ConstPU' and args.constpu_unlabeled == 'sgd'


def constpu_uses_uniform(args):
    """True when ConstPU draws unlabeled peptides uniformly from the 3-mutation neighborhood."""
    return args.method == 'ConstPU' and args.constpu_unlabeled == 'uniform'


def constpu_draws_during_training(args):
    """True when ConstPU builds its unlabeled batch inside the training loop."""
    return constpu_uses_sgd(args) or constpu_uses_uniform(args)


def _predict_fitted(model, test_sets, device):
    """Score every test file with one fitted torch model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for test_set in test_sets:
            scores = model(torch.tensor(test_set['feature'], dtype=torch.float32, device=device))
            predictions.append(scores.cpu().detach().numpy())
    return predictions


def _records_for_tests(predictions, test_sets, **extra):
    """Score each test file.

    One test file returns that record. Several files return a dict keyed by
    assay name, so binding and expression each keep their own Spearman rho.
    """
    if len(test_sets) == 1:
        predictions = [predictions]
    if len(predictions) != len(test_sets):
        raise ValueError(f'{len(predictions)} predictions for {len(test_sets)} test files')
    records = {}
    for test_set, y_pred in zip(test_sets, predictions):
        if test_set['name']:
            logging.info('Test assay: %s', test_set['name'])
        y_pred, scores = score_predictions(y_pred, test_set['activity'], test_set['freq'])
        log_scores(scores)
        record = prediction_record(y_pred, scores, **extra)
        if not test_set['name']:
            return record
        records[test_set['name']] = record
    return records


def _evaluate_twostep(observed_feature, random_features, test_sets, classifier_model, device, penalty_coeff, set_index=None):
    """Score 2Step on unlabeled sets and 10 spy seeds.

    Each spy seed is trained once. Every test file is scored from that fit.
    set_index selects one unlabeled set, 0 through 9. None runs all 10 sets.
    Output:
        result_dict: predictions keyed by unlabeled-set index, then spy seed.
            Several test files return a dict of those structures, keyed by assay.
        all_scores: one score dict for every unlabeled set and spy seed.
            Several test files return a dict of those lists, keyed by assay.
    """
    several = len(test_sets) > 1
    testing_feature = [test_set['feature'] for test_set in test_sets] if several else test_sets[0]['feature']
    assay_names = [test_set['name'] or 'test' for test_set in test_sets]
    result_dict = {name: {} for name in assay_names}
    all_scores = {name: [] for name in assay_names}
    set_indexes = range(10) if set_index is None else (set_index,)
    for i in set_indexes:
        for name in assay_names:
            result_dict[name][i] = {}
        seed = 100 + i
        logging.info(f'Running 2Step with RAND unlabeled data seed: {i+1}th')
        unobserved_feature = random_features[seed]
        for spy_seed in range(10):
            y_pred, _, _, _ = run_twostep(
                observed_feature=observed_feature, unobserved_feature=unobserved_feature, testing_feature=testing_feature,
                test_activity_binary=None if several else test_sets[0]['activity'],
                test_freq=None if several else test_sets[0]['freq'],
                classifier_model=classifier_model, device=device, spy_seed=spy_seed, penalty_coeff=penalty_coeff
            )
            predictions = y_pred if several else [y_pred]
            for test_set, prediction in zip(test_sets, predictions):
                name = test_set['name'] or 'test'
                if test_set['name']:
                    logging.info('Test assay: %s', test_set['name'])
                prediction, scores = score_predictions(prediction, test_set['activity'], test_set['freq'])
                result_dict[name][i][spy_seed] = prediction_record(
                    prediction, scores, spy_seed=spy_seed, random_set=seed,
                )
                all_scores[name].append(scores)
    if not several:
        return result_dict['test'], all_scores['test']
    return result_dict, all_scores


def result_stem(args):
    """Filename stem that states the settings which change the fit.

    Problem, method, representation, classifier, parent years, mutation
    distance, and penalty. Epsilon, total infections, and the mutation rate
    are fixed and are left out.
    """
    parts = [args.property, args.method, args.representation]
    if args.classifier_model and args.method not in ('OCSVM', 'iForest'):
        parts.append(args.classifier_model)
    if args.parent_years and args.method not in ('OCSVM', 'iForest'):
        parts.append('parents_' + '-'.join(args.parent_years))
        parts.append(mutation_label(args.obs_dist_max))
    if constpu_draws_during_training(args):
        if constpu_uses_uniform(args):
            parts.append('uniform_three_point_mutation')
            parts.append(f'unobs_{args.constpu_unobs_batch}')
        else:
            parts.append(f'sgd_unobs_{args.constpu_unobs_batch}')
        if args.constpu_q is not None:
            parts.append(f'q_{float(args.constpu_q):.6g}')
            parts.append('final' if args.constpu_final_only else 'cv')
    if args.method == '2Step' and args.rand_unlabeled_size:
        parts.append('unobs_' + '-'.join(str(size) for size in args.rand_unlabeled_size))
    if args.method == '2Step' and args.twostep_set_index is not None:
        parts.append(f'set_{args.twostep_set_index}')
    if args.method in ('ConstPU', '2Step'):
        penalty = float(args.penalty_coeff)
        penalty_text = str(int(penalty)) if penalty.is_integer() else str(penalty)
        parts.append(f'penalty_{penalty_text}')
    return '_'.join(parts)

def main(args, param_dict):
    logging.info('Starting main function')
    """Main function."""
    
    # Validate arguments. E_Gen is only for PULSE. The other unlabeled
    # methods are fixed to RAND below, so a mismatched flag is replaced.
    use_unlabeled = args.use_unlabel == 'True'
    
    if args.method in ['OCSVM', 'iForest']:
        if use_unlabeled:
            logging.info(f'Warning: use_unlabel and classifier model are ignored for {args.method} method!')
        use_unlabeled = False
        args.classifier_model = None
        args.unlabeled_type = None
    elif args.method == 'ConstPU' and args.constpu_unlabeled == 'sgd':
        if args.representation != 'CHEM':
            raise ValueError(
                'ConstPU SGD draws unobserved peptides during training and featurizes them with CHEM'
            )
        logging.info(
            'ConstPU SGD draws unobserved peptides within 2 mutations of an observed peptide. '
            'The fixed unlabeled set is not loaded.'
        )
        use_unlabeled = True
    elif constpu_uses_uniform(args):
        if args.representation != 'CHEM':
            raise ValueError(
                'ConstPU uniform draws unobserved peptides during training and featurizes them with CHEM'
            )
        logging.info(
            'ConstPU uniform draws negatives uniformly from the unobserved 3-mutation neighborhood. '
            'The unlabeled term is the mean of log(1 - q s(u)). The fixed unlabeled set is not loaded.'
        )
        use_unlabeled = True
    elif args.method in ['ConstPU', '2Step']:
        if not use_unlabeled or args.unlabeled_type != 'RAND':
            logging.info(f'{args.method} uses RAND unlabeled data. E_Gen is reserved for PULSE.')
        use_unlabeled = True
        args.unlabeled_type = 'RAND'
    
    # Set up device
    device = setup_device(args.device)
    logging.info(f'Using device: {device}')
    
    
    # Create output directories -- specific to property.
    # 2Step with --rand_unlabeled_size draws peptides within obs_dist_max
    # mutations of an observed sequence, so those results go in 2Step_near.
    if args.method == '2Step' and args.rand_unlabeled_size:
        method_dir = '2Step_near'
    elif args.method == 'ConstPU':
        method_dir = 'ConstPU_uniform'
    else:
        method_dir = args.method
    args.output_dir = f'{args.output_dir}/{args.property}/{method_dir}'
    os.makedirs(f'{args.output_dir}', exist_ok=True) # create result/property/method
    
    # Parent years are the shortest prefix of before_2000, 2000, 2001, ...
    # whose counts exceed parent_count_fraction of every training count.
    if args.rand_unlabeled_size and args.method != '2Step':
        raise ValueError('--rand_unlabeled_size is only used by 2Step')
    if args.parent_years:
        args.parent_years = canonical_parent_years(args.parent_years)
    elif args.method not in ('OCSVM', 'iForest'):
        train_frame = pd.read_csv(f'data/{args.property}/{args.property}_train.csv')
        args.parent_years = select_parent_years(train_frame, args.parent_count_fraction)

    # Load train and test data
    logging.info(f'Loading data for property: {args.property}')
    _, observed_nuc, observed_nuc_count, observed_aa, test_sets = load_train_test_data(property_name=args.property)
    explicit_rand_sizes = args.method == '2Step' and bool(args.rand_unlabeled_size)
    if args.method in ['OCSVM', 'iForest'] or constpu_draws_during_training(args) or explicit_rand_sizes:
        unlabeled_aa = None
    else:
        unlabeled_aa = load_unlabeled_data(
            property_name=args.property,
            observed_aa=observed_aa,
            parent_years=args.parent_years,
            mutation_rate=param_dict['nuc_mutation_rate'],
            obs_dist_max=args.obs_dist_max,
            near_observed=args.method == 'ConstPU',
        )
    
    # Observed and test features come from the cached archives. Random
    # unlabeled peptides are cached separately, per parent-year choice.
    logging.info(f'Loading cached {args.representation} features...')
    observed_feature = load_cached_features(args.property, args.representation, observed_aa, 'observed')
    for test_set in test_sets:
        test_set['feature'] = load_cached_features(
            args.property, args.representation, test_set['aa'], 'test',
        )
    several_tests = len(test_sets) > 1
    testing_feature = [test_set['feature'] for test_set in test_sets] if several_tests else test_sets[0]['feature']
    random_features = None
    if unlabeled_aa is not None:
        random_features = load_random_features(
            property_name=args.property,
            representation=args.representation,
            unlabeled_aa=unlabeled_aa,
            parent_years=args.parent_years,
            mutation_rate=param_dict['nuc_mutation_rate'],
            obs_dist_max=args.obs_dist_max,
        )
    result_dict = None
    if args.method == 'OCSVM':
        #checked 04/07/2026
        logging.info('Running OCSVM')
        y_pred = run_ocsvm(observed_feature=observed_feature, testing_feature=testing_feature) #checked 04/07/2026
        result_dict = _records_for_tests(y_pred, test_sets)
    elif args.method == 'iForest':
        #checked 04/07/2026
        logging.info('Running iForest')
        y_pred = run_iforest(observed_feature=observed_feature, testing_feature=testing_feature) #checked 04/07/2026
        result_dict = _records_for_tests(y_pred, test_sets)
    elif args.method == '2Step':
        runs = []
        if args.rand_unlabeled_size:
            for size in args.rand_unlabeled_size:
                logging.info('Preparing 10 unlabeled sets of %d peptides for 2Step', size)
                sized_unlabeled = load_unlabeled_data(
                    property_name=args.property,
                    observed_aa=observed_aa,
                    parent_years=args.parent_years,
                    mutation_rate=param_dict['nuc_mutation_rate'],
                    obs_dist_max=args.obs_dist_max,
                    num_sequences=size,
                    near_observed=True,
                )
                sized_features = load_random_features(
                    property_name=args.property,
                    representation=args.representation,
                    unlabeled_aa=sized_unlabeled,
                    parent_years=args.parent_years,
                    mutation_rate=param_dict['nuc_mutation_rate'],
                    obs_dist_max=args.obs_dist_max,
                    num_sequences=size,
                )
                runs.append((size, sized_features))
        else:
            runs.append((None, random_features))
        result_dict = {}
        for size, features in runs:
            per_size, all_scores = _evaluate_twostep(
                observed_feature, features, test_sets,
                args.classifier_model, device, args.penalty_coeff,
                set_index=args.twostep_set_index,
            )
            n_unobserved = features[100].shape[0]
            n_sets = 1 if args.twostep_set_index is not None else 10
            score_groups = all_scores if several_tests else {'test': all_scores}
            payloads = per_size if several_tests else {'test': per_size}
            for assay, scores in score_groups.items():
                logging.info('==============================================')
                assay_label = f', assay {assay}' if several_tests else ''
                logging.info(
                    'Average Result of 2Step with %d unobserved peptides (%d unlabeled sets x 10 spy seeds)%s:',
                    n_unobserved,
                    n_sets,
                    assay_label,
                )
                log_score_summary(scores)
            if size is None:
                result_dict = per_size
            elif several_tests:
                for assay, payload in payloads.items():
                    result_dict.setdefault(assay, {})[size] = payload
            else:
                result_dict[size] = per_size
    
    elif args.method == 'ConstPU' and args.constpu_unlabeled in ('sgd', 'uniform'):
        if constpu_uses_uniform(args):
            logging.info(
                'Running ConstPU with uniform draws from the unobserved 3-mutation neighborhood'
            )
        else:
            logging.info('Running ConstPU SGD with the observed mean plus the unlabeled mean')
        y_pred, best_const_pu_model, best_q, cv_auc = run_constpu_sgd(
            observed_feature=observed_feature,
            observed_peptides=observed_aa,
            testing_feature=test_sets[0]['feature'],
            classifier_model=args.classifier_model,
            device=device,
            penalty_coeff=args.penalty_coeff,
            unobs_batch=args.constpu_unobs_batch,
            q=args.constpu_q,
            final_only=args.constpu_final_only,
            uniform=constpu_uses_uniform(args),
        )
        logging.info('==============================================')
        logging.info('Result of ConstPU SGD')
        logging.info(f'Best q: {best_q:.6e}')
        if y_pred is None:
            logging.info(f'Average corrected AUC: {cv_auc:.6f}')
            result_dict = {'best_q': best_q, 'avg_corrected_auc': cv_auc, 'cv_only': True}
            best_model = None
        elif several_tests:
            result_dict = _records_for_tests(
                _predict_fitted(best_const_pu_model, test_sets, device),
                test_sets,
                best_q=best_q,
            )
            result_dict['best_q'] = best_q
            best_model = best_const_pu_model
        else:
            y_pred, scores = score_predictions(y_pred, test_sets[0]['activity'], test_sets[0]['freq'])
            log_scores(scores)
            result_dict = prediction_record(y_pred, scores, best_q=best_q)
            best_model = best_const_pu_model
        logging.info('==============================================')
    elif args.method == 'ConstPU':
        result_dict = {}
        best_model = {}
        all_scores = {test_set['name'] or 'test': [] for test_set in test_sets}
        for i in range(10):
            logging.info(f'Running ConstPU with RAND unlabeled data seed: {i+1}th')
            seed = 100 + i
            unobserved_feature = random_features[seed]
            y_pred,best_const_pu_model,best_q = run_constpu(
                observed_feature=observed_feature, unobserved_feature=unobserved_feature, testing_feature=testing_feature,
                classifier_model=args.classifier_model, device=device, penalty_coeff=args.penalty_coeff
            )
            logging.info('==============================================')
            logging.info(f'Result of ConstPU with RAND unlabeled data seed: {i+1}th')
            if several_tests:
                predictions = _predict_fitted(best_const_pu_model, test_sets, device)
            else:
                predictions = [y_pred]
            for test_set, prediction in zip(test_sets, predictions):
                name = test_set['name'] or 'test'
                if test_set['name']:
                    logging.info('Test assay: %s', test_set['name'])
                prediction, scores = score_predictions(prediction, test_set['activity'], test_set['freq'])
                log_scores(scores)
                record = prediction_record(prediction, scores, best_q=best_q, random_set=seed)
                all_scores[name].append(scores)
                if several_tests:
                    result_dict.setdefault(name, {})[i] = record
                else:
                    result_dict[i] = record
            logging.info('==============================================')
            best_model[i] = copy.deepcopy(best_const_pu_model)
        logging.info('==============================================')
        for name, scores in all_scores.items():
            assay = f' on {name}' if several_tests else ''
            logging.info(f'Average Result of ConstPU with RAND unlabeled data (10 times){assay}:')
            log_score_summary(scores)
        logging.info('==============================================')
    
    stem = result_stem(args)
    result_path = f'{args.output_dir}/{stem}_result.pth'
    if constpu_draws_during_training(args):
        if best_model is not None:
            model_path = f'{args.output_dir}/fitted_model_{stem}.pth'
            if constpu_draws_during_training(args):
                model_path = f'{args.output_dir}/fitted_model_{stem}_q_{result_dict["best_q"]:.6g}.pth'
            torch.save(best_model.state_dict(), model_path)
            logging.info(f'{args.method} model saved to: {model_path}')
        torch.save(result_dict, result_path)
    elif args.method == 'ConstPU':
        q_by_repeat = result_dict
        if result_dict and isinstance(next(iter(result_dict)), str):
            q_by_repeat = next(iter(result_dict.values()))
        for i in range(10):
            random_seed = 100 + i
            const_pu_model_path = (
                f'{args.output_dir}/fitted_model_{stem}'
                f'_q_{q_by_repeat[i]["best_q"]}_random_set_{random_seed}.pth'
            )
            torch.save(best_model[i].state_dict(), const_pu_model_path)
            logging.info(f'ConstPU model saved to: {const_pu_model_path}')
        torch.save(result_dict, result_path)
    else:
        torch.save(result_dict, result_path)
    logging.info(f'Result saved to: {result_path}')
    logging.info('Done!')



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--method', type=str, required=True,
        choices=['ConstPU', '2Step', 'OCSVM', 'iForest'],
        help='Model/method to use'
    )
    
    parser.add_argument(
        '--property', type=str, required=True,
        choices=['Binding_Wuhan', 'Express_Wuhan', 'Binding_OmicronXBB15', 'Express_Omicron', 'cell_omicron', 'Alpha', 'Beta', 'Eta'],
        help='Property/dataset to use'
    )
    
    parser.add_argument(
        '--use_unlabel', type=str, required=True,
        choices=['True', 'False'],
        help='Whether to use unlabeled data'
    )
    
    parser.add_argument(
        '--unlabeled_type', type=str, default=None,
        choices=['RAND', 'E_Gen', 'None'],
        help='Unlabeled source. 2Step and ConstPU with --constpu_unlabeled fixed use RAND. '
             'ConstPU with --constpu_unlabeled sgd or uniform draws peptides during training.'
    )
    
    parser.add_argument(
        '--classifier_model', type=str, default=None,
        choices=['LR', 'MLP', 'WD', 'kNN', 'WDT', 'Transformer', 'NAM', 'None'],
        help='Classifier model type. NAM is a per-site network and requires CHEM features.'
    )
    
    parser.add_argument(
        '--representation', type=str, required=True,
        choices=['CHEM', 'ESM2'],
        help='Representation type'
    )
    
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda, mps, cpu). Auto-detected if not specified'
    )
    
    parser.add_argument(
        '--obs_dist_max', type=int, default=2,
        help='Maximum nucleotide mutations from the parent pool. '
             'Observations farther than this are dropped, and generated neighbors stop at this distance.'
    )

    parser.add_argument(
        '--output_dir', type=str, default='result',
        help='Output directory for results'
    )

    parser.add_argument(
        '--epsilon', type=float, default=10,
        help='Epsilon parameter'
    )

    parser.add_argument(
        '--total_infections', type=float, default=2.4e10,
        help='Total infections parameter'
    )

    parser.add_argument(
        '--nuc_mutation_rate', type=float, default=2e-4,
        help='Nucleotide mutation rate'
    )

    parser.add_argument(
        '--parent_count_fraction', type=float, default=0.01,
        help='Parent pool is the shortest prefix of before_2000, 2000, 2001, ... '
             'whose counts exceed this fraction of the training counts. '
             'Ignored when --parent_years is set. The reported runs use the default.'
    )

    parser.add_argument(
        '--parent_years', nargs='+', default=None,
        help='Optional explicit parent-year columns. When omitted, years are chosen '
             'with --parent_count_fraction.'
    )

    parser.add_argument(
        '--penalty_coeff', type=float, default=50,
        help='Penalty coefficient'
    )

    parser.add_argument(
        '--constpu_unlabeled', type=str, default='fixed',
        choices=['fixed', 'sgd', 'uniform'],
        help='ConstPU unlabeled term. fixed draws a saved set of peptides within '
             '2 mutations of an observed peptide. sgd draws a new batch of those '
             'peptides each step and weights each by 1 / |N(parent)|. uniform draws '
             'a batch uniformly from the unobserved 3-mutation neighborhood; the '
             'unlabeled term is the mean of log(1 - q s(u)).'
    )

    parser.add_argument(
        '--constpu_unobs_batch', type=int, default=128,
        help='Unobserved peptides drawn per step when --constpu_unlabeled is sgd or uniform.'
    )

    parser.add_argument(
        '--constpu_q', type=float, default=None,
        help='ConstPU sgd or uniform only. Run this q instead of 0.1 through 1.0. '
             'The job reports that q\'s mean corrected AUC and does not score the test set. '
             'Add --constpu_final_only to fit the test model at this q.'
    )

    parser.add_argument(
        '--constpu_final_only', action='store_true',
        help='ConstPU sgd or uniform only. Requires --constpu_q. Skip cross-validation, '
             'fit on all observed peptides, and compute the test Spearman rho.'
    )

    parser.add_argument(
        '--rand_unlabeled_size', type=int, nargs='+', default=None,
        help='2Step only. Each value is the number of unobserved peptides in a set. '
             'Each peptide is drawn within --obs_dist_max nucleotide substitutions of an observed peptide. '
             'The reported runs use --obs_dist_max 3 and --rand_unlabeled_size 1000. '
             'Ten sets are drawn for each value, and each set is fit with 10 spy seeds.'
    )

    parser.add_argument(
        '--twostep_set_index', type=int, default=None,
        help='2Step only. Run unlabeled set 0-9 (random seed 100 plus this index) '
             'across all 10 spy seeds. When omitted, all 10 unlabeled sets are run.'
    )
    
    args = parser.parse_args()

    # Optional validation
    if args.use_unlabel == 'True' and args.unlabeled_type is None:
        parser.error('--unlabeled_type is required when --use_unlabel=True')
    if args.twostep_set_index is not None:
        if args.method != '2Step':
            parser.error('--twostep_set_index is only used by 2Step')
        if not 0 <= args.twostep_set_index <= 9:
            parser.error('--twostep_set_index must be an integer from 0 to 9')
    if args.constpu_q is not None or args.constpu_final_only:
        if args.method != 'ConstPU' or args.constpu_unlabeled not in ('sgd', 'uniform'):
            parser.error('--constpu_q and --constpu_final_only are only used by ConstPU sgd or uniform')
        if args.constpu_final_only and args.constpu_q is None:
            parser.error('--constpu_final_only requires --constpu_q')
        if args.constpu_q is not None and not 0 < args.constpu_q <= 1:
            parser.error('--constpu_q must be in (0, 1]')

    return args


def setup_device(device_arg):
    """Set up and return the device."""
    if device_arg:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logging.info('Setting up random seed and deterministic algorithms')

    args = parse_args()

    param_dict = {
        'epsilon': args.epsilon,
        'total_infections': args.total_infections,
        'nuc_mutation_rate': args.nuc_mutation_rate
    }

    main(args, param_dict)