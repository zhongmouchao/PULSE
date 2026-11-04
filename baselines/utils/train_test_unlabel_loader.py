import logging
import os

import pandas as pd

from utils.pulse_support import prepare_pulse_support
import numpy as np

from utils.gen_unlabeled import RAND_unlabeled, sample_near_observed


def _binary_activity(values):
    """Map test activity to 0 (negative) or 1 (positive)."""
    binary = []
    for value in values:
        if pd.isna(value):
            raise ValueError('Test activity contains a missing value')
        try:
            label = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Activity must be 0 or 1, got {value!r}') from exc
        if label not in (0, 1):
            raise ValueError(f'Activity must be 0 or 1, got {value!r}')
        binary.append(label)
    return binary


def _read_test_table(path):
    """Read one test CSV.

    `activity` (0/1) is used for AUC and average precision. `freq` is used
    for Spearman correlation. A file may contain either column or both.
    """
    test_df = pd.read_csv(path)
    if 'aa_sequence' not in test_df.columns:
        raise KeyError(f'{path} needs an aa_sequence column')
    has_activity = 'activity' in test_df.columns
    has_freq = 'freq' in test_df.columns
    if not has_activity and not has_freq:
        raise ValueError(f'{path} must contain an activity column, a freq column, or both')
    logging.info('Test file %s: activity=%s, freq=%s, rows=%d', path, has_activity, has_freq, len(test_df))
    return {
        'aa': list(test_df['aa_sequence']),
        'activity': _binary_activity(test_df['activity'].tolist()) if has_activity else None,
        'freq': test_df['freq'].tolist() if has_freq else None,
    }


def _test_files(property_name):
    """Return (assay name or None, path) for every test file of this property.

    `<property>_test.csv` is the single-assay layout. `<property>_test_<assay>.csv`
    is one assay of a shared training file, for example Alpha_test_binding.csv
    and Alpha_test_expression.csv. The two layouts are not mixed.
    """
    data_path = f'data/{property_name}'
    single = os.path.join(data_path, f'{property_name}_test.csv')
    prefix = f'{property_name}_test_'
    # Companion tables, not held-out assays.
    ignored_suffixes = {'raw', 'esm1v_results'}
    named = []
    if os.path.isdir(data_path):
        for filename in sorted(os.listdir(data_path)):
            if filename.startswith(prefix) and filename.endswith('.csv'):
                assay = filename[len(prefix):-len('.csv')]
                if assay and assay not in ignored_suffixes:
                    named.append((assay, os.path.join(data_path, filename)))
    if os.path.exists(single) and named:
        raise ValueError(
            f'{property_name} has {property_name}_test.csv and {property_name}_test_<assay>.csv. '
            'Use one layout.'
        )
    if os.path.exists(single):
        return [(None, single)]
    if named:
        return named
    raise FileNotFoundError(
        f'{data_path} needs {property_name}_test.csv or {property_name}_test_<assay>.csv'
    )


def load_train_test_data(property_name):
    """Load training data and every test file for that property.

    One training table can have several test assays. Each assay is scored
    after the same fit.

    Args:
        property_name (str): Property name
    Returns:
        data_path (str): Path to the training and testing data
        observed_nuc (list): List of observed nucleotide sequences (training data)
        observed_nuc_freq (list): List of observed nucleotide frequencies (training data)
        observed_aa (list): List of observed amino acid sequences (training data)
        test_sets (list[dict]): One dict per test file, with name, aa, activity, and freq.
            name is None when the property has a single `<property>_test.csv`.
    """

    data_path = f'data/{property_name}'
    
    # Load training data
    train_df = pd.read_csv(f'{data_path}/{property_name}_train.csv')
    test_sets = []
    for assay, path in _test_files(property_name):
        table = _read_test_table(path)
        table['name'] = assay
        test_sets.append(table)
    logging.info(
        'Loaded %d test file(s) for %s: %s',
        len(test_sets),
        property_name,
        ', '.join(test_set['name'] or 'test' for test_set in test_sets),
    )

    observed_nuc = train_df['nuc_sequence'].tolist()
    count_column = 'count' if 'count' in train_df.columns else 'Count'
    if count_column not in train_df.columns:
        raise KeyError(f'{data_path}/{property_name}_train.csv needs a count or Count column')
    observed_nuc_count = train_df[count_column].tolist()
    observed_aa = sorted(set(train_df['aa_sequence']))
    
    # Filter by sequence length based on property
    if property_name in ['Binding', 'Fusion']:
        observed_aa = [i for i in observed_aa if len(i) == 23]
    elif property_name == 'Evasion':
        observed_aa = [i for i in observed_aa if len(i) == 11]
    elif property_name == 'proteingym_A0':
        observed_aa = [i for i in observed_aa if len(i) == 87]
    elif property_name == 'proteingym_PSAE':
        observed_aa = [i for i in observed_aa if len(i) == 68]
    elif property_name == 'SARS':
        observed_aa = [i for i in observed_aa if len(i) == 48]
    elif property_name == 'SARS2':
        observed_aa = [i for i in observed_aa if len(i) == 48]
    elif property_name == 'H1':
        observed_aa = [i for i in observed_aa if len(i) == 565]
    elif property_name == 'H3':
        observed_aa = [i for i in observed_aa if len(i) == 566]
    elif property_name == 'ANTI_Wuhan':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name == 'Binding_Wuhan':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name == 'Binding_OmicronXBB15':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name == 'cell_omicron':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name == 'Express_Wuhan':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name == 'Express_Omicron':
        observed_aa = [i for i in observed_aa if len(i) == 201]
    elif property_name in ('Binding_KP311', 'ANTI_KP311', 'cell_KP311'):
        observed_aa = [i for i in observed_aa if len(i) == 1247]
    elif property_name in ('Alpha', 'Beta', 'Eta'):
        observed_aa = [i for i in observed_aa if len(i) == 201]
    
    return data_path, observed_nuc, observed_nuc_count, observed_aa, test_sets


def load_unlabeled_data(property_name, observed_aa, parent_years, mutation_rate=2e-4, obs_dist_max=2, num_sequences=None, near_observed=False):
    """Draw unlabeled peptides for ConstPU, 2Step, and kNN.

    The number of peptides matches the PULSE unlabeled set for the same parent
    years, unless num_sequences is set. Sequences are generated in memory from
    a fixed seed. H1 and H3 keep a fixed set of 500 because they have no
    year-split parent pool, unless num_sequences is set. Ten sets are returned,
    under seeds 100 through 109.

    near_observed draws each peptide from 1 to obs_dist_max mutations from an
    observed peptide and rejects peptides already in the observed set.
    Otherwise each peptide is a uniform draw over the 20 amino acids.
    """
    lengths = {len(peptide) for peptide in observed_aa}
    if len(lengths) != 1:
        raise ValueError(f'{property_name} training peptides do not share one length: {sorted(lengths)}')
    length_sequence = lengths.pop()
    if num_sequences is not None:
        num_sequences = int(num_sequences)
        if num_sequences <= 0:
            raise ValueError(f'num_sequences must be positive, got {num_sequences}')
    elif property_name in ('H1', 'H3'):
        num_sequences = 500
    else:
        support = prepare_pulse_support(
            property_name, mutation_rate, parent_years, obs_dist_max=obs_dist_max,
        )
        num_sequences = support['summary']['n_unlabeled_peptides']
        peptide_length = support['summary']['sequence_length'] // 3
        if peptide_length != length_sequence:
            raise ValueError(
                f'{property_name} peptide length is {length_sequence}, '
                f'but the parent pool encodes length {peptide_length}'
            )
    if near_observed:
        logging.info(
            'Generating 10 sets of %d peptides within %d mutations of an observed %s peptide (length %d)',
            num_sequences,
            int(obs_dist_max),
            property_name,
            length_sequence,
        )
    else:
        logging.info(
            'Generating 10 sets of %d random peptides of length %d for %s',
            num_sequences,
            length_sequence,
            property_name,
        )
    excluded = {peptide.upper() for peptide in observed_aa} if near_observed else None
    unlabeled_aa = {}
    for seed in range(10):
        if near_observed:
            unlabeled_aa[100 + seed] = sample_near_observed(
                observed_aa,
                num_sequences,
                np.random.default_rng(seed),
                max_mutations=int(obs_dist_max),
                excluded=excluded,
            )
        else:
            unlabeled_aa[100 + seed] = RAND_unlabeled(
                num_sequences=num_sequences,
                length_sequence=length_sequence,
                seed=seed,
            )
    return unlabeled_aa