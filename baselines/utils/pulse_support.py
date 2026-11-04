"""Parent / observation split and emergence support for PULSE.

The caller chooses which year labels form the parent pool. Every other year
column is an observation. Support S(y) is computed from the parents only, and
the peptide likelihood uses the observation sequences as positives.
"""

import fcntl
import logging
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.Seq import Seq

TRANSITION = {'A': ['G'], 'C': ['T'], 'G': ['A'], 'T': ['C']}
TRANSVERSION = {'A': ['C', 'T'], 'C': ['A', 'G'], 'G': ['C', 'T'], 'T': ['A', 'G']}
TRANSITION_FACTOR = 0.9
TRANSVERSION_FACTOR = 0.05
EPS_EXPSUM = 1e-12
OBS_DIST_MAX = 2
INVALID_AA = {'*', 'X', 'J', 'B', 'Z'}
_BASE2INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _count_series(frame):
    if 'count' in frame.columns:
        return frame['count'].astype(float)
    if 'Count' in frame.columns:
        return frame['Count'].astype(float)
    raise KeyError('Training table needs a count or Count column')


def _time_columns(frame):
    """Year columns, then month columns, in chronological order.

    A year column is a four-digit name such as 2000. A month column is
    YYYY-MM, such as 2019-01.
    """
    years = []
    months = []
    for column in frame.columns:
        text = str(column)
        if re.fullmatch(r'\d{4}', text):
            years.append(text)
        elif re.fullmatch(r'\d{4}-\d{2}', text) and 1 <= int(text[5:7]) <= 12:
            months.append(text)
    years.sort(key=int)
    months.sort()
    return years + months


def _year_columns(frame):
    return _time_columns(frame)


def canonical_parent_years(values):
    """Normalize parent-year tokens.

    `before_2000`, `before-2000`, and `before 2000` all mean counts from
    before the first year column. A four-digit token is that year column.
    """
    if isinstance(values, str):
        values = [values]
    years = []
    for value in values:
        for piece in str(value).split(','):
            token = piece.strip()
            if not token:
                continue
            key = token.lower().replace('-', '_').replace(' ', '_')
            if key == 'before_2000':
                canonical = 'before_2000'
            elif re.fullmatch(r'\d{4}-\d{2}', token) and 1 <= int(token[5:7]) <= 12:
                canonical = token
            elif token.isdigit():
                canonical = token
            else:
                raise ValueError(
                    f'Unrecognized parent year {token!r}. '
                    'Use before_2000 or a year column such as 2000.'
                )
            if canonical not in years:
                years.append(canonical)
    if not years:
        raise ValueError('At least one parent year is required')
    return years


def parent_year_tag(parent_years):
    return '-'.join(canonical_parent_years(parent_years))


def select_parent_years(frame, fraction):
    """Smallest parent pool whose counts exceed `fraction` of the total.

    Counts are summed over every nucleotide sequence. Yearly tables start
    with before_2000, then 2000, 2001, and so on. Monthly tables walk
    YYYY-MM columns in order. The walk stops at the first period that puts
    the cumulative count strictly above the threshold.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f'parent count fraction must be in (0, 1], got {fraction}')
    years = _year_columns(frame)
    before = _before_2000_counts(frame, years)
    total = float(_count_series(frame).sum())
    if total <= 0:
        raise ValueError('The training counts sum to zero')
    threshold = fraction * total
    has_months = any('-' in year for year in years)
    ordered = list(years)
    if not has_months or float(before.sum()) > 0:
        ordered = ['before_2000'] + ordered
    cumulative = 0.0
    chosen = []
    for year in ordered:
        if year == 'before_2000':
            mass = float(before.sum())
        else:
            mass = float(frame[year].astype(float).sum())
        cumulative += mass
        chosen.append(year)
        if cumulative > threshold:
            break
    logging.info(
        'Parent years %s cover %.0f / %.0f counts (%.4f%%). Threshold is %.4f%%.',
        ', '.join(chosen),
        cumulative,
        total,
        100.0 * cumulative / total,
        100.0 * fraction,
    )
    return chosen


def _before_2000_counts(frame, years):
    if 'before 2000' in frame.columns:
        return frame['before 2000'].astype(float)
    before = _count_series(frame)
    if years:
        before = before - frame[years].astype(float).sum(axis=1)
    # `unknown` is an observation column, not mass from before the first year.
    if 'unknown' in frame.columns:
        before = before - frame['unknown'].astype(float)
    if (before < -1e-6).any():
        raise ValueError('count is smaller than the sum of the year columns')
    return before.clip(lower=0)


def parent_observation_counts(frame, parent_years):
    """Return parent counts, observation counts, and the before-2000 mass.

    `parent_years` selects the parent columns. `before_2000` is the column
    `before 2000` when it exists, otherwise `count` minus the year columns.
    Observation counts are the remaining year columns, plus `unknown` when
    that column is not itself a parent.
    """
    years = _year_columns(frame)
    parent_years = canonical_parent_years(parent_years)
    missing = [year for year in parent_years if year != 'before_2000' and year not in frame.columns]
    if missing:
        available = ', '.join(years)
        raise ValueError(f'Parent years {missing} are not columns in the training table. Available years: {available}')

    before = _before_2000_counts(frame, years)
    parent_count = np.zeros(len(frame), dtype=float)
    for year in parent_years:
        if year == 'before_2000':
            parent_count += before.to_numpy()
        else:
            parent_count += frame[year].astype(float).to_numpy()

    observation_columns = [column for column in years if column not in parent_years]
    if 'unknown' in frame.columns and 'unknown' not in parent_years:
        observation_columns.append('unknown')
    if observation_columns:
        observation_count = frame[observation_columns].astype(float).sum(axis=1).to_numpy()
    else:
        observation_count = np.zeros(len(frame), dtype=float)
    return parent_count, observation_count, before.to_numpy(), observation_columns


def _valid_peptide(nucleotide):
    try:
        peptide = str(Seq(nucleotide).translate())
    except Exception:
        return None
    if any(character in peptide for character in INVALID_AA):
        return None
    return peptide


def _one_mutants(sequence):
    mutants = []
    for index, base in enumerate(sequence):
        for nucleotide in TRANSITION.get(base, []):
            mutants.append((sequence[:index] + nucleotide + sequence[index + 1:], TRANSITION_FACTOR))
        for nucleotide in TRANSVERSION.get(base, []):
            mutants.append((sequence[:index] + nucleotide + sequence[index + 1:], TRANSVERSION_FACTOR))
    return mutants


def _nuc_array(sequence, cache):
    encoded = cache.get(sequence)
    if encoded is None:
        encoded = np.array([_BASE2INT.get(base, 4) for base in sequence], dtype=np.uint8)
        cache[sequence] = encoded
    return encoded


def exact_support(target_sequences, parent_matrix, parent_frequency, mutation_rate, chunk=512):
    """Sum mu^k * transition/transversion weights * parent frequency over parents.

    Distance 0 is excluded. Degenerate bases are ignored.
    """
    if len(target_sequences) == 0:
        return np.zeros(0, dtype=np.float64)
    cache = {}
    parent_valid = (parent_matrix < 4).astype(np.float32)
    parent_bases = [(parent_matrix == base).astype(np.float32) for base in range(4)]
    targets = np.stack([_nuc_array(sequence, cache) for sequence in target_sequences])
    support = np.zeros(len(target_sequences), dtype=np.float64)
    log_mu = np.log(mutation_rate)
    log_transition = np.log(TRANSITION_FACTOR)
    log_transversion = np.log(TRANSVERSION_FACTOR)
    for start in range(0, len(targets), chunk):
        block = targets[start:start + chunk]
        valid = (block < 4).astype(np.float32) @ parent_valid.T
        matches = np.zeros_like(valid)
        for base in range(4):
            matches += (block == base).astype(np.float32) @ parent_bases[base].T
        distance = valid - matches
        transitions = (
            (block == 0).astype(np.float32) @ parent_bases[2].T
            + (block == 2).astype(np.float32) @ parent_bases[0].T
            + (block == 1).astype(np.float32) @ parent_bases[3].T
            + (block == 3).astype(np.float32) @ parent_bases[1].T
        )
        log_probability = (
            distance * log_mu
            + transitions * log_transition
            + (distance - transitions) * log_transversion
        )
        contribution = np.where(distance > 0, np.exp(log_probability), 0.0) * parent_frequency[None, :]
        support[start:start + chunk] = contribution.sum(axis=1)
    return support


def min_distance_to_pool(target_sequences, parent_matrix, chunk=512):
    if len(target_sequences) == 0:
        return np.zeros(0, dtype=np.float64)
    cache = {}
    parent_valid = (parent_matrix < 4).astype(np.float32)
    parent_bases = [(parent_matrix == base).astype(np.float32) for base in range(4)]
    targets = np.stack([_nuc_array(sequence, cache) for sequence in target_sequences])
    distances = np.zeros(len(target_sequences), dtype=np.float64)
    for start in range(0, len(targets), chunk):
        block = targets[start:start + chunk]
        valid = (block < 4).astype(np.float32) @ parent_valid.T
        matches = np.zeros_like(valid)
        for base in range(4):
            matches += (block == base).astype(np.float32) @ parent_bases[base].T
        distances[start:start + chunk] = (valid - matches).min(axis=1)
    return distances


def _substitution_options(base):
    return TRANSITION.get(base, []) + TRANSVERSION.get(base, [])


def _add_shell(
    parents,
    distance,
    blocked,
    support_by_nucleotide,
    parent_matrix,
    parent_frequency,
    mutation_rate,
    batch_size=2048,
):
    """Add sequences reached by `distance` substitutions and keep those above EPS.

    An intermediate that is already known or already supported is not expanded.
    Candidates are scored in batches so the full shell does not have to sit in memory.
    """
    seen = set()
    batch = []
    added = 0

    def flush():
        nonlocal added, batch
        if not batch:
            return
        values = exact_support(batch, parent_matrix, parent_frequency, mutation_rate)
        for sequence, value in zip(batch, values):
            if value < EPS_EXPSUM or sequence in support_by_nucleotide:
                continue
            if _valid_peptide(sequence) is None:
                continue
            support_by_nucleotide[sequence] = float(value)
            added += 1
        batch = []

    def consider(sequence):
        nonlocal batch
        if sequence in blocked or sequence in seen:
            return
        seen.add(sequence)
        batch.append(sequence)
        if len(batch) >= batch_size:
            flush()

    def walk(sequence, steps_left):
        if steps_left == 1:
            for index, base in enumerate(sequence):
                prefix = sequence[:index]
                suffix = sequence[index + 1:]
                for nucleotide in _substitution_options(base):
                    consider(prefix + nucleotide + suffix)
            return
        for index, base in enumerate(sequence):
            prefix = sequence[:index]
            suffix = sequence[index + 1:]
            for nucleotide in _substitution_options(base):
                mutant = prefix + nucleotide + suffix
                if mutant in blocked:
                    continue
                walk(mutant, steps_left - 1)

    for parent_index, parent in enumerate(parents):
        walk(parent, distance)
        if (parent_index + 1) % 10 == 0 or parent_index + 1 == len(parents):
            flush()
            logging.info(
                'shell-%d progress: %d / %d parents, %d candidates scored, %d kept',
                distance,
                parent_index + 1,
                len(parents),
                len(seen),
                added,
            )
    flush()
    return added


def _shell1_support(parent_sequences, parent_frequency_by_sequence, known_nucleotides, mutation_rate):
    support = defaultdict(float)
    for sequence in parent_sequences:
        frequency = parent_frequency_by_sequence[sequence]
        for mutant, factor in _one_mutants(sequence):
            if mutant in known_nucleotides:
                continue
            if _valid_peptide(mutant) is None:
                continue
            support[mutant] += mutation_rate * factor * frequency
    return dict(support)


def _build_support(frame, mutation_rate, parent_years, obs_dist_max):
    parent_years = canonical_parent_years(parent_years)
    parent_count, observation_count, before_count, observation_columns = parent_observation_counts(frame, parent_years)
    nucleotides = frame['nuc_sequence'].astype(str).tolist()
    known_nucleotides = set(nucleotides)
    parent_mask = parent_count > 0
    observation_mask = observation_count > 0
    parent_sequences = [seq for seq, keep in zip(nucleotides, parent_mask) if keep]
    parent_counts = parent_count[parent_mask]
    if parent_counts.sum() <= 0:
        raise ValueError(f'The parent pool ({", ".join(parent_years)}) is empty')
    parent_frequency = parent_counts / parent_counts.sum()
    frequency_by_sequence = dict(zip(parent_sequences, parent_frequency))
    parent_set = set(parent_sequences)

    sequence_length = len(parent_sequences[0])
    self_probability = float(np.exp(sequence_length * np.log1p(-mutation_rate)))
    rescue_bound = (mutation_rate * TRANSITION_FACTOR) ** 2
    shell1_rescue = rescue_bound if obs_dist_max >= 2 else 0.0

    cache = {}
    parent_matrix = np.stack([_nuc_array(sequence, cache) for sequence in parent_sequences])

    shell1 = _shell1_support(parent_sequences, frequency_by_sequence, known_nucleotides, mutation_rate)
    pre_candidates = [seq for seq, value in shell1.items() if value + shell1_rescue >= EPS_EXPSUM]
    pre_candidates.sort()
    shell1_exact = exact_support(pre_candidates, parent_matrix, parent_frequency, mutation_rate)
    support_by_nucleotide = {}
    for sequence, value in zip(pre_candidates, shell1_exact):
        if value >= EPS_EXPSUM and _valid_peptide(sequence) is not None:
            support_by_nucleotide[sequence] = float(value)

    shell_counts = {}
    for distance in range(2, obs_dist_max + 1):
        rescue_bound_d = (mutation_rate * TRANSITION_FACTOR) ** distance
        if rescue_bound_d < EPS_EXPSUM:
            break
        frequency_min = 0.5 * EPS_EXPSUM / rescue_bound_d
        frequent_parents = [
            seq for seq in parent_sequences
            if frequency_by_sequence[seq] >= frequency_min
        ]
        blocked = known_nucleotides.union(support_by_nucleotide)
        added = _add_shell(
            frequent_parents,
            distance,
            blocked,
            support_by_nucleotide,
            parent_matrix,
            parent_frequency,
            mutation_rate,
        )
        shell_counts[distance] = added
        logging.info(
            'shell-%d parents with enough frequency: %d, candidates added: %d',
            distance,
            len(frequent_parents),
            added,
        )
    n_shell2 = shell_counts.get(2, 0)

    peptide_to_candidates = defaultdict(list)
    for sequence in support_by_nucleotide:
        peptide_to_candidates[_valid_peptide(sequence)].append(sequence)

    observation_sequences = [seq for seq, keep in zip(nucleotides, observation_mask) if keep]
    distances = min_distance_to_pool(observation_sequences, parent_matrix)
    kept_observations = [
        sequence for sequence, distance in zip(observation_sequences, distances)
        if distance <= obs_dist_max
    ]
    n_excluded = len(observation_sequences) - len(kept_observations)

    observation_support = {}
    observed_exact = exact_support(kept_observations, parent_matrix, parent_frequency, mutation_rate)
    for sequence, value in zip(kept_observations, observed_exact):
        self_support = self_probability * float(frequency_by_sequence[sequence]) if sequence in parent_set else 0.0
        observation_support[sequence] = float(value) + self_support

    peptide_to_observed = defaultdict(list)
    for sequence in kept_observations:
        peptide = _valid_peptide(sequence)
        if peptide is not None:
            peptide_to_observed[peptide].append(sequence)

    batch_peptides = set()
    for sequence in observation_sequences:
        peptide = _valid_peptide(sequence)
        if peptide is not None:
            batch_peptides.add(peptide)
    observed_peptides = sorted(peptide for peptide in batch_peptides if peptide in peptide_to_observed)
    unlabeled_peptides = sorted(set(peptide_to_candidates) - set(observed_peptides))

    candidate_nucleotides = sorted(support_by_nucleotide)
    candidate_index = {sequence: index for index, sequence in enumerate(candidate_nucleotides)}
    observed_nucleotides = sorted(observation_support)
    observed_index = {sequence: index for index, sequence in enumerate(observed_nucleotides)}

    def membership(peptide_list, peptide_to_nucleotides, nucleotide_index):
        peptide_index = []
        nucleotide_rows = []
        for row, peptide in enumerate(peptide_list):
            for nucleotide in peptide_to_nucleotides.get(peptide, []):
                if nucleotide in nucleotide_index:
                    peptide_index.append(row)
                    nucleotide_rows.append(nucleotide_index[nucleotide])
        return np.asarray(peptide_index, dtype=np.int64), np.asarray(nucleotide_rows, dtype=np.int64)

    obs_pep_idx, obs_nuc_idx = membership(observed_peptides, peptide_to_observed, observed_index)
    pos_pep_idx, pos_nuc_idx = membership(observed_peptides, peptide_to_candidates, candidate_index)
    unobs_pep_idx, unobs_nuc_idx = membership(unlabeled_peptides, peptide_to_candidates, candidate_index)

    positive_support = np.array([support_by_nucleotide[seq] for seq in candidate_nucleotides], dtype=np.float64)
    all_support = np.concatenate([
        positive_support,
        np.array([observation_support[seq] for seq in observed_nucleotides], dtype=np.float64),
    ])
    positive_values = all_support[all_support > 0]
    support_median = float(np.median(positive_values)) if len(positive_values) else 1.0

    summary = {
        'n_parents': len(parent_sequences),
        'n_parent_only': int((parent_mask & ~observation_mask).sum()),
        'n_observations': len(observation_sequences),
        'n_observations_kept': len(kept_observations),
        'n_observations_excluded': n_excluded,
        'parent_years': parent_years,
        'observation_columns': observation_columns,
        'before_2000_mass': float(np.sum(before_count)),
        'before_2000_used': 'before_2000' in parent_years,
        'parent_mass': float(parent_counts.sum()),
        'n_observed_peptides': len(observed_peptides),
        'n_unlabeled_peptides': len(unlabeled_peptides),
        'n_candidate_nucleotides': len(candidate_nucleotides),
        'obs_dist_max': obs_dist_max,
        'n_shell2': n_shell2,
        'n_shell3': shell_counts.get(3, 0),
        'support_median': support_median,
        'sequence_length': sequence_length,
        'self_probability': self_probability,
    }
    logging.info(
        'Parents (%s): %d nucleotides, mass %.0f',
        ', '.join(parent_years),
        summary['n_parents'],
        summary['parent_mass'],
    )
    if summary['before_2000_mass'] > 0 and not summary['before_2000_used']:
        logging.info(
            'before-2000 mass %.0f is not in the parent years, so it is left out of both pools',
            summary['before_2000_mass'],
        )
    logging.info(
        'Observations (%s): %d nucleotides, %d within %d mutations, %d excluded',
        ', '.join(observation_columns) if observation_columns else 'none',
        summary['n_observations'],
        summary['n_observations_kept'],
        obs_dist_max,
        summary['n_observations_excluded'],
    )
    logging.info(
        'Peptides: %d observed, %d unlabeled; candidate nucleotides: %d',
        summary['n_observed_peptides'],
        summary['n_unlabeled_peptides'],
        summary['n_candidate_nucleotides'],
    )
    return {
        'observed_peptides': observed_peptides,
        'unlabeled_peptides': unlabeled_peptides,
        's_cand': np.array([support_by_nucleotide[seq] for seq in candidate_nucleotides], dtype=np.float32),
        's_obs': np.array([observation_support[seq] for seq in observed_nucleotides], dtype=np.float32),
        'obs_pep_idx': obs_pep_idx,
        'obs_nuc_idx': obs_nuc_idx,
        'pos_pep_idx': pos_pep_idx,
        'pos_nuc_idx': pos_nuc_idx,
        'unobs_pep_idx': unobs_pep_idx,
        'unobs_nuc_idx': unobs_nuc_idx,
        'beta_init': float(-np.log(support_median)),
        'summary': summary,
    }


def prepare_pulse_support(property_name, mutation_rate, parent_years, data_root='data', obs_dist_max=OBS_DIST_MAX):
    """Build or load the cached parent/observation support for one property."""
    if obs_dist_max < 1:
        raise ValueError(f'obs_dist_max must be at least 1, got {obs_dist_max}')
    parent_years = canonical_parent_years(parent_years)
    data_dir = os.path.join(data_root, property_name)
    cache_path = os.path.join(
        data_dir,
        f'{property_name}_pulse_parents_{parent_year_tag(parent_years)}_mu_{mutation_rate}_dist_{obs_dist_max}.pkl',
    )
    os.makedirs(data_dir, exist_ok=True)
    lock_path = cache_path + '.lock'
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(cache_path):
            logging.info('Loading PULSE support from %s', cache_path)
            with open(cache_path, 'rb') as handle:
                prepared = pickle.load(handle)
            summary = prepared['summary']
            logging.info(
                'Loaded parents=%d, observed peptides=%d, unlabeled peptides=%d',
                summary['n_parents'],
                summary['n_observed_peptides'],
                summary['n_unlabeled_peptides'],
            )
            return prepared

        frame = pd.read_csv(os.path.join(data_dir, f'{property_name}_train.csv'))
        logging.info(
            'Building PULSE support for %s. Parent years: %s. Observation distance: %d. Other year columns are observations.',
            property_name,
            ', '.join(parent_years),
            obs_dist_max,
        )
        prepared = _build_support(frame, mutation_rate, parent_years, obs_dist_max)
        temporary = cache_path + '.tmp'
        with open(temporary, 'wb') as handle:
            pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
        logging.info('Saved PULSE support to %s', cache_path)
        return prepared
