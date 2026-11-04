import random

import numpy as np


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# X and a gap stay put. Only the 20 standard amino acids are mutated.
_EXTRA = 'X-'
_DECODE = np.frombuffer((AMINO_ACIDS + _EXTRA).encode('ascii'), dtype=np.uint8)
_N_AA = len(AMINO_ACIDS)


def RAND_unlabeled(num_sequences, length_sequence, seed=42):
    """Draw peptides uniformly from the 20 amino acids.

    Each position is an independent uniform draw, so every peptide of this
    length has the same probability. The same seed always returns the same set.
    """
    rng = random.Random(seed)
    return [''.join(rng.choices(AMINO_ACIDS, k=length_sequence)) for _ in range(num_sequences)]


def _codes(peptides):
    """Encode peptides as rows of amino-acid indices.

    Standard amino acids are 0..19. X is 20 and a dash is 21. Those two are
    not mutation sites. Any other letter is rejected.

    Inputs:
        peptides (list[str]): one common length.
    Output:
        ndarray uint8: (n, L).
    """
    if not peptides:
        raise ValueError('near-observed sampling needs at least one observed peptide')
    length = len(peptides[0])
    if length < 1 or any(len(peptide) != length for peptide in peptides):
        raise ValueError('observed peptides must share one positive length')
    table = np.full(256, -1, dtype=np.int16)
    for index, amino in enumerate(AMINO_ACIDS + _EXTRA):
        table[ord(amino)] = index
    raw = np.frombuffer(''.join(peptides).encode('ascii'), dtype=np.uint8)
    codes = table[raw].reshape(len(peptides), length)
    if np.any(codes < 0):
        raise ValueError('observed peptides contain a letter outside the 20 amino acids, X, and -')
    return codes.astype(np.uint8, copy=False)


def _decode(rows):
    """Turn encoded rows back into peptide strings."""
    return [_DECODE[row].tobytes().decode('ascii') for row in rows]


def _mutation_distance(left, right):
    """Standard-amino-acid edits from left rows to right rows.

    A dash or X must match. A difference that introduces or removes one of
    those symbols is not a mutation the sampler can draw, so those pairs
    get distance -1.

    Inputs:
        left (ndarray): (a, L) codes.
        right (ndarray): (b, L) codes.
    Output:
        ndarray int16: (a, b) distances.
    """
    diff = left[:, None, :] != right[None, :, :]
    frozen = (left[:, None, :] >= _N_AA) | (right[None, :, :] >= _N_AA)
    distance = (diff & ~frozen).sum(axis=2).astype(np.int16)
    distance[(diff & frozen).any(axis=2)] = -1
    return distance


def _ball_count(length, max_mutations):
    """Number of peptides at distance 1 through max_mutations from one peptide.

    Each of the chosen sites is replaced by one of the other 19 amino acids.
    """
    total = 0
    choose = 1
    for distance in range(1, min(int(max_mutations), length) + 1):
        choose = choose * (length - distance + 1) // distance
        total += choose * (19 ** distance)
    return int(total)


def _close_observed_counts(codes, radius):
    """Count other rows at Hamming distance 1 through radius.

    Rows that differ in at most `radius` positions share an identical block
    when the sequence is split into radius + 1 blocks. Pairs are compared
    inside those blocks. Short sequences are compared directly.
    """
    n_rows, length = codes.shape
    hits = np.zeros(n_rows, dtype=np.int64)
    if n_rows < 2 or radius < 1:
        return hits
    if length <= radius:
        for left in range(n_rows):
            distance = _mutation_distance(codes[left:left + 1], codes[left + 1:])[0]
            close = np.flatnonzero((distance >= 1) & (distance <= radius)) + left + 1
            hits[left] += close.size
            hits[close] += 1
        return hits

    n_parts = radius + 1
    edges = np.linspace(0, length, n_parts + 1, dtype=int)
    seen = set()
    block = 128
    for part in range(n_parts):
        start, end = int(edges[part]), int(edges[part + 1])
        if start == end:
            continue
        piece = np.ascontiguousarray(codes[:, start:end])
        keys = piece.view(np.dtype((np.void, piece.shape[1] * piece.dtype.itemsize))).ravel()
        order = np.argsort(keys, kind='mergesort')
        sorted_keys = keys[order]
        change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        bounds = np.concatenate(([0], change, [n_rows]))
        for group_start, group_end in zip(bounds[:-1], bounds[1:]):
            if group_end - group_start < 2:
                continue
            members = order[group_start:group_end]
            sub = codes[members]
            group_size = sub.shape[0]
            for left0 in range(0, group_size, block):
                left1 = min(group_size, left0 + block)
                left = sub[left0:left1]
                for right0 in range(left0, group_size, block):
                    right1 = min(group_size, right0 + block)
                    right = sub[right0:right1]
                    distance = _mutation_distance(left, right)
                    if left0 == right0:
                        row_i, row_j = np.triu_indices(left1 - left0, k=1)
                        distance = distance[row_i, row_j]
                        keep = np.flatnonzero((distance >= 1) & (distance <= radius))
                        global_i = members[left0 + row_i[keep]]
                        global_j = members[left0 + row_j[keep]]
                    else:
                        row_i, row_j = np.nonzero((distance >= 1) & (distance <= radius))
                        global_i = members[left0 + row_i]
                        global_j = members[right0 + row_j]
                    for first, second in zip(global_i.tolist(), global_j.tolist()):
                        low, high = (first, second) if first < second else (second, first)
                        key = low * n_rows + high
                        if key in seen:
                            continue
                        seen.add(key)
                        hits[low] += 1
                        hits[high] += 1
    return hits


def unobserved_neighborhood_sizes(observed, max_mutations=2):
    """Unobserved peptides within max_mutations of each observed peptide.

    The count is every peptide at distance 1 through max_mutations, minus
    observed peptides that fall in that ball. The parent itself is distance 0
    and is not included.

    Inputs:
        observed (list[str]): peptides, in the same order the sampler indexes.
        max_mutations (int): radius, 1 or 2 for the near-observed sampler.
    Output:
        ndarray int64: one neighborhood size per observed peptide.
    """
    parents = [peptide.upper() for peptide in observed]
    codes = _codes(parents)
    mutable = (codes < _N_AA).sum(axis=1).astype(int)
    radius = int(max_mutations)
    ball = np.array([_ball_count(int(count), radius) for count in mutable], dtype=np.int64)
    sizes = ball - _close_observed_counts(codes, radius)
    can_mutate = mutable > 0
    if np.any(sizes[can_mutate] <= 0):
        raise ValueError('an observed peptide has no unobserved sequences within the mutation radius')
    return sizes


def sample_near_observed(observed, num_sequences, rng, max_mutations=2, excluded=None, return_parents=False):
    """Draw peptides at most max_mutations away from one observed peptide.

    Each draw picks an observed peptide and changes between 1 and max_mutations
    standard amino acids, with that count chosen uniformly. Positions that are
    X or a dash are left as they are. A result that is already observed is
    rejected.
    The same generator state returns the same peptides.

    Inputs:
        observed (list[str]): peptides that can be the mutation parent.
        num_sequences (int): number of accepted peptides.
        rng (numpy Generator): random source.
        max_mutations (int): largest Hamming distance from the parent.
        excluded (set[str] or None): peptides to reject. Defaults to observed.
        return_parents (bool): also return the parent index of each draw.
    Output:
        list[str]: num_sequences peptides outside excluded.
        ndarray int64, optional: parent row in `observed` for each peptide.
    """
    if num_sequences <= 0:
        raise ValueError(f'num_sequences must be positive, got {num_sequences}')
    if max_mutations < 1:
        raise ValueError(f'max_mutations must be at least 1, got {max_mutations}')
    parents = [peptide.upper() for peptide in observed]
    codes = _codes(parents)
    n_observed, length = codes.shape
    max_k = min(int(max_mutations), length)
    mutable = [np.flatnonzero(codes[row] < _N_AA) for row in range(n_observed)]
    eligible = np.flatnonzero([sites.size > 0 for sites in mutable])
    if eligible.size == 0:
        raise RuntimeError('no observed peptide has a standard amino acid to mutate')
    blocked = excluded if excluded is not None else set(parents)
    accepted = []
    accepted_parents = []
    rounds = 0
    while len(accepted) < num_sequences:
        rounds += 1
        if rounds > 100:
            raise RuntimeError(
                f'Could not draw {num_sequences} peptides within {max_k} mutations of an observed peptide'
            )
        need = num_sequences - len(accepted)
        draw = need + max(32, need // 10)
        parent_ids = eligible[rng.integers(0, eligible.size, size=draw)]
        rows = codes[parent_ids].copy()
        for draw_index, parent in enumerate(parent_ids.tolist()):
            sites = mutable[parent]
            k = int(rng.integers(1, min(max_k, sites.size) + 1))
            chosen = rng.choice(sites, size=k, replace=False)
            for position in np.atleast_1d(chosen):
                current = int(rows[draw_index, position])
                replacement = int(rng.integers(0, _N_AA - 1))
                if replacement >= current:
                    replacement += 1
                rows[draw_index, position] = replacement
        texts = _decode(rows)
        for peptide, parent in zip(texts, parent_ids.tolist()):
            if peptide in blocked:
                continue
            accepted.append(peptide)
            accepted_parents.append(parent)
            if len(accepted) == num_sequences:
                break
    if return_parents:
        return accepted, np.asarray(accepted_parents, dtype=np.int64)
    return accepted
