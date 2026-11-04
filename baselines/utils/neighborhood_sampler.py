"""Uniform draws from the unobserved three-mutation neighborhood.

The proposal picks a parent with probability proportional to its raw neighborhood
size, then a peptide uniformly inside that neighborhood. Observed peptides are
rejected. Any other peptide u is kept with probability 1 / n(u), where n(u) is
the number of observed peptides within the mutation radius of u. Accepted
peptides are uniformly distributed on

    U = {u not in P : 1 <= Hamming distance from u to some observed peptide <= radius}.
"""

import numpy as np

from utils.gen_unlabeled import _N_AA, _codes, _decode


def _shell_sizes(n_mutable, radius, n_alternatives=19):
    """Counts of peptides at distance 1, 2, ..., radius from one peptide.

    Output:
        ndarray int64: length `radius`. A shell past n_mutable is zero.
    """
    shells = np.zeros(int(radius), dtype=np.int64)
    choose = 1
    limit = min(int(radius), int(n_mutable))
    for distance in range(1, limit + 1):
        choose = choose * (n_mutable - distance + 1) // distance
        shells[distance - 1] = choose * (n_alternatives ** distance)
    return shells


class UniformNeighborhoodSampler:
    """Rejection sampler whose accepted peptides are uniform on U.

    Inputs:
        observed (list[str]): distinct peptides that define the neighborhoods.
        radius (int): largest Hamming distance, 3 for CBC.
        rng (numpy Generator): random source.
    """

    def __init__(self, observed, radius, rng):
        if radius < 1:
            raise ValueError(f'radius must be at least 1, got {radius}')
        peptides = [peptide.upper() for peptide in observed]
        if len(peptides) != len(set(peptides)):
            raise ValueError('CBC neighborhoods are built from distinct observed peptides')
        self.rng = rng
        self.radius = int(radius)
        self.codes = _codes(peptides)
        self.n_observed, self.length = self.codes.shape
        self.observed = set(peptides)
        self.mutable = [np.flatnonzero(self.codes[row] < _N_AA) for row in range(self.n_observed)]
        self.shells = np.stack([
            _shell_sizes(int(sites.size), self.radius) for sites in self.mutable
        ])
        self.neighborhood_size = self.shells.sum(axis=1)
        eligible = np.flatnonzero(self.neighborhood_size > 0)
        if eligible.size == 0:
            raise RuntimeError('no observed peptide has a standard amino acid to mutate')
        self.eligible = eligible
        weights = self.neighborhood_size[eligible].astype(np.float64)
        self.parent_prob = weights / weights.sum()
        self._block_index = self._build_block_index()

    def _build_block_index(self):
        """Map each of radius + 1 blocks to the observed rows that share it.

        A peptide within `radius` edits matches an observed peptide on at least
        one of those blocks. Short sequences are compared against every row.
        """
        if self.length <= self.radius:
            return None
        n_parts = self.radius + 1
        edges = np.linspace(0, self.length, n_parts + 1, dtype=int)
        blocks = []
        for start, end in zip(edges[:-1], edges[1:]):
            start, end = int(start), int(end)
            if start == end:
                return None
            table = {}
            piece = np.ascontiguousarray(self.codes[:, start:end])
            for row in range(self.n_observed):
                table.setdefault(piece[row].tobytes(), []).append(row)
            blocks.append((start, end, {
                key: np.asarray(rows, dtype=np.int32) for key, rows in table.items()
            }))
        return blocks

    def _candidate_rows(self, code):
        """Observed-row indices that could lie within the radius of `code`."""
        if self._block_index is None:
            return np.arange(self.n_observed, dtype=np.int32)
        parts = []
        for start, end, table in self._block_index:
            rows = table.get(code[start:end].tobytes())
            if rows is not None:
                parts.append(rows)
        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def parent_count(self, code):
        """How many observed peptides have this peptide in their raw neighborhood.

        A raw neighbor matches on frozen positions (X and dash) and differs in
        1 through `radius` standard amino acids.
        """
        rows = self._candidate_rows(code)
        if rows.size == 0:
            return 0
        parents = self.codes[rows]
        diff = parents != code
        frozen = parents >= _N_AA
        invalid = np.any(diff & frozen, axis=1)
        distance = np.sum(diff & ~frozen, axis=1)
        close = (~invalid) & (distance >= 1) & (distance <= self.radius)
        return int(np.count_nonzero(close))

    def _mutate(self, parent):
        """One peptide drawn uniformly from the raw neighborhood of `parent`."""
        shells = self.shells[parent]
        distances = np.flatnonzero(shells > 0)
        probabilities = shells[distances].astype(np.float64)
        probabilities /= probabilities.sum()
        distance = int(distances[int(self.rng.choice(distances.size, p=probabilities))]) + 1
        sites = self.mutable[parent]
        chosen = self.rng.choice(sites, size=distance, replace=False)
        row = self.codes[parent].copy()
        for position in np.atleast_1d(chosen):
            current = int(row[position])
            replacement = int(self.rng.integers(0, _N_AA - 1))
            if replacement >= current:
                replacement += 1
            row[position] = replacement
        return row

    def sample(self, num_sequences, max_attempts_factor=1000000):
        """Draw `num_sequences` peptides uniformly from U.

        Output:
            peptides (list[str]): accepted unobserved peptides.
            attempts (int): raw proposals used, including rejected ones.
        """
        if num_sequences <= 0:
            raise ValueError(f'num_sequences must be positive, got {num_sequences}')
        accepted = []
        attempts = 0
        limit = int(num_sequences) * int(max_attempts_factor)
        while len(accepted) < num_sequences:
            remaining = num_sequences - len(accepted)
            chunk = max(remaining * 8, 32)
            if attempts + chunk > limit:
                raise RuntimeError(
                    f'Could not draw {num_sequences} uniform peptides within {limit} proposals'
                )
            parents = self.rng.choice(self.eligible, size=chunk, p=self.parent_prob)
            rows = np.empty((chunk, self.length), dtype=np.uint8)
            for index, parent in enumerate(parents.tolist()):
                rows[index] = self._mutate(int(parent))
            texts = _decode(rows)
            for row, text in zip(rows, texts):
                attempts += 1
                if text in self.observed:
                    continue
                parents_near = self.parent_count(row)
                if parents_near < 1:
                    raise RuntimeError('a proposed peptide is outside every observed neighborhood')
                if float(self.rng.random()) < 1.0 / parents_near:
                    accepted.append(text)
                    if len(accepted) == num_sequences:
                        break
        return accepted, attempts
