"""Peptide representations (featurizers) — registry pattern.

A featurizer maps a list of peptide strings to a float32 (n, dim) matrix and
carries a `.dim` attribute (set on first featurize call). Non-canonical
residues ('X', '-', '*') must be handled gracefully.

To add a new representation (e.g. ESM2):
    1. write a class with .featurize(pep_list) -> np.ndarray and .dim;
    2. register it:  FEATURIZERS['esm2'] = ESM2Featurizer
    3. select per task:  representation: {name: esm2}

ESM2 skeleton (drop in when needed):

    class ESM2Featurizer:
        name = 'esm2'
        def __init__(self):
            import torch, esm                      # pip install fair-esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model.eval()                      # move to GPU if available
            self.dim = 1280
        def featurize(self, peptides):             # mean-pooled residue embs
            import torch
            bc = self.alphabet.get_batch_converter()
            _, _, toks = bc([(str(i), p) for i, p in enumerate(peptides)])
            with torch.no_grad():
                out = self.model(toks, repr_layers=[33])['representations'][33]
            mask = (toks != self.alphabet.padding_idx)
            mask &= (toks != self.alphabet.cls_idx) & (toks != self.alphabet.eos_idx)
            emb = [(out[i, mask[i]].mean(0)).numpy() for i in range(len(peptides))]
            return np.stack(emb).astype(np.float32)
    # ESM2 is length-agnostic, which also lifts chemical3's equal-length rule.
"""
import numpy as np

# (Hydrophobicity, Mol, Volume) per amino acid, standardized to zero mean
# and unit variance across the twenty amino acids.
CHEM3 = {
    'G': (0.3034, -1.8255, -2.0116), 'L': (-1.2380, 0.0875, 0.6305),
    'F': (-1.4379, 1.2005, 1.2056), 'A': (-0.4606, -1.3471, -1.3052),
    'I': (-1.1491, 0.0875, 0.6305), 'V': (-0.8027, -0.3931, -0.0312),
    'M': (-0.7982, 0.4565, 0.5364), 'C': (-0.2430, -0.4957, -0.8120),
    'W': (-0.6294, 2.4349, 2.1449), 'T': (0.3300, -0.7218, -0.6236),
    'S': (0.3522, -1.2002, -1.2953), 'Y': (-0.9448, 1.3738, 1.2973),
    'P': (-1.1358, -0.5793, -0.7079), 'N': (1.0852, -0.5000, -0.6732),
    'D': (0.8497, -0.6623, -0.7475), 'Q': (0.8764, -0.0193, 0.0630),
    'E': (0.2679, -0.1816, -0.0709), 'K': (1.9336, 0.4467, 0.6776),
    'R': (1.1873, 1.0969, 0.7966), 'H': (1.6537, 0.7414, 0.2959),
}
_ZERO3 = (0.0, 0.0, 0.0)


class Chemical3:
    """Per-residue (hydrophobicity, mol, volume) triplets, flattened:
    dim = 3 * L. All peptides within a task must share one length;
    non-canonical residues map to (0, 0, 0)."""
    name = 'chemical3'

    def __init__(self):
        self.dim = None

    def featurize(self, peptides):
        if len(peptides) == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        L = len(peptides[0])
        if any(len(p) != L for p in peptides):
            raise ValueError('chemical3 requires equal-length peptides '
                             '(use a length-agnostic featurizer otherwise)')
        if self.dim is not None and self.dim != 3 * L:
            raise ValueError(f'chemical3: peptide length {L} does not match the '
                             f'training length {self.dim // 3}')
        self.dim = 3 * L
        return np.array([[CHEM3.get(aa, _ZERO3) for aa in p] for p in peptides],
                        dtype=np.float32).reshape(len(peptides), -1)


FEATURIZERS = {'chemical3': Chemical3}


def featurize_chunked(feat, peptides, chunk=100_000):
    """feat.featurize over slices, written into one preallocated float32 array.

    Bit-identical to feat.featurize(peptides) (same per-row computation, same
    order, same dtype) but peak memory is output + one chunk instead of the
    full numpy build + the torch copy. Wrap with torch.from_numpy (zero-copy).
    """
    n = len(peptides)
    if n <= chunk:
        return feat.featurize(peptides)
    first = feat.featurize(peptides[:chunk])
    out = np.empty((n, first.shape[1]), dtype=np.float32)
    out[:chunk] = first
    del first
    for s in range(chunk, n, chunk):
        out[s:s + chunk] = feat.featurize(peptides[s:s + chunk])
    return out


def build_featurizer(cfg):
    name = cfg.representation.name
    cls = FEATURIZERS.get(name)
    if cls is None:
        raise ValueError(f"unknown representation {name!r}; "
                         f'available: {sorted(FEATURIZERS)}')
    return cls()
