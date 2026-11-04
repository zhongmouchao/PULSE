import fcntl
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Amino-acid property table: (Hydrophobicity, Mol, Volume).
# Same values as the one-batch Fusion notebook.
convert = [
    ('G',  0.3034, -1.8255, -2.0116),
    ('L', -1.2380,  0.0875,  0.6305),
    ('F', -1.4379,  1.2005,  1.2056),
    ('A', -0.4606, -1.3471, -1.3052),
    ('I', -1.1491,  0.0875,  0.6305),
    ('V', -0.8027, -0.3931, -0.0312),
    ('M', -0.7982,  0.4565,  0.5364),
    ('C', -0.2430, -0.4957, -0.8120),
    ('W', -0.6294,  2.4349,  2.1449),
    ('T',  0.3300, -0.7218, -0.6236),
    ('S',  0.3522, -1.2002, -1.2953),
    ('Y', -0.9448,  1.3738,  1.2973),
    ('P', -1.1358, -0.5793, -0.7079),
    ('N',  1.0852, -0.5000, -0.6732),
    ('D',  0.8497, -0.6623, -0.7475),
    ('Q',  0.8764, -0.0193,  0.0630),
    ('E',  0.2679, -0.1816, -0.0709),
    ('K',  1.9336,  0.4467,  0.6776),
    ('R',  1.1873,  1.0969,  0.7966),
    ('H',  1.6537,  0.7414,  0.2959),
]

convert_df = pd.DataFrame(convert, columns=['Amino Acid', 'Hydrophobicity', 'Mol', 'Volume'])
aa_to_properties = {
    row['Amino Acid']: [row['Hydrophobicity'], row['Mol'], row['Volume']]
    for _, row in convert_df.iterrows()
}
# Gaps and unknown residues have no amino-acid properties; use zeros.
aa_to_properties['-'] = [0.0, 0.0, 0.0]
aa_to_properties['X'] = [0.0, 0.0, 0.0]

#this is to get the CHEM representation of a peptide sequence

def chem(peptide):
    feature = []
    for aa in peptide:
        key = aa if aa == '-' else aa.capitalize()
        feature.append(aa_to_properties[key])
    return feature

#this is to get the ESM2 embedding of a list of peptides
@torch.no_grad()
def esm2_get_embedding(peptides, device, batch_size=32, model=None, tokenizer=None):
    from transformers import EsmModel, EsmTokenizer
    embeddings = []
    model_name = "facebook/esm2_t30_150M_UR50D" # 0.5s for one sequence on A100
    if tokenizer is None:
        tokenizer = EsmTokenizer.from_pretrained(model_name)
    if model is None:
        model = EsmModel.from_pretrained(model_name).to(device)
        model.eval()
    esm2_model = model
    for i in tqdm(range(0, len(peptides), batch_size), desc="ESM2 embedding"):
        batch = peptides[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = esm2_model(**inputs)
        hidden_states = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]

        # Mask padding tokens, average only over real amino acids
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        lengths = attention_mask.sum(dim=1).unsqueeze(-1)
        mean_embeddings = sum_hidden / lengths  # [batch, hidden_dim]
        embeddings.append(mean_embeddings.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def get_features(aa_seq, representation):
    if representation == 'CHEM':
        aa_feature = [chem(seq) for seq in aa_seq]
        aa_feature = np.array(aa_feature).reshape(len(aa_feature), -1)
    elif representation == 'ESM2':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model_name = "facebook/esm2_t6_8M_UR50D"
        #model_name = "facebook/esm2_t12_35M_UR50D"  # Small & fast; you can choose a larger checkpoint if needed
        #model_name = "facebook/esm2_t36_3B_UR50D"  # Best; 20s for one sequence on A100
        #model_name = "facebook/esm2_t33_650M_UR50D" # Good trade-off; 0.5s for one sequence on A100
        model_name = "facebook/esm2_t30_150M_UR50D" # 0.5s for one sequence on A100
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        esm2_model = EsmModel.from_pretrained(model_name).to(device)
        esm2_model.eval()
        aa_feature = esm2_get_embedding(
            peptides=aa_seq,
            device=device,
            batch_size=32,
            model=esm2_model,
            tokenizer=tokenizer,
        )
    return aa_feature


def load_cached_features(property_name, representation, sequences, split, data_root='data'):
    """Load observed or test features from data/<property>/<property>_{chem,esm2}.npz.

    Rows are returned in the same order as `sequences`. Unlabeled peptides are not
    stored in these archives; call get_features for those.
    """
    if representation == 'CHEM':
        suffix = 'chem'
    elif representation == 'ESM2':
        suffix = 'esm2'
    else:
        raise ValueError(f'Unknown representation: {representation}')
    if split not in ('observed', 'test'):
        raise ValueError(f'Unknown split: {split}')

    path = os.path.join(data_root, property_name, f'{property_name}_{suffix}.npz')
    query = [str(seq) for seq in sequences]

    # Compute-and-cache fallback: the public repository ships no feature
    # archives; they are regenerated once from the property CSVs (see
    # baselines/prepare_data.py) and reused afterwards.
    existing = {}
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        existing = {k: old[k] for k in old.files}
        old.close()
    if f'{split}_aa' not in existing:
        logging.info('No cached %s %s features at %s; computing and caching now',
                     split, representation, path)
        computed = np.asarray(get_features(query, representation), dtype=np.float32)
        existing[f'{split}_aa'] = np.array(query)
        existing[f'{split}_{suffix}'] = computed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temporary = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(temporary, **existing)
        os.replace(temporary, path)
        return computed

    cached_aa = [str(seq) for seq in existing[f'{split}_aa'].tolist()]
    features = existing[f'{split}_{suffix}']
    if query == cached_aa:
        logging.info(f'Loaded {split} {representation} features {features.shape} from {path}')
        return features

    index = {}
    for i, seq in enumerate(cached_aa):
        index.setdefault(seq, i)
    missing = [seq for seq in query if seq not in index]
    if missing:
        preview = ', '.join(missing[:3])
        raise KeyError(
            f'{len(missing)} {split} sequences are absent from {path} (for example: {preview})'
        )
    aligned = features[[index[seq] for seq in query]]
    logging.info(f'Loaded {split} {representation} features {aligned.shape} from {path}')
    return aligned


def featurize_peptides(property_name, representation, peptides, data_root='data'):
    """Feature rows for an arbitrary peptide list.

    Peptides present in the observed or test cache are read from that archive.
    The rest are computed with get_features.
    """
    peptides = [str(seq) for seq in peptides]
    if len(peptides) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if representation == 'CHEM':
        suffix = 'chem'
    elif representation == 'ESM2':
        suffix = 'esm2'
    else:
        raise ValueError(f'Unknown representation: {representation}')

    lookup = {}
    path = os.path.join(data_root, property_name, f'{property_name}_{suffix}.npz')
    if os.path.exists(path):
        archive = np.load(path, allow_pickle=True)
        for split in ('observed', 'test'):
            if f'{split}_aa' not in archive.files:
                continue
            for seq, row in zip(archive[f'{split}_aa'].tolist(), archive[f'{split}_{suffix}']):
                lookup.setdefault(str(seq), np.asarray(row, dtype=np.float32))

    missing = [seq for seq in peptides if seq not in lookup]
    if missing:
        logging.info(f'Computing {representation} features for {len(missing)} peptides absent from {path}')
        computed = np.asarray(get_features(missing, representation), dtype=np.float32)
        for seq, row in zip(missing, computed):
            lookup[seq] = row

    return np.stack([lookup[seq] for seq in peptides]).astype(np.float32)


def random_feature_cache_path(property_name, representation, parent_years, mutation_rate, data_root='data', obs_dist_max=2, num_sequences=None):
    """Path of the cached features for one property's random unlabeled sets.

    num_sequences is part of the filename when the caller chooses a set size.
    The PULSE-matched sets keep the original filename.
    """
    from utils.pulse_support import parent_year_tag

    if representation == 'CHEM':
        suffix = 'chem'
    elif representation == 'ESM2':
        suffix = 'esm2'
    else:
        raise ValueError(f'Unknown representation: {representation}')
    tag = parent_year_tag(parent_years)
    filename = f'{property_name}_rand_{suffix}_parents_{tag}_mu_{mutation_rate}_dist_{obs_dist_max}'
    if num_sequences is not None:
        filename += f'_n_{int(num_sequences)}'
    filename += '.npz'
    return os.path.join(data_root, property_name, filename)


def _features_for_sequences(sequences, representation):
    if representation == 'CHEM':
        return np.asarray(get_features(sequences, representation), dtype=np.float32)
    if representation != 'ESM2':
        raise ValueError(f'Unknown representation: {representation}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError(
            'ESM2 features for random peptides require a GPU. '
            'Submit the job with a GPU so the cache can be written; do not run it on the login node.'
        )
    model_name = 'facebook/esm2_t30_150M_UR50D'
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    embedded = esm2_get_embedding(
        sequences,
        device,
        batch_size=32,
        model=model,
        tokenizer=tokenizer,
    )
    return np.asarray(embedded, dtype=np.float32)


def load_random_features(property_name, representation, unlabeled_aa, parent_years, mutation_rate, data_root='data', obs_dist_max=2, num_sequences=None):
    """Feature matrices for the random unlabeled sets, loaded from cache when present.

    The archive stores one amino-acid list and one feature matrix per seed.
    A later run with the same parent years reuses it instead of embedding again.
    """
    if representation == 'CHEM':
        suffix = 'chem'
    elif representation == 'ESM2':
        suffix = 'esm2'
    else:
        raise ValueError(f'Unknown representation: {representation}')

    path = random_feature_cache_path(
        property_name, representation, parent_years, mutation_rate,
        data_root=data_root, obs_dist_max=obs_dist_max, num_sequences=num_sequences,
    )
    seeds = sorted(unlabeled_aa)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lock_path = path + '.lock'
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _cached_random_features(path, seeds, unlabeled_aa, suffix)
        missing = [seed for seed in seeds if seed not in cached]
        if missing:
            flat = [seq for seed in missing for seq in unlabeled_aa[seed]]
            logging.info(
                'Computing %s features for %d random peptides across seeds %s',
                representation,
                len(flat),
                ', '.join(str(seed) for seed in missing),
            )
            computed = _features_for_sequences(flat, representation)
            offset = 0
            for seed in missing:
                count = len(unlabeled_aa[seed])
                cached[seed] = computed[offset:offset + count]
                offset += count
            payload = {}
            for seed in seeds:
                payload[f'aa_{seed}'] = np.array([str(seq) for seq in unlabeled_aa[seed]])
                payload[f'{suffix}_{seed}'] = cached[seed]
            temporary = f'{path}.{os.getpid()}.tmp.npz'
            np.savez(temporary, **payload)
            os.replace(temporary, path)
            logging.info('Saved random %s features to %s', representation, path)
        else:
            logging.info('Loaded random %s features for %d seeds from %s', representation, len(seeds), path)
    return {seed: cached[seed] for seed in seeds}


def _cached_random_features(path, seeds, unlabeled_aa, suffix):
    """Return cached matrices whose stored sequences still match this run."""
    cached = {}
    if not os.path.exists(path):
        return cached
    archive = np.load(path, allow_pickle=True)
    try:
        for seed in seeds:
            aa_key = f'aa_{seed}'
            feature_key = f'{suffix}_{seed}'
            if aa_key not in archive.files or feature_key not in archive.files:
                continue
            stored = [str(seq) for seq in archive[aa_key].tolist()]
            if stored == [str(seq) for seq in unlabeled_aa[seed]]:
                cached[seed] = np.asarray(archive[feature_key], dtype=np.float32)
    finally:
        archive.close()
    return cached