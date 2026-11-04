from typing import Any, List, Optional, Tuple

import torch

# idx -> nuc
base_idx_to_letter = {0:'A', 1:'C', 2:'G', 3:'T'}

# Reverse the dictionary to get letter to index
base_letter_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def nuc_seq_to_idx(seq: str) -> torch.Tensor:
    """
    Convert a nucleotide sequence (e.g., 'ACGT') into an index tensor.
    Args:
        seq (str): nucleotide sequence (only A, C, G, T)
    Returns:
        torch.Tensor of shape (len(seq),): Index tensor using 0-3 for A,C,G,T
    """
    onehot_index = torch.zeros((len(seq)), dtype=torch.int8)
    for i, base in enumerate(seq):
        idx = base_letter_to_idx[base]
        onehot_index[i] = idx
    return onehot_index

genetic_code = {
    'AAA':'K', 'AAC':'N', 'AAG':'K', 'AAT':'N',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AGA':'R', 'AGC':'S', 'AGG':'R', 'AGT':'S',
    'ATA':'I', 'ATC':'I', 'ATG':'M', 'ATT':'I',
    'CAA':'Q', 'CAC':'H', 'CAG':'Q', 'CAT':'H',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'GAA':'E', 'GAC':'D', 'GAG':'E', 'GAT':'D',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'TAA':'*', 'TAC':'Y', 'TAG':'*', 'TAT':'Y',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TGA':'*', 'TGC':'C', 'TGG':'W', 'TGT':'C',
    'TTA':'L', 'TTC':'F', 'TTG':'L', 'TTT':'F',
}

aa_list = [
    'A', 'C','D', 'E', 'F',
    'G', 'H', 'I','K', 'L', 
    'M','N', 'P','Q','R',
    'S','T', 'V', 'W', 'Y',
    '*', 
]
aa_to_int = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
aa_to_idx = {aa: idx for idx, aa in enumerate(aa_list)}

def check_stop_codons(nuc_idx: torch.Tensor) -> torch.Tensor:
    """
    Check for stop codons (*) in nucleotide sequences using vectorized operations.
    
    Args:
        nuc_idx: Tensor of shape (batch_size, length) with indices 0-3 for A,C,G,T
                 where length must be divisible by 3
    
    Returns:
        Tensor of shape (batch_size,) with True for sequences containing stop codons, False otherwise
    """
    if nuc_idx.ndim == 1:
        nuc_idx = nuc_idx.unsqueeze(0)
    
    B, L = nuc_idx.shape
    assert L % 3 == 0, "nucleotide length must be multiple of 3"
    
    # Reshape to (batch_size, num_codons, 3) for vectorized comparison
    codons = nuc_idx.view(B, -1, 3)
    
    # Define stop codon index combinations directly
    # TAA = [3, 0, 0], TAG = [3, 0, 2], TGA = [3, 2, 0]
    stop_codons = torch.tensor([
        [3, 0, 0],  # TAA
        [3, 0, 2],  # TAG  
        [3, 2, 0],  # TGA
    ], dtype=nuc_idx.dtype, device=nuc_idx.device)
    
    # Vectorized comparison: (batch_size, num_codons, 3) == (3, 3) -> (batch_size, num_codons, 3, 3)
    # Then check if all 3 positions match for any stop codon
    matches = (codons.unsqueeze(-2) == stop_codons.unsqueeze(0).unsqueeze(0)).all(dim=-1)
    
    # Check if any codon in any sequence matches any stop codon
    has_stop = matches.any(dim=-1).any(dim=-1)
    
    return has_stop

def nuc_idx_to_aa_idx(nuc_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert a batch of nucleotide index sequences to amino acid index sequences.
    Args:
        nuc_idx: Tensor of shape (batch_size, length) with indices 0-3 for A,C,G,T
        device: torch.device
    Returns:
        Tensor of shape (batch_size, length//3) with indices 0-20 for amino acids (incl. *)
    """
    if nuc_idx.ndim==1:
        nuc_idx = nuc_idx.unsqueeze(0)
    B, L = nuc_idx.shape
    assert L % 3 == 0, "nuc length must be multiple of 3"
    aa_len = L // 3
    aa_idx_batch = torch.zeros((B, aa_len), dtype=torch.int8, device=device)
    for b in range(B):
        for i in range(aa_len):
            codon_idx = nuc_idx[b, i*3:i*3+3]
            codon = ''.join(base_idx_to_letter[int(idx.item())] for idx in codon_idx)
            aa = genetic_code.get(codon, '*')
            aa_idx = aa_to_idx[aa]
            aa_idx_batch[b, i] = aa_idx
    return aa_idx_batch

aa_chem_property = {
    'G': (0.15, 0, 0),
    'L': (-3.32, 0, 0),
    'F': (-3.77, 0, 0),
    'A': (-1.57, 0, 0),
    'I': (-3.12, 0, 0),
    'V': (-2.34, 0, 0),
    'M': (-2.33, 0, 0),
    'C': (-1.08, 0, 0),
    'W': (-1.95, 1, 0),
    'T': (0.21, 1, 1),
    'S': (0.26, 1, 1),
    'Y': (-2.66, 1, 1),
    'P': (-3.09, 0, 0),
    'N': (1.91, 1, 1),
    'D': (1.38, 0, 2),
    'Q': (1.44, 1, 1),
    'E': (0.07, 0, 2),
    'K': (3.82, 1, 0),
    'R': (2.14, 3, 0),
    'H': (3.19, 2, 2)
}

prop_mat = torch.zeros((len(aa_list), 3), dtype=torch.float)
for aa, idx in aa_to_idx.items():
    prop_mat[idx] = torch.tensor(aa_chem_property.get(aa, (0.0, 0.0, 0.0)), dtype=torch.float)

codon_to_aa_idx = {}
for codon, aa in genetic_code.items():
    idx1 = base_letter_to_idx[codon[0]]
    idx2 = base_letter_to_idx[codon[1]] 
    idx3 = base_letter_to_idx[codon[2]]
    codon_idx = (idx1, idx2, idx3)
    codon_to_aa_idx[codon_idx] = aa_to_idx[aa]

def nuc_idx_to_aa_chem(nuc_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert nucleotide index tensor to amino acid chemical property tensor.
    Args:
        nuc_idx: Tensor of shape (batch, L) with indices 0-3 for A,C,G,T
        device: torch.device
    Returns:
        Tensor of shape (batch, L//3, 3) representing chemical properties of each amino acid
    """
    if nuc_idx.ndim == 1:
        nuc_idx = nuc_idx.unsqueeze(0)
    B, L = nuc_idx.shape
    assert L % 3 == 0, "nuc length must be multiple of 3"
    aa_len = L // 3
    codon_indices = nuc_idx.view(B, aa_len, 3)
    codon_lookup = torch.full((4, 4, 4), aa_to_idx['*'], dtype=torch.long, device=device)
    for (idx1, idx2, idx3), aa_idx in codon_to_aa_idx.items():
        codon_lookup[idx1, idx2, idx3] = aa_idx
    aa_indices = codon_lookup[codon_indices[:, :, 0].long(), 
                              codon_indices[:, :, 1].long(), 
                              codon_indices[:, :, 2].long()]
    aa_indices = mask_after_stop(aa_indices)
    prop_mat_device = prop_mat.to(device)
    aa_chem = torch.nn.functional.embedding(aa_indices, prop_mat_device)
    return aa_chem

def aa_idx_to_seq(aa_idx: torch.Tensor):
    """
    Convert amino acid index tensor(s) to amino acid sequence string(s).
    Args:
        aa_idx (Tensor):
            - Shape (length) for a single sequence using indices 0-20 for amino acids (incl. *)
            - Shape (batch, length) for a batch of sequences
    Returns:
        str or List[str]: Amino acid sequence(s)
    """
    idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}
    if aa_idx.dim() == 2:
        return [
            ''.join(idx_to_aa[int(aa_idx_i.item())] for aa_idx_i in one_hot_seq)
            for one_hot_seq in aa_idx
        ]
    elif aa_idx.dim() == 1:
        return ''.join(idx_to_aa[int(aa_idx_i.item())] for aa_idx_i in aa_idx)
    else:
        raise ValueError("Input tensor must be 1D or 2D (batch, length)")

def aa_seq_to_idx(seq: str) -> torch.Tensor:
    """
    Convert an amino acid sequence (e.g., 'GWATV') into an index tensor.
    Args:
        seq (str): Amino acid sequence
    Returns:
        torch.Tensor: Index tensor of shape (len(seq)) using indices 0-20 for amino acids (incl. *)
    """
    onehot_index = torch.zeros((len(seq)), dtype=torch.int8)
    for i, base in enumerate(seq):
        idx = aa_to_idx[base]
        onehot_index[i] = idx
    return onehot_index

def aa_idx_to_aa_chem(aa_idx: torch.Tensor, device: torch.device, prop_mat_: torch.Tensor=prop_mat) -> torch.Tensor:
    """
    Convert amino acid index tensor to chemical property tensor using prop_mat.
    Args:
        aa_idx: Tensor of shape (length) or (batch_size,length) with amino acid indices
        device: torch.device
        prop_mat_: Tensor of shape (21, num_properties_used_to_represent_aminoacid)
    Returns:
        Tensor of shape (length, 3) or (batch_size, length, 3)
    """
    aa_indices = aa_idx.long().to(device)
    aa_indices = mask_after_stop(aa_indices)
    prop_mat_ = prop_mat_.to(device)
    aa_chem = torch.nn.functional.embedding(aa_indices, prop_mat_)
    return aa_chem

def mask_after_stop(aa_indices: torch.Tensor, stop_idx: int = 20) -> torch.Tensor:
    """
    For each sequence in aa_indices, set all values after the first occurrence of stop_idx to stop_idx.
    Args:
        aa_indices: Tensor of shape (batch, length) or (length,) with values 0-20
        stop_idx: The index to treat as the stop codon (default 20)
    Returns:
        Tensor of same shape as aa_indices with masking applied
    """
    squeeze = False
    if aa_indices.dim() == 1:
        aa_indices = aa_indices.unsqueeze(0)
        squeeze = True
    is_stop = (aa_indices == stop_idx)
    stop_cumsum = is_stop.cumsum(dim=1)
    mask = stop_cumsum > 0
    result = aa_indices.clone()
    result[mask] = stop_idx
    if squeeze:
        result = result.squeeze(0)
    return result

def nuc_idx_to_seq(nuc_idx: torch.Tensor):
    """
    Convert nucleotide index tensor to nucleotide sequence string(s).
    Args:
        nuc_idx: Tensor of shape (batch, L) or (L,) with indices (0=A, 1=C, 2=G, 3=T)
    Returns:
        str or List[str]: Nucleotide sequence(s)
    """
    if nuc_idx.dim() == 2:
        return [
            ''.join(base_idx_to_letter[int(idx.item())] for idx in seq)
            for seq in nuc_idx
        ]
    elif nuc_idx.dim() == 1:
        return ''.join(base_idx_to_letter[int(idx.item())] for idx in nuc_idx)
    else:
        raise ValueError("Input tensor must be 1D or 2D (batch, length)")