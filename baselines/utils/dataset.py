import math

import torch
from torch.distributions import Multinomial
from torch.utils.data import Dataset

from utils.utils_func import base_letter_to_idx, nuc_idx_to_aa_idx

# Amino acid to integer mapping
aa_to_int = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}



class SequenceDataset(Dataset):
    '''
    SequenceDataset class
    Input:
        alphabet (list): list of amino acid alphabet
        observed_dict (dict): dictionary of observed sequences 
                key: amino acid sequence, 
                value: tensor of shape (no_nuc, aa_length*3+1), 
                where no_nuc is the number of nucleotide sequences associated with the amino acid sequence 
                and the last column is the count of the nucleotide sequences
        p_aa_mutation (float): probability of amino acid mutation
        device (torch.device): device
    '''
    def __init__(self, alphabet, observed_dict, p_aa_mutation, device):
        self.device = device
        self.observed_dict = observed_dict
        self.p_aa_mutation = p_aa_mutation
        self.size_obs_aa = len(observed_dict)
        self.seq_len = len(list(observed_dict.keys())[0])

    def __len__(self):
        '''
        Return the size of the dataset
        '''
        return self.size

    def __getitem__(self, indices):
        # Deprecated: use vectorized batch sampling instead
        raise NotImplementedError("Use sample_batch for fast batch sampling.")

def unique_batch_loader(observed_dict, p_aa_mutation, batch_size, device, cal_weight=False):
    '''
    Batch loader that samples amino acid sequences and returns flags for observed vs unobserved
    
    Input:
        observed_dict (dict): dictionary with amino acid sequences as keys (tuples)
                             and nucleotide sequences as values (tensors)
        p_aa_mutation (float): probability of amino acid mutation
        batch_size (int): batch size
        device (torch.device): device
        cal_weight (bool): whether to calculate importance sampling weights
    Output:                 
        mutated (tensor size (batch_size, aa_length)): mutated amino acid sequences
        ims_w_weight (tensor size (batch_size, 1)): importance sampling weight
        flag (tensor size (batch_size, 1)): flag (1 if observed, 0 if unobserved)
        random_wt_indices (tensor): indices of selected wild-type sequences
        random_n_mutation (tensor): number of mutations per sequence
    '''
    # Get mutated sequences and weights
    mutated, ims_w_weight, random_wt_indices, random_n_mutation = sample_batch(
        observed_dict, p_aa_mutation, batch_size, device, cal_weight=cal_weight
    )
    
    # Define observed amino acid sequences set
    observed_aa_set = set(observed_dict.keys())
    
    # Check if each mutated sequence is in the observed set
    flag = torch.tensor(
        [1 if tuple(seq.tolist()) in observed_aa_set else 0 for seq in mutated],
        dtype=torch.long, device=device
    ).unsqueeze(-1)
    
    return mutated, ims_w_weight, flag, random_wt_indices, random_n_mutation


def sample_batch(observed_dict, p_aa_mutation, batch_size, device, cal_weight=False):
    '''
    Sample batch with amino acid mutations from observed sequences
    
    Input:
        observed_dict (dict): dictionary with amino acid sequences as keys (tuples of integers)
                             and nucleotide sequences as values (tensors)
        p_aa_mutation (float): probability of amino acid mutation
        batch_size (int): batch size
        device (torch.device): device
        cal_weight (bool): whether to calculate importance sampling weights
    Output:     
        mutated (tensor size (batch_size, aa_length)): mutated amino acid sequences (integer encoding)
        ims_w_weight (tensor size (batch_size, 1)): importance sampling weight (None if cal_weight=False)
        random_wt_indices (tensor): indices of selected wild-type sequences
        random_n_mutation (tensor): number of mutations per sequence
    '''
    # Get all observed amino acid sequences (keys) and their frequencies
    aa_keys = list(observed_dict.keys())
    aa_length = len(aa_keys[0])
    num_wt = len(aa_keys)
    
    # Random number of mutations per sequence (Binomial distribution)
    n_mutation_binom_dist = torch.distributions.Binomial(total_count=aa_length, probs=p_aa_mutation)
    random_n_mutation = n_mutation_binom_dist.sample((batch_size,)).to(torch.int64).to(device)
    
    # Random sample WT sequences uniformly (wild-type amino acid sequences)
    random_wt_indices = torch.randint(low=0, high=num_wt, size=(batch_size,), dtype=torch.int64, device=device)
    
    # Get the selected WT sequences as tensors
    wt_seqs = torch.stack([torch.tensor(aa_keys[idx], dtype=torch.long, device=device) 
                           for idx in random_wt_indices])
    
    # Clone for mutation (keeps original WT sequence if n_mut = 0)
    mutated = wt_seqs.clone()
    
    # Apply mutations (if n_mut = 0, sequence remains unchanged as WT)
    for b in range(batch_size):
        n_mut = int(random_n_mutation[b].item())
        if n_mut > 0:
            # Randomly select positions to mutate (without replacement)
            positions = torch.randperm(aa_length, device=device)[:n_mut]
            
            # For each position, uniformly sample from 19 other amino acids
            for pos in positions:
                current_aa = mutated[b, pos].item()
                # Create list of possible amino acids (0-19 excluding current)
                possible_aas = torch.tensor([i for i in range(20) if i != current_aa], 
                                           dtype=torch.long, device=device)
                # Uniformly sample one amino acid
                new_aa = possible_aas[torch.randint(19, (1,), device=device)]
                mutated[b, pos] = new_aa
        # Note: If n_mut = 0, we skip mutation and keep the WT sequence
    
    # Calculate importance sampling weights if requested
    if cal_weight:
        # Probability of choosing the WT (uniform sampling)
        prob_choose_wt = 1.0 / num_wt
        
        # Probability of the number of mutations (binomial PMF without combination term)
        prob_n_mutation = (p_aa_mutation ** random_n_mutation.float()) * \
                         ((1 - p_aa_mutation) ** (aa_length - random_n_mutation.float()))
        
        # Probability of the specific mutations (uniform over 19 choices per position)
        prob_mutations = torch.where(random_n_mutation > 0, 
                                     (1.0 / 19.0) ** random_n_mutation.float(),
                                     torch.ones_like(random_n_mutation, dtype=torch.float32))
        
        ims_w_weight = (prob_choose_wt * prob_n_mutation * prob_mutations).unsqueeze(-1)
    else:
        ims_w_weight = None
    
    return mutated, ims_w_weight, random_wt_indices, random_n_mutation
