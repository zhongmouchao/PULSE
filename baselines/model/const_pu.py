import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from utils.dataset import SequenceDataset

# Set up logging
logging.basicConfig(level=logging.INFO)

class ConstPU(nn.Module):
    def __init__(self, classifier_model: nn.Module, 
                    device: torch.device,
                    q: float,
                    exact_opt: bool = True,
                    dataset: Optional[SequenceDataset] = None,
                    nuc_representation_fn: Optional[Callable] = None,
):
        '''
        ConstPU model
        Input:
            classifier_model (nn.Module): classifier model
            q: float, class prior
            device: device (torch.device)
            dataset: dataset (SequenceDataset) (optional) #TODO for importance sampling with SGD
            nuc_representation_fn (Callable): representation function for nucleotide (optional) # TODO for for importance sampling with SGD

        '''
        super().__init__()
        self.classifier = classifier_model.to(device) 
        self.q = q
        self.device = device
        self.exact_opt = exact_opt # whether to use exact optimization or importance sampling with SGD
        self.exact_opt = exact_opt
        if not self.exact_opt:
            self.nuc_representation_fn = nuc_representation_fn
            self.dataset = dataset
        else:
            self.nuc_representation_fn = None
            self.dataset = None

    def forward(self, x:torch.Tensor):
        '''
        Forward pass of ConstPU model -- classifier prediction
        Input:
            x (torch.Tensor): tensor of shape (batch_size, length) the one-hot encoding of the sequence (amino acid using indices 0-20)
        Output:
            probability of the sequences being active (tensor): (batch_size, 1)
        '''
        input_rep_flat = x.view(x.size(0), -1)  # shape: (batch, length*3)
        out_classify = self.classifier(input_rep_flat) #compute the probability of the sequences being active
        return out_classify

    def log_likelihood(self, x_positive: torch.Tensor, x_unlabeled: torch.Tensor, exact_opt: bool = True):
        """Sum of log likelihood over an already split positive and unlabeled batch.

        Positive sequences contribute log(q * s(x)). Unlabeled sequences contribute
        log(1 - q * s(x)), with s(x) the classifier probability.
        """
        if not exact_opt:
            raise ValueError("Importance sampling with SGD is not implemented for proteinPU")
        classifier_labeled = torch.clamp(self.forward(x_positive), min=1e-8, max=1.0)
        classifier_unlabeled = torch.clamp(self.forward(x_unlabeled), min=1e-8, max=1.0)
        pos_term = torch.log(self.q * classifier_labeled)
        neg_term = torch.log(torch.clamp(1 - self.q * classifier_unlabeled, min=1e-8))
        return pos_term.sum() + neg_term.sum()

    def sgd_log_likelihood(self, x_positive, x_unlabeled, neighborhood_size):
        """Mean labeled log term plus the weighted mean unlabeled log term.

        The labeled piece is the exact average of log(q s(x)) over the observed
        peptides in this step. Each unlabeled peptide is weighted by
        1 / |N(parent)|, the reciprocal of its parent's unobserved neighborhood.

        Inputs:
            x_positive (Tensor): (n_labeled, feature_dim) observed peptides.
            x_unlabeled (Tensor): (m, feature_dim) freshly drawn unobserved peptides.
            neighborhood_size (Tensor): (m,) unobserved neighborhood size of each draw's parent.
        Output:
            total (Tensor): sum of the two averages.
            labeled_mean (Tensor): observed average, detached.
            unlabeled_mean (Tensor): weighted unlabeled average, detached.
        """
        classifier_labeled = torch.clamp(self.forward(x_positive), min=1e-8, max=1.0)
        classifier_unlabeled = torch.clamp(self.forward(x_unlabeled), min=1e-8, max=1.0)
        labeled = torch.log(self.q * classifier_labeled).mean()
        log_unlabeled = torch.log(torch.clamp(1 - self.q * classifier_unlabeled, min=1e-8)).reshape(-1)
        weights = self.importance_sampling_weight(neighborhood_size).reshape(-1)
        weighted = torch.sum(weights * log_unlabeled.double()) / torch.sum(weights)
        unlabeled = weighted.to(dtype=labeled.dtype)
        return labeled + unlabeled, labeled.detach(), unlabeled.detach()

    def uniform_sgd_log_likelihood(self, x_positive, x_unlabeled):
        """Mean of log(q s(x)) plus the mean of log(1 - q s(u)).

        x_unlabeled is a uniform sample from the unobserved neighborhood, so the
        second mean estimates the average over that neighborhood. The acceptance
        step already corrected for peptides with several parents, and this term
        applies no further weight.

        Output:
            total (Tensor): sum of the two averages.
            labeled_mean (Tensor): observed average, detached.
            unlabeled_mean (Tensor): unlabeled average, detached.
        """
        classifier_labeled = torch.clamp(self.forward(x_positive), min=1e-8, max=1.0)
        classifier_unlabeled = torch.clamp(self.forward(x_unlabeled), min=1e-8, max=1.0)
        labeled = torch.log(self.q * classifier_labeled).mean()
        unlabeled = torch.log(torch.clamp(1 - self.q * classifier_unlabeled, min=1e-8)).mean()
        return labeled + unlabeled, labeled.detach(), unlabeled.detach()

    def importance_sampling_weight(self, neighborhood_size):
        """Importance weight 1 / |N(parent)| for each unlabeled peptide.

        Inputs:
            neighborhood_size: number of unobserved peptides at distance
            1 or 2 from the parent of each draw.
        Output:
            Tensor: one positive weight per draw, on this model's device.
        """
        size = torch.as_tensor(neighborhood_size, dtype=torch.float64, device=self.device)
        if torch.any(size <= 0):
            raise ValueError('neighborhood size must be positive')
        return 1.0 / size
