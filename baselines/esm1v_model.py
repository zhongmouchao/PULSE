#!/usr/bin/env python
"""
ESM-1v model script for computing pseudo-log-likelihoods of protein sequences.
This script is converted from esm1v_model.ipynb for SLURM job submission.
"""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for SLURM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from scipy.stats import pearsonr, spearmanr


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Filled in main(); flushed on SIGTERM/SIGINT so preemption keeps progress.
_CHECKPOINT_STATE = None


def _atomic_write_csv(path, frame):
    """Write a CSV via a temp file so a kill mid-write cannot corrupt the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def checkpoint_paths(output_path):
    output_path = Path(output_path)
    partial = output_path.with_name(output_path.stem + '.partial.csv')
    meta = output_path.with_name(output_path.stem + '.partial.meta.json')
    return partial, meta


def save_checkpoint(partial_path, meta_path, completed_frame, meta):
    _atomic_write_csv(partial_path, completed_frame)
    _atomic_write_json(meta_path, meta)
    logging.info(
        'Checkpoint saved: %d / %d sequences -> %s',
        len(completed_frame),
        meta['n_total'],
        partial_path,
    )


def load_checkpoint(partial_path, meta_path, df_test, wild_type_seq, test_csv, model_name):
    """Return (start_index, completed_frame) or (0, empty frame) if no valid checkpoint."""
    empty = pd.DataFrame(
        columns=[
            'aa_sequence',
            'esm1v_log_likelihood',
            'esm1v_log_likelihood_norm',
            'esm1v_log_likelihood_diff_from_wt',
        ]
    )
    if not partial_path.exists() or not meta_path.exists():
        return 0, empty

    with open(meta_path) as handle:
        meta = json.load(handle)

    expected = {
        'n_total': int(len(df_test)),
        'test_csv': str(Path(test_csv).resolve()),
        'wild_type_sequence': wild_type_seq,
        'model_name': model_name,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            logging.warning(
                'Checkpoint meta mismatch for %s (%r vs %r); starting from scratch.',
                key,
                meta.get(key),
                value,
            )
            return 0, empty

    completed = pd.read_csv(partial_path)
    n_done = len(completed)
    if n_done == 0:
        return 0, empty
    if n_done > len(df_test):
        logging.warning('Checkpoint longer than test set; starting from scratch.')
        return 0, empty

    expected_seqs = df_test['aa_sequence'].iloc[:n_done].tolist()
    if completed['aa_sequence'].tolist() != expected_seqs:
        logging.warning('Checkpoint sequences do not match the test CSV; starting from scratch.')
        return 0, empty

    logging.info('Resuming from checkpoint: %d / %d sequences already done.', n_done, len(df_test))
    return n_done, completed


def _handle_termination(signum, _frame):
    state = _CHECKPOINT_STATE
    if state is not None and state.get('dirty'):
        logging.warning('Caught signal %s; flushing checkpoint before exit.', signum)
        try:
            save_checkpoint(
                state['partial_path'],
                state['meta_path'],
                state['completed_frame'],
                state['meta'],
            )
        except Exception:
            logging.exception('Failed to flush checkpoint on signal %s', signum)
    sys.exit(128 + signum)


def load_esm1v_model(model_name="esm1v_t33_650M_UR90S_1", device=None):
    """
    Load ESM-1v model and alphabet.
    
    Args:
        model_name: Name of the ESM-1v model
        device: Device to load model on (None for auto-detect)
        
    Returns:
        model: ESM-1v model
        alphabet: ESM alphabet
        device: Device used
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    try:
        # Method 1: Try torch.hub (most reliable for ESM-1v)
        model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
    except Exception:
        try:
            # Method 2: Try the standard pretrained loader
            import esm
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        except (AttributeError, ImportError):
            # Method 3: Try direct import if available
            from esm import pretrained
            model, alphabet = pretrained.load_model_and_alphabet(model_name)
    
    model = model.to(device)
    model.eval()
    return model, alphabet, device


def compute_esm1v_log_likelihood(sequence, model, alphabet, device):
    """
    Compute pseudo-log-likelihood of a sequence using ESM-1v.
    
    Args:
        sequence: Amino acid sequence string
        model: ESM-1v model
        alphabet: ESM alphabet
        device: Device to run computation on
        
    Returns:
        pll: Total pseudo-log-likelihood
        pll_norm: Normalized pseudo-log-likelihood (per position)
    """
    batch_converter = alphabet.get_batch_converter()
    
    seq_name = "seq"
    data = [(seq_name, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Sequence length (exclude BOS=0 and EOS=last)
    L = batch_tokens.size(1) - 2
    
    pll = 0.0
    with torch.no_grad():
        for i in range(1, L+1):
            # Mask position i
            masked_tokens = batch_tokens.clone()
            masked_tokens[0, i] = alphabet.mask_idx
            
            # Forward
            results = model(masked_tokens)
            logits = results["logits"]  # shape: [1, L+2, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            true_token = batch_tokens[0, i]
            pll += log_probs[0, i, true_token].item()
    
    pll_norm = pll / L
    return pll, pll_norm

def load_wild_type_from_file(wt_file_path):
    """
    Load wild type sequence from a text file.
    
    Args:
        wt_file_path: Path to wild type text file
        
    Returns:
        wild_type_sequence: The wild type sequence
    """
    with open(wt_file_path, 'r') as f:
        content = f.read()
        wild_type_seq = content.split('\n')[0]
    logging.info(f"Loaded wild type sequence from file: {wt_file_path}")
    return wild_type_seq


def plot_roc_pr_curves(y_true, y_scores, output_dir, filename_prefix):
    """
    Plot ROC and Precision-Recall curves and save them.
    
    Args:
        y_true: True binary labels (1 for positive, 0 for negative)
        y_scores: Prediction scores
        output_dir: Directory to save plots
        filename_prefix: Prefix for output filenames
        
    Returns:
        roc_auc: AUC-ROC score
        avg_precision: Average precision score
    """
    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot Precision-Recall curve
    ax2.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    # Baseline is the proportion of positive examples
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline (AP = {baseline:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / f"{filename_prefix}_roc_pr.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"ROC and PR curves saved to: {plot_path}")
    
    return roc_auc, avg_precision


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute ESM-1v log-likelihoods for protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--property', type=str, required=True,
        help='Property/dataset name (e.g., fusion, binding, evasion, proteingym_A0, proteingym_PSAE)'
    )
    
    parser.add_argument(
        '--test_csv', type=str, required=True,
        help='Path to test CSV file'
    )
    
    parser.add_argument(
        '--train_csv', type=str, default=None,
        help='Path to training CSV file (for finding wild type). If not provided, will use wt_file.'
    )
    
    parser.add_argument(
        '--wt_file', type=str, default=None,
        help='Path to wild type text file. Used if train_csv is not provided.'
    )
    
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Path to output CSV file. If not provided, will be auto-generated.'
    )
    
    parser.add_argument(
        '--model_name', type=str, default='esm1v_t33_650M_UR90S_1',
        help='ESM-1v model name'
    )
    
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda, mps, cpu). Auto-detected if not specified'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='result',
        help='Output directory for plots'
    )

    parser.add_argument(
        '--checkpoint_every', type=int, default=10,
        help='Write a partial checkpoint every N newly scored sequences '
             '(also flushed on SIGTERM/SIGINT for preemption).'
    )
    
    args = parser.parse_args()
    
    # Convert empty strings to None for optional arguments
    if args.train_csv == '':
        args.train_csv = None
    if args.wt_file == '':
        args.wt_file = None
    if args.output_csv == '':
        args.output_csv = None
    if args.device == '':
        args.device = None
    
    return args
    


def main():
    """Main function."""
    global _CHECKPOINT_STATE
    args = parse_args()
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    logging.info(f'Using device: {device}')
    
    # Load test sequences
    logging.info(f"Loading test sequences from: {args.test_csv}")
    df_test = pd.read_csv(args.test_csv)
    label_columns = ['aa_sequence']
    if 'freq' in df_test.columns:
        label_columns.append('freq')
    if 'activity' in df_test.columns:
        label_columns.append('activity')
    df_test = df_test[label_columns].copy()
    df_test = df_test.reset_index(drop=True)
    logging.info(f"Loaded {len(df_test)} test sequences")
    
    # Find or load wild type
    if args.wt_file:
        logging.info(f"\nLoading wild type from file: {args.wt_file}")
        wild_type_seq = load_wild_type_from_file(args.wt_file)
        logging.info(f"Wild type sequence: {wild_type_seq}")
    else:
        raise ValueError("Either --train_csv or --wt_file must be provided")

    # Output / checkpoint paths (keep property folder casing, e.g. ANTI_Wuhan)
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = Path(
            f"data/{args.property}/{args.property}_test_esm1v_results.csv"
        )
    partial_path, meta_path = checkpoint_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx, completed = load_checkpoint(
        partial_path,
        meta_path,
        df_test,
        wild_type_seq,
        args.test_csv,
        args.model_name,
    )
    if start_idx >= len(df_test):
        logging.info('Checkpoint already covers the full test set; finalizing.')
    else:
        # Load ESM-1v model
        logging.info(f"\nLoading ESM-1v model: {args.model_name}")
        model, alphabet, device = load_esm1v_model(args.model_name, device)
        logging.info(f"Model loaded successfully on device: {device}")

        # Prefer WT scores from a matching checkpoint so resumed diffs stay consistent.
        wt_pll = None
        wt_pll_norm = None
        if meta_path.exists() and start_idx > 0:
            with open(meta_path) as handle:
                meta = json.load(handle)
            wt_pll = meta.get('wt_pll')
            wt_pll_norm = meta.get('wt_pll_norm')

        if wt_pll is None or wt_pll_norm is None:
            logging.info(f"\nComputing wild type log-likelihood...")
            wt_pll, wt_pll_norm = compute_esm1v_log_likelihood(
                wild_type_seq, model, alphabet, device
            )
        logging.info(
            f"Wild type log-likelihood: {wt_pll:.4f} (normalized: {wt_pll_norm:.4f})"
        )

        meta = {
            'n_total': int(len(df_test)),
            'test_csv': str(Path(args.test_csv).resolve()),
            'wild_type_sequence': wild_type_seq,
            'model_name': args.model_name,
            'wt_pll': float(wt_pll),
            'wt_pll_norm': float(wt_pll_norm),
        }
        _CHECKPOINT_STATE = {
            'partial_path': partial_path,
            'meta_path': meta_path,
            'completed_frame': completed,
            'meta': meta,
            'dirty': False,
        }
        signal.signal(signal.SIGTERM, _handle_termination)
        signal.signal(signal.SIGINT, _handle_termination)

        logging.info(
            "\nComputing log-likelihoods for test sequences "
            f"(starting at {start_idx + 1}/{len(df_test)})..."
        )
        seqs = completed['aa_sequence'].tolist() if len(completed) else []
        plls = completed['esm1v_log_likelihood'].tolist() if len(completed) else []
        plls_norm = (
            completed['esm1v_log_likelihood_norm'].tolist() if len(completed) else []
        )
        pll_diffs = (
            completed['esm1v_log_likelihood_diff_from_wt'].tolist()
            if len(completed)
            else []
        )

        def _frame_from_lists():
            return pd.DataFrame(
                {
                    'aa_sequence': seqs,
                    'esm1v_log_likelihood': plls,
                    'esm1v_log_likelihood_norm': plls_norm,
                    'esm1v_log_likelihood_diff_from_wt': pll_diffs,
                }
            )

        since_checkpoint = 0
        for idx in range(start_idx, len(df_test)):
            seq = df_test.at[idx, 'aa_sequence']
            pll, pll_norm = compute_esm1v_log_likelihood(seq, model, alphabet, device)
            pll_diff = pll - wt_pll
            seqs.append(seq)
            plls.append(pll)
            plls_norm.append(pll_norm)
            pll_diffs.append(pll_diff)
            completed = _frame_from_lists()
            _CHECKPOINT_STATE['completed_frame'] = completed
            _CHECKPOINT_STATE['dirty'] = True
            since_checkpoint += 1

            done = idx + 1
            if done % 10 == 0 or done == len(df_test):
                logging.info(f"Processed {done}/{len(df_test)} sequences...")

            if since_checkpoint >= args.checkpoint_every or done == len(df_test):
                save_checkpoint(partial_path, meta_path, completed, meta)
                _CHECKPOINT_STATE['dirty'] = False
                since_checkpoint = 0

    # Merge scores back onto the original test frame
    score_cols = [
        'esm1v_log_likelihood',
        'esm1v_log_likelihood_norm',
        'esm1v_log_likelihood_diff_from_wt',
    ]
    if len(completed) != len(df_test):
        raise RuntimeError(
            f'Expected {len(df_test)} scored sequences, found {len(completed)}'
        )
    if completed['aa_sequence'].tolist() != df_test['aa_sequence'].tolist():
        raise RuntimeError('Completed scores are out of order relative to the test CSV')

    for column in score_cols:
        df_test[column] = completed[column].to_numpy()
    df_test['wild_type_sequence'] = wild_type_seq

    # Save final results and remove partial checkpoint
    df_test.to_csv(output_path, index=False)
    logging.info(f"\nResults saved to: {output_path}")
    for path in (partial_path, meta_path):
        if path.exists():
            path.unlink()
            logging.info('Removed checkpoint file: %s', path)
    
    # Print summary statistics
    logging.info("\n=== Summary Statistics ===")
    if 'freq' in df_test.columns:
        logging.info(f"Frequency: {df_test['freq'].mean():.4f}")
    if 'activity' in df_test.columns:
        logging.info(
            f"Positive sequences (activity='+'): "
            f"{len(df_test[df_test['activity'] == '+'])}"
        )
        logging.info(
            f"Negative sequences (activity='-'): "
            f"{len(df_test[df_test['activity'] == '-'])}"
        )
    logging.info(f"\nESM-1v log-likelihood difference from WT:")
    logging.info(f"  Mean: {df_test['esm1v_log_likelihood_diff_from_wt'].mean():.4f}")
    logging.info(f"  Std: {df_test['esm1v_log_likelihood_diff_from_wt'].std():.4f}")
    logging.info(f"  Min: {df_test['esm1v_log_likelihood_diff_from_wt'].min():.4f}")
    logging.info(f"  Max: {df_test['esm1v_log_likelihood_diff_from_wt'].max():.4f}")
    
    # Compute and plot ROC / Spearman
    logging.info("\n=== Computing Classification Performance ===")
    y_scores = np.asarray(df_test['esm1v_log_likelihood_diff_from_wt'].values, dtype=float)
    if y_scores.size > 0 and np.all(y_scores == y_scores[0]):
        logging.info(
            'all test sequences received same prediction -- resampling predicted values uniformly from (0,1)'
        )
        y_scores = np.random.uniform(0.0, 1.0, size=y_scores.shape)
    if 'activity' in df_test.columns:
        y_true = []
        for value in df_test['activity'].tolist():
            if value in ('+', 1, '1') or value == 1.0:
                y_true.append(1)
            elif value in ('-', 0, '0') or value == 0.0:
                y_true.append(0)
            else:
                raise ValueError(f'Activity must be 0/1 or +/- , got {value!r}')
        y_true = np.asarray(y_true, dtype=int)
        filename_prefix = f"{args.property.lower()}_esm1v"
        plot_output_dir = Path(args.output_dir) / "zeroshot"
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        roc_auc, avg_precision = plot_roc_pr_curves(
            y_true, y_scores, plot_output_dir, filename_prefix
        )
        logging.info(f"AUC-ROC: {roc_auc:.4f}")
        logging.info(f"Average Precision: {avg_precision:.4f}")
    if 'freq' in df_test.columns:
        rho, p_rho = spearmanr(df_test['freq'], y_scores)
        logging.info(f"Spearman rho: {rho}, and p-value: {p_rho}")

    logging.info("\nDone!")


if __name__ == '__main__':
    main()

