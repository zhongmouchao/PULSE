#!/usr/bin/env python3
"""prepare_data.py — build the baselines' internal data layout from the
repository's shared data folder.

The public repository ships no sequence data. After downloading the five
training files and eleven benchmark files into ../data/ (see ../data/DATA.md),
run this script once from the baselines/ directory:

    python prepare_data.py                # EVE-truncated benchmarks (paper table)
    python prepare_data.py --suffix ""    # full, untruncated DMS libraries

The suffix selects which benchmark set is converted; re-running with a
different suffix overwrites the internal test CSVs.

For every baseline property it writes, under data/<property>/:

  <property>_train.csv            nucleotide/aa sequences + surveillance counts,
                                  converted from ../data/SARS_<variant>_RBD_training_counts.csv
  <property>_test[_<assay>].csv   benchmark table(s), converted from
                                  ../data/SARS_<assay>_<variant>_EVE_Truncated.csv
  WT.txt                          wild-type sequence, copied from ../wildtypes/

CHEM features (hydrophobicity, molar refractivity, volume — the same three
descriptors PULSE uses) are NOT precomputed here: each runner computes them on
first use and caches them as data/<property>/<property>_chem.npz.

The mapping between baseline properties and shared files:

  property              variant           assays
  Binding_Wuhan         Wuhan_Hu_1_RBD    binding
  Express_Wuhan         Wuhan_Hu_1_RBD    expression
  Binding_OmicronXBB15  Omicron_XBB15_RBD binding
  Express_Omicron       Omicron_XBB15_RBD expression
  cell_omicron          Omicron_XBB15_RBD cell_entry
  Alpha / Beta / Eta    <Variant>_RBD     binding + expression
"""
import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA = os.path.join(HERE, '..', 'data')
WILDTYPES = os.path.join(HERE, '..', 'wildtypes')

# property -> (variant, [(output test suffix, shared assay name)])
PROPERTIES = {
    'Binding_Wuhan':        ('Wuhan_Hu_1_RBD',   [('test', 'binding')]),
    'Express_Wuhan':        ('Wuhan_Hu_1_RBD',   [('test', 'expression')]),
    'Binding_OmicronXBB15': ('Omicron_XBB15_RBD', [('test', 'binding')]),
    'Express_Omicron':      ('Omicron_XBB15_RBD', [('test', 'expression')]),
    'cell_omicron':         ('Omicron_XBB15_RBD', [('test', 'cell_entry')]),
    'Alpha':                ('Alpha_RBD',  [('test_binding', 'binding'),
                                            ('test_expression', 'expression')]),
    'Beta':                 ('Beta_RBD',   [('test_binding', 'binding'),
                                            ('test_expression', 'expression')]),
    'Eta':                  ('Eta_RBD',    [('test_binding', 'binding'),
                                            ('test_expression', 'expression')]),
}

# shared training counts file -> baseline internal names
TRAIN_COLUMN_MAP = {'Motif Nucleotide': 'nuc_sequence', 'Motif Peptide': 'aa_sequence'}


def _is_time_col(name):
    """YYYY or YYYY-MM surveillance-period column."""
    parts = str(name).split('-')
    return 1 <= len(parts) <= 2 and all(p.isdigit() and len(p) in (2, 4) for p in parts)


def convert_training(variant, out_path):
    src = os.path.join(SHARED_DATA, f'SARS_{variant}_training_counts.csv')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found. Download the training files into ../data/ first '
            '(see ../data/DATA.md).')
    df = pd.read_csv(src)
    df = df.rename(columns={k: v for k, v in TRAIN_COLUMN_MAP.items() if k in df.columns})
    if 'nuc_sequence' not in df.columns or 'aa_sequence' not in df.columns:
        raise KeyError(f'{src}: expected nucleotide/peptide columns '
                       f'(Motif Nucleotide/Motif Peptide); got {list(df.columns)[:4]}...')
    count_cols = [c for c in df.columns if _is_time_col(c)]
    if 'unknown' in df.columns:
        count_cols.append('unknown')
    if not count_cols:
        raise KeyError(f'{src}: no surveillance count columns found')
    df['count'] = df[count_cols].sum(axis=1).astype(int)
    df.to_csv(out_path, index=False)
    logging.info('  %-28s <- %s (%d rows)', os.path.basename(out_path),
                 os.path.basename(src), len(df))


def convert_benchmark(variant, assay, out_path, suffix):
    src = os.path.join(SHARED_DATA, f'SARS_{assay}_{variant}{suffix}.csv')
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found (see ../data/DATA.md).')
    df = pd.read_csv(src)
    rename = {}
    if 'mutated_sequence' in df.columns:
        rename['mutated_sequence'] = 'aa_sequence'
    if 'DMS_score' in df.columns:
        rename['DMS_score'] = 'freq'
    df = df.rename(columns=rename)
    keep = [c for c in ('mutation', 'aa_sequence', 'freq') if c in df.columns]
    if 'aa_sequence' not in keep or 'freq' not in keep:
        raise KeyError(f'{src}: expected mutated_sequence/DMS_score columns; '
                       f'got {list(df.columns)}')
    df[keep].to_csv(out_path, index=False)
    logging.info('  %-28s <- %s (%d rows)', os.path.basename(out_path),
                 os.path.basename(src), len(df))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--suffix', default='_EVE_Truncated',
                        help='benchmark filename suffix; "" selects the full, '
                             'untruncated DMS libraries (default: %(default)r)')
    args = parser.parse_args()
    done, skipped = 0, []
    for prop, (variant, tests) in PROPERTIES.items():
        out_dir = os.path.join(HERE, 'data', prop)
        os.makedirs(out_dir, exist_ok=True)
        try:
            logging.info('%s (variant %s):', prop, variant)
            convert_training(variant, os.path.join(out_dir, f'{prop}_train.csv'))
            for test_suffix, assay in tests:
                convert_benchmark(variant, assay,
                                  os.path.join(out_dir, f'{prop}_{test_suffix}.csv'),
                                  args.suffix)
        except FileNotFoundError as e:
            logging.warning('  SKIPPED: %s', e)
            skipped.append(prop)
            continue
        wt_src = os.path.join(WILDTYPES, f'SARS_{variant}_WT.txt')
        shutil.copy2(wt_src, os.path.join(out_dir, 'WT.txt'))
        done += 1
    logging.info('Done: %d properties prepared, %d skipped (%s). '
                 'CHEM features are computed and cached on first use.',
                 done, len(skipped), ', '.join(skipped) if skipped else 'none')


if __name__ == '__main__':
    main()
