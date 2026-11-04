#!/usr/bin/env bash
# Train PULSE on all five SARS-CoV-2 RBD variants with the paper configuration.
# One training run per variant; the checkpoint is saved under models/.
# Usage:  bash scripts/train_all.sh [sweep.yaml]   (default: sweeps/paper.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP="${1:-sweeps/paper.yaml}"

for TASK in SARS_Wuhan_Hu_1_RBD SARS_Alpha_RBD SARS_Beta_RBD SARS_Eta_RBD SARS_Omicron_XBB15_RBD; do
    echo "=== $TASK (sweep: $SWEEP) ==="
    python pulse/train_sgd.py --task "$TASK" --sweep "$SWEEP"
done
