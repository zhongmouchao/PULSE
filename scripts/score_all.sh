#!/usr/bin/env bash
# Score every saved PULSE checkpoint against every benchmark of its variant
# and append the rows to results/results.csv. Training is done once per
# variant (scripts/train_all.sh); this is the evaluate-many step.
#
# Usage:  bash scripts/score_all.sh [suffix]
#   default suffix "_EVE_Truncated" scores the truncated benchmark subsets
#   used for the paper's comparison table; pass "" (empty string) to score
#   the full, untruncated DMS libraries instead:
#       bash scripts/score_all.sh ""
set -euo pipefail
cd "$(dirname "$0")/.."

SUFFIX="${1:-_EVE_Truncated}"

score_variant () {
    local VARIANT="$1"; shift
    local BENCH
    for ASSAY in "$@"; do
        BENCH="data/SARS_${ASSAY}_${VARIANT}${SUFFIX}.csv"
        echo "=== ${VARIANT} <- ${BENCH} ==="
        python pulse/score_benchmark.py \
            --scan "${VARIANT}" \
            --benchmark "${BENCH}" \
            --save-results
    done
}

score_variant Wuhan_Hu_1_RBD   binding expression
score_variant Alpha_RBD        binding expression
score_variant Beta_RBD         binding expression
score_variant Eta_RBD          binding expression
score_variant Omicron_XBB15_RBD binding expression cell_entry
