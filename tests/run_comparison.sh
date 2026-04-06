#!/usr/bin/env bash
# Run from repo root:
#   bash tests/run_comparison.sh
#
# This script:
# 1. Runs the original EquiProp implementation and dumps states/gradients
# 2. Runs the new EquiProp implementation and dumps states/gradients
# 3. Compares the dumps
#
# Options (set via environment variables):
#   BATCHES=5       Number of training batches (default: 2)
#   DUMP_EVERY=1    Dump frequency (default: 1)
#   MODE=synchronous  Update mode (default: asynchronous)
#   VERBOSE=1       Enable verbose comparison output (default: 1)
#   MANUAL_GRADS=1  Use manual gradients in new impl (default: 1)

set -e

BATCHES=${BATCHES:-2}
DUMP_EVERY=${DUMP_EVERY:-1}
MODE=${MODE:-asynchronous}
VERBOSE=${VERBOSE:-1}
MANUAL_GRADS=${MANUAL_GRADS:-1}

ORIG_DUMP_DIR="/data/me22b227/dumps/original"
NEW_DUMP_DIR="/data/me22b227/dumps/new"

echo "=============================================="
echo "  EquiProp Implementation Comparison Test"
echo "=============================================="
echo "  Batches:      $BATCHES"
echo "  Dump every:   $DUMP_EVERY"
echo "  Mode:         $MODE"
echo "  Manual grads: $MANUAL_GRADS"
echo "=============================================="

# Clean previous dumps
rm -rf "$ORIG_DUMP_DIR" "$NEW_DUMP_DIR"
mkdir -p /data/me22b227/dumps

# Step 1: Run original implementation
echo ""
echo ">>> Step 1/3: Running ORIGINAL implementation..."
echo ""
python tests/debug_dump_original.py \
    --batches "$BATCHES" \
    --dump_every "$DUMP_EVERY" \
    --dump_dir "$ORIG_DUMP_DIR" \
    --save_initial_weights "/data/me22b227/dumps/initial_weights.pt" \
    --save_inputs \
    --inputs_file "/data/me22b227/dumps/shared_inputs.pt" \
    --mode "$MODE" \
    --epochs 1

# Step 2: Run new implementation (loading same inputs)
echo ""
echo ">>> Step 2/3: Running NEW implementation..."
echo ""

EXTRA_ARGS=""
if [ "$MANUAL_GRADS" = "1" ]; then
    EXTRA_ARGS="--manual_grads"
fi

python tests/debug_dump_new.py \
    --batches "$BATCHES" \
    --dump_every "$DUMP_EVERY" \
    --dump_dir "$NEW_DUMP_DIR" \
    --load_inputs \
    --inputs_file "/data/me22b227/dumps/shared_inputs.pt" \
    --mode "$MODE" \
    --epochs 1 \
    $EXTRA_ARGS

# Step 3: Compare
echo ""
echo ">>> Step 3/3: Comparing outputs..."
echo ""

VERBOSE_FLAG=""
if [ "$VERBOSE" = "1" ]; then
    VERBOSE_FLAG="--verbose"
fi

python tests/compare_dumps.py \
    --orig_dir "$ORIG_DUMP_DIR" \
    --new_dir "$NEW_DUMP_DIR" \
    $VERBOSE_FLAG

echo ""
echo "Done! Dumps are in $ORIG_DUMP_DIR and $NEW_DUMP_DIR"
