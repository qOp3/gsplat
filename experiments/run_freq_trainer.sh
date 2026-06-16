#!/usr/bin/env bash
# Wavelet-guided coarse-to-fine training using freq_trainer.py.
#
# Prerequisites:
#   1. Generate wavelet frequency maps for the dataset (run once):
#        python image_processing/wavelet_freq_map.py \
#            --input  gsplat/data/grape/images \
#            --output frequency_maps/grape_wavelet \
#            --levels 4 --wavelet db2
#
#   2. Pass FREQ_MAP_DIR to this script (see defaults below).
#
# Run from the gsplat/ repo root:
#   bash experiments/run_freq_trainer.sh
#
# Override via env vars:
#   DATA_DIR=data/garden FREQ_MAP_DIR=../../frequency_maps/garden_wavelet GPU=1 \
#       bash experiments/run_freq_trainer.sh

set -euo pipefail

# ── configurable ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/grape}"
DATA_FACTOR="${DATA_FACTOR:-4}"
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-30000}"
STAGE_STEPS="${STAGE_STEPS:-5000 17000 27000}"
RESULT_DIR="${RESULT_DIR:-results/grape_freq_trainer}"
LOG_DIR="${RESULT_DIR}/logs"
COARSE_KEEP_RATIO="${COARSE_KEEP_RATIO:-0.6}"

# Path to pre-computed wavelet maps (stem_wl_L1..L4.png files)
# Set to "" to disable wavelet maps and fall back to LoG scoring.
FREQ_MAP_DIR="${FREQ_MAP_DIR:-../../frequency_maps/grape_wavelet}"

FREQ_SPAWN_WEIGHT="${FREQ_SPAWN_WEIGHT:-0.4}"
FREQ_RANGE_REG="${FREQ_RANGE_REG:-true}"
FREQ_RANGE_STRENGTH="${FREQ_RANGE_STRENGTH:-0.5}"
FREQ_PRUNE="${FREQ_PRUNE:-true}"
# ─────────────────────────────────────────────────────────────────────────────

COARSE_STEP=$(echo "${STAGE_STEPS}" | awk '{print $1 - 1}')
COARSE_CKPT="${RESULT_DIR}/ckpts/ckpt_${COARSE_STEP}_rank0.pt"
FINAL_CKPT="${RESULT_DIR}/ckpts/ckpt_$((MAX_STEPS - 1))_rank0.pt"

mkdir -p "${LOG_DIR}"

# Build optional freq-map args
FREQ_ARGS=()
if [ -n "${FREQ_MAP_DIR}" ]; then
    FREQ_ARGS+=(--freq_map_dir "${FREQ_MAP_DIR}")
fi
if [ "${FREQ_PRUNE}" = "true" ]; then
    FREQ_ARGS+=(--freq_prune)
fi
if [ "${FREQ_RANGE_REG}" = "true" ]; then
    FREQ_ARGS+=(--freq_range_reg)
fi

COMMON_ARGS=(
    --data_dir                "${DATA_DIR}"
    --data_factor             "${DATA_FACTOR}"
    --max_steps               "${MAX_STEPS}"
    --packed
    --progressive
    --stage_steps             ${STAGE_STEPS}
    --coarse_init_scale_mult  2.5
    --mid_spawn_scale_mult    0.6
    --fine_spawn_scale_mult   0.25
    --coarse_ssim_lambda      0.10
    --mid_ssim_lambda         0.20
    --fine_ssim_lambda        0.25
    --fine_absgrad
    --fine_grow_grad2d        0.0008
    --band_range_reg          0.01
    --coarse_prune_keep_ratio "${COARSE_KEEP_RATIO}"
    --freq_spawn_weight       "${FREQ_SPAWN_WEIGHT}"
    --freq_range_strength     "${FREQ_RANGE_STRENGTH}"
    --no-normalize_world_space
    --disable_viewer
    --result_dir              "${RESULT_DIR}"
    "${FREQ_ARGS[@]}"
)

echo "════════════════════════════════════════════════════════"
echo "  Freq-trainer  (wavelet-guided coarse-to-fine)"
echo "  data        : ${DATA_DIR} (factor ${DATA_FACTOR})"
echo "  steps       : ${MAX_STEPS}  stages: ${STAGE_STEPS}"
echo "  freq maps   : ${FREQ_MAP_DIR:-<disabled, using LoG>}"
echo "  coarse ckpt : ${COARSE_CKPT}"
echo "  keep_ratio  : ${COARSE_KEEP_RATIO}  spawn_w: ${FREQ_SPAWN_WEIGHT}"
echo "  range_reg   : ${FREQ_RANGE_REG}  strength: ${FREQ_RANGE_STRENGTH}"
echo "  GPU         : ${GPU}  result: ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Stage 1: coarse only ─────────────────────────────────────────────────────
if [ -f "${COARSE_CKPT}" ]; then
    echo "[Stage 1] Coarse checkpoint already exists — skipping."
else
    echo "[Stage 1] Training coarse band (0 → ${COARSE_STEP})..."
    echo "  log: ${LOG_DIR}/stage1_coarse.log"
    echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${GPU}" \
        python -m examples.freq_trainer default \
        "${COMMON_ARGS[@]}" \
        --coarse_only \
        2>&1 | tee "${LOG_DIR}/stage1_coarse.log"

    echo ""
    echo "[Stage 1] Done at $(date '+%Y-%m-%d %H:%M:%S')"

    if [ ! -f "${COARSE_CKPT}" ]; then
        echo "[ERROR] Expected coarse checkpoint not found: ${COARSE_CKPT}"
        exit 1
    fi
fi

# ── Stage 2: mid + fine ──────────────────────────────────────────────────────
if [ -f "${FINAL_CKPT}" ]; then
    echo "[Stage 2] Final checkpoint already exists — skipping."
else
    echo ""
    echo "[Stage 2] Resuming mid+fine training from step $((COARSE_STEP + 1))..."
    echo "  log: ${LOG_DIR}/stage2_mid_fine.log"
    echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${GPU}" \
        python -m examples.freq_trainer default \
        "${COMMON_ARGS[@]}" \
        --resume_ckpt "${COARSE_CKPT}" \
        2>&1 | tee "${LOG_DIR}/stage2_mid_fine.log"

    echo ""
    echo "[Stage 2] Done at $(date '+%Y-%m-%d %H:%M:%S')"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "Training complete. Results: ${RESULT_DIR}"
echo ""
echo "Metrics (final eval):"
for stage_log in stage1_coarse stage2_mid_fine; do
    log="${LOG_DIR}/${stage_log}.log"
    [ -f "${log}" ] || continue
    psnr=$(grep -oP 'PSNR: \K[0-9.]+' "${log}" | tail -1)
    ssim=$(grep -oP 'SSIM: \K[0-9.]+' "${log}" | tail -1)
    lpips=$(grep -oP 'LPIPS: \K[0-9.]+' "${log}" | tail -1)
    printf "  %-22s PSNR=%-7s SSIM=%-7s LPIPS=%s\n" \
        "${stage_log}" "${psnr:-N/A}" "${ssim:-N/A}" "${lpips:-N/A}"
done
