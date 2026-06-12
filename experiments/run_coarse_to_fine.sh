#!/usr/bin/env bash
# Two-stage coarse-to-fine progressive training.
# Stage 1: coarse-only (steps 0 → stage_steps[0]-1), saves checkpoint.
# Stage 2: resume from coarse ckpt, continues mid+fine (steps stage_steps[0] → max_steps-1).
#
# Run from the gsplat/ repo root:
#   bash experiments/run_coarse_to_fine.sh
#
# Override via env vars:
#   DATA_DIR=data/garden GPU=1 bash experiments/run_coarse_to_fine.sh

set -euo pipefail

# ── configurable ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/grape}"
DATA_FACTOR="${DATA_FACTOR:-4}"
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-30000}"
STAGE_STEPS="${STAGE_STEPS:-5000 17000 27000}"
RESULT_DIR="${RESULT_DIR:-results/grape_coarse_to_fine}"
LOG_DIR="${RESULT_DIR}/logs"
COARSE_KEEP_RATIO="${COARSE_KEEP_RATIO:-0.6}"
SPAWN_SCORE_DELTA="${SPAWN_SCORE_DELTA:-0.3}"
# ─────────────────────────────────────────────────────────────────────────────

COARSE_STEP=$(echo "${STAGE_STEPS}" | awk '{print $1 - 1}')
COARSE_CKPT="${RESULT_DIR}/ckpts/ckpt_${COARSE_STEP}_rank0.pt"

mkdir -p "${LOG_DIR}"

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
    --spawn_score_delta       "${SPAWN_SCORE_DELTA}"
    --coarse_prune_keep_ratio "${COARSE_KEEP_RATIO}"
    --no-normalize_world_space
    --disable_viewer
    --result_dir              "${RESULT_DIR}"
)

echo "════════════════════════════════════════════════════════"
echo "  Coarse-to-fine training"
echo "  data        : ${DATA_DIR} (factor ${DATA_FACTOR})"
echo "  steps       : ${MAX_STEPS}  stages: ${STAGE_STEPS}"
echo "  coarse ckpt : ${COARSE_CKPT}"
echo "  keep_ratio  : ${COARSE_KEEP_RATIO}  freq_delta: ${SPAWN_SCORE_DELTA}"
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
        python -m examples.simple_trainer default \
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
FINAL_CKPT="${RESULT_DIR}/ckpts/ckpt_$((MAX_STEPS - 1))_rank0.pt"

if [ -f "${FINAL_CKPT}" ]; then
    echo "[Stage 2] Final checkpoint already exists — skipping."
else
    echo ""
    echo "[Stage 2] Resuming mid+fine training from step $((COARSE_STEP + 1))..."
    echo "  log: ${LOG_DIR}/stage2_mid_fine.log"
    echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${GPU}" \
        python -m examples.simple_trainer default \
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
