#!/usr/bin/env bash
# Ablation: baseline vs. LoG frequency-adaptive spawn scoring + coarse pruning.
# Run from the gsplat/ repo root:
#   bash experiments/run_freq_spawn_ablation.sh
# Override defaults via env vars:
#   DATA_DIR=data/garden GPU=1 bash experiments/run_freq_spawn_ablation.sh
# Run a single experiment:
#   ONLY=freq_spawn_prune bash experiments/run_freq_spawn_ablation.sh

set -euo pipefail

# ── configurable ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/grape}"
DATA_FACTOR="${DATA_FACTOR:-4}"
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-30000}"
RESULT_ROOT="${RESULT_ROOT:-results/ablation_freq_spawn}"
LOG_DIR="${RESULT_ROOT}/logs"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
    --data_dir        "${DATA_DIR}"
    --data_factor     "${DATA_FACTOR}"
    --max_steps       "${MAX_STEPS}"
    --packed
    --progressive
    --stage_steps     5000 17000 27000
    --coarse_init_scale_mult  2.5
    --mid_spawn_scale_mult    0.6
    --fine_spawn_scale_mult   0.25
    --coarse_ssim_lambda      0.10
    --mid_ssim_lambda         0.20
    --fine_ssim_lambda        0.25
    --fine_absgrad
    --fine_grow_grad2d        0.0008
    --band_range_reg          0.01
    --no-normalize_world_space
    --disable_viewer
)

# Experiments: name -> extra args
# baseline          : no freq score, no coarse pruning
# freq_spawn        : LoG freq score only
# coarse_prune      : coarse pruning only
# freq_spawn_prune  : LoG freq score + coarse pruning (full method)
declare -A EXPERIMENTS
EXPERIMENTS["baseline"]="--spawn_score_delta 0.0 --coarse_prune_keep_ratio 1.0"
EXPERIMENTS["freq_spawn"]="--spawn_score_delta 0.3 --coarse_prune_keep_ratio 1.0"
EXPERIMENTS["coarse_prune"]="--spawn_score_delta 0.0 --coarse_prune_keep_ratio 0.6"
EXPERIMENTS["freq_spawn_prune"]="--spawn_score_delta 0.3 --coarse_prune_keep_ratio 0.6"

EXPERIMENT_ORDER="baseline freq_spawn coarse_prune freq_spawn_prune"

run_experiment() {
    local name="$1"
    local extra_args="$2"
    local result_dir="${RESULT_ROOT}/${name}"
    local log_file="${LOG_DIR}/${name}.log"

    if [ -f "${result_dir}/ckpts/ckpt_$((MAX_STEPS - 1))_rank0.pt" ]; then
        echo "[SKIP] ${name}: checkpoint already exists."
        return
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Starting: ${name}"
    echo "  result_dir: ${result_dir}"
    echo "  log: ${log_file}"
    echo "  extra: ${extra_args}"
    echo "════════════════════════════════════════════════════════"
    echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # shellcheck disable=SC2086
    MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${GPU}" \
        python -m examples.simple_trainer default \
        --result_dir "${result_dir}" \
        "${COMMON_ARGS[@]}" \
        ${extra_args} \
        2>&1 | tee "${log_file}"

    echo ""
    echo "  finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "Ablation: freq-adaptive spawn scoring + coarse pruning"
echo "  data    : ${DATA_DIR} (factor ${DATA_FACTOR})"
echo "  steps   : ${MAX_STEPS}"
echo "  GPU     : ${GPU}"
echo "  results : ${RESULT_ROOT}"
echo ""

ONLY="${ONLY:-}"
for name in ${EXPERIMENT_ORDER}; do
    if [ -n "${ONLY}" ] && [ "${name}" != "${ONLY}" ]; then
        echo "[SKIP] ${name}: not in ONLY=${ONLY}"
        continue
    fi
    run_experiment "${name}" "${EXPERIMENTS[$name]}"
done

echo ""
echo "All experiments done."
echo "Results under: ${RESULT_ROOT}/"
echo ""
echo "Quick PSNR summary (last eval per run):"
for name in ${EXPERIMENT_ORDER}; do
    log="${LOG_DIR}/${name}.log"
    if [ -f "${log}" ]; then
        last=$(grep -oP 'PSNR: \K[0-9.]+' "${log}" | tail -1)
        ssim=$(grep -oP 'SSIM: \K[0-9.]+' "${log}" | tail -1)
        lpips=$(grep -oP 'LPIPS: \K[0-9.]+' "${log}" | tail -1)
        printf "  %-20s PSNR=%-7s SSIM=%-7s LPIPS=%s\n" \
            "${name}" "${last:-N/A}" "${ssim:-N/A}" "${lpips:-N/A}"
    fi
done
