#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for examples/deepseek/ptq.py
# Defaults to model "zai-org/GLM-4.5-Air" and runs with torchrun to ensure
# torch.distributed is initialized even on single GPU.
#
# Config via ENV (override when invoking):
#   MODEL_PATH        HF repo or local path (default: zai-org/GLM-4.5-Air)
#   OUTPUT_DIR        Output directory for amax & quant config (default: ./outputs/glm45_air_ptq)
#   QUANT_CFG         Name of quant config in modelopt.torch.quantization (REQUIRED)
#   BATCH_SIZE        Calibration batch size (default: 1)
#   CALIB_SIZE        Number of calibration samples (default: 32)
#   NPROC             Number of ranks (processes) per node (default: 1)
#   ENABLE_FP8_KVCACHE  1 to enable FP8 KV cache (default: 0)
#   ENABLE_WO_QUANT     1 to enable MLA WO quant (default: 0)
#   TRUST_REMOTE_CODE   1 to pass --trust_remote_code (default: 1)
#   CUDA_VISIBLE_DEVICES Comma list of GPUs (default: 0)
#
# Example:
#   QUANT_CFG=FP8_DEFAULT_CFG NPROC=2 bash examples/deepseek/run_ptq_glm45_air.sh
#   QUANT_CFG=FP4_DEFAULT_CFG ENABLE_WO_QUANT=1 bash examples/deepseek/run_ptq_glm45_air.sh


### modify 
# ^/site-packages/modelopt/torch/quantization/model_calib.py
# line 87: add "and False" to disable sync_amax_across_distributed_group


MODEL_PATH=${MODEL_PATH:-"/mnt/data/models/GLM-4.5-Air-Test/"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/glm45_air_ptq"}
QUANT_CFG=${QUANT_CFG:-""}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_SIZE=${CALIB_SIZE:-1}
NPROC=${NPROC:-2}
ENABLE_FP8_KVCACHE=${ENABLE_FP8_KVCACHE:-0}
ENABLE_WO_QUANT=${ENABLE_WO_QUANT:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
export TOKENIZERS_PARALLELISM=false

if [[ -z "${QUANT_CFG}" ]]; then
  echo "[ERROR] QUANT_CFG is required (must match a config in modelopt.torch.quantization)."
  echo "        e.g., QUANT_CFG=FP8_DEFAULT_CFG or your custom preset."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Flags
FP8_FLAG=$([[ "${ENABLE_FP8_KVCACHE}" == "1" ]] && echo "--enable_fp8_kvcache" || echo "")
WO_FLAG=$([[ "${ENABLE_WO_QUANT}" == "0" ]] && echo "--enable_wo_quant" || echo "")
TRUST_FLAG=$([[ "${TRUST_REMOTE_CODE}" == "1" ]] && echo "--trust_remote_code" || echo "")

# Always use torchrun (even for single rank) so dist.barrier() is valid
CMD=(
  torchrun --nproc_per_node="${NPROC}" ptq.py \
    --model_path "${MODEL_PATH}" \
    --quant_cfg "${QUANT_CFG}" \
    --output_path "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --calib_size "${CALIB_SIZE}" \
    ${FP8_FLAG} ${WO_FLAG} ${TRUST_FLAG}
)

# Print command for visibility
printf "Running:\n  %q\n" "${CMD[@]}"

# Execute
"${CMD[@]}"
