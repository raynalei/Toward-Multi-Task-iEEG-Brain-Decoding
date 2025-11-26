set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"        # so data/ and logs/ land under repo root

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
export PYTHONUNBUFFERED=1
export ROOT_DIR_BRAINTREEBANK=/home/lucas/Work/codetest/project11785/braintreebank/

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project11785


# subjects / trials：与原脚本一致（12 组）
#subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
#trials=(  1 2 0 4 0 1 0 1 0 1  0  1)
subjects=(3)
trials=(0)

eval_names=(
  "frame_brightness"
  "global_flow"
  "local_flow"
  "face_num"
  "volume"
  "pitch"
  "delta_volume"
  "speech"
  "onset"
  "gpt2_surprisal"
  "word_length"
  "word_gap"
  "word_index"
  "word_head_pos"
  "word_part_speech"
)

eval_names=($(IFS=,; echo "${eval_names[*]}"))

preprocess=(
    #'none' # no preprocessing, just raw voltage
    #'stft_absangle', # magnitude and phase after FFT
    #'stft_realimag' # real and imaginary parts after FFT
    #'stft_abs' # just magnitude after FFT ("spectrogram")
    'laplacian-stft_abs' # just magnitude after FFT ("spectrogram")

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

splits_type=("CrossSession")
classifier_type=("transformer")


SPLIT="${splits_type[0]}"
save_dir="data/eval_results_lite_${SPLIT}_laplacian-stft_abs_lucas_transformer_embedding_6_3"
mkdir -p "${save_dir}" data/logs

echo "Save dir: ${save_dir}"
echo "Split type: ${SPLIT}"


for (( idx=0; idx<${#subjects[@]}; idx++ )); do
  SUBJECT=${subjects[$idx]}
  TRIAL=${trials[$idx]}


  EVAL_NAME=${eval_names[0]}
  PREP=${preprocess[0]}
  CLF=${classifier_type[0]}

  echo ">>> Running eval for eval ${EVAL_NAME}, subject ${SUBJECT}, trial ${TRIAL}, preprocess ${PREP}, classifier ${CLF}"


  LOG_BASE="data/logs/local_s${SUBJECT}t${TRIAL}"

python -u examples/eval_population.py \
    --eval_name "${EVAL_NAME}" \
    --subject_id "${SUBJECT}" \
    --trial_id "${TRIAL}" \
    --preprocess.type "${PREP}" \
    --verbose \
    --save_dir "${save_dir}" \
    --split_type "${SPLIT}" \
    --classifier_type "${CLF}" \
    --only_1second \
    > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

  echo ">>> Done: subject=${SUBJECT}, trial=${TRIAL}. Logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
done

echo "All local runs finished."
