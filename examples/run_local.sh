set -euo pipefail


export PYTHONUNBUFFERED=1

export ROOT_DIR_BRAINTREEBANK=""


# subjects / trials：
subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
trials=(  1 2 0 4 0 1 0 1 0 1  0  1)

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

preprocess=('none')
splits_type=("CrossSession")
classifier_type=("transformer")


SPLIT="${splits_type[0]}"
save_dir="data/eval_results_lite_${SPLIT}"
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

  python -u eval_population.py \
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
