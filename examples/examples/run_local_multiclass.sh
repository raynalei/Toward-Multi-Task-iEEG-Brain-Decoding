!/bin/sh

SBATCH -A cis250217p                 
SBATCH -p GPU-shared                 
SBATCH --gres=gpu:v100-32:1          
SBATCH -t 8:00:00 
SBATCH -N 1
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=2
SBATCH --mem=48G
SBATCH --array=1-108                   
SBATCH -o /ocean/projects/cis250217p/ylei5/Neuroprobe/data/logs/%A_%a.out
SBATCH -e /ocean/projects/cis250217p/ylei5/Neuroprobe/data/logs/%A_%a.err



############################################
# Runtime environment
############################################

# source /ocean/projects/cis250217p/shared/envs/miniconda3/bin/activate
# conda activate yu_env

echo "===== NODE INFO ====="
hostname
echo "===== GPU INFO ====="
nvidia-smi || echo "No NVIDIA GPU visible (CPU job or driver hidden)."

export PYTHONUNBUFFERED=1
export PYTHONPATH=/ocean/projects/cis250217p/ylei5:$PYTHONPATH
export ROOT_DIR_BRAINTREEBANK=/ocean/projects/cis250217p/shared/data/braintreebank

# interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00 -A cis250217p
# source /ocean/projects/cis250217p/shared/envs/miniconda3/bin/activate



############################################
# Sweep settings
############################################

# declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
# declare -a trials=(  1 2 0 4 0 1 0 1 0 1  0  1)
declare -a subjects=(3 3)
declare -a trials=(0 1)

declare -a eval_names=(
  "frame_brightness" "global_flow" 
  "local_flow" "face_num"
  "volume" "pitch" "delta_volume" "speech" "onset"
  "gpt2_surprisal" "word_length" "word_gap"
  "word_index" "word_head_pos" "word_part_speech"
)

eval_names=($(IFS=,; echo "${eval_names[*]}"))

declare -a preprocess=(
    # 'none' # no preprocessing, just raw voltage
    #'stft_absangle', # magnitude and phase after FFT
    #'stft_realimag' # real and imaginary parts after FFT
    # 'stft_abs' # just magnitude after FFT ("spectrogram")
    'laplacian-stft_abs' # just magnitude after FFT ("spectrogram")

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

declare -a splits_type=(
  # "WithinSession"
  "CrossSession"
  # "CrossSubject"   
)

declare -a classifier_type=(
    # "linear"
    # "cnn"
    # "transformer"
    "mamba"
)

EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} % ${#splits_type[@]} ))
CLASSIFIER_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} % ${#classifier_type[@]} ))

EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
CLASSIFIER_TYPE=${classifier_type[$CLASSIFIER_TYPE_IDX]}
save_dir="/ocean/projects/cis250217p/ylei5/Neuroprobe/data/eval_results_lite_${SPLITS_TYPE}_multiclass"

echo "Eval: $EVAL_NAME | subj=$SUBJECT trial=$TRIAL"
echo "Preproc: $PREPROCESS | clf=$CLASSIFIER_TYPE | split=$SPLITS_TYPE"
echo "Save dir: $save_dir"

if [[ "$SPLITS_TYPE" == "DS_DM" && "$SUBJECT" == "2" ]]; then
  echo "Cross-subject split invalid for subject 2; exiting."
  exit 0
fi


echo "===== SLURM/GPU CHECK ====="
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import torch, os, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    raise SystemExit("No CUDA available")
PY
############################################
# Launch
############################################
cd /ocean/projects/cis250217p/ylei5
python -m  Neuroprobe.examples.eval_population_mod \
  --eval_name "$EVAL_NAME" \
  --subject_id "$SUBJECT" \
  --trial_id "$TRIAL" \
  --preprocess.type "$PREPROCESS" \
  --verbose \
  --save_dir "$save_dir" \
  --split_type "$SPLITS_TYPE" \
  --classifier_type "$CLASSIFIER_TYPE" \
  --binary_tasks False \
  --only_1second
