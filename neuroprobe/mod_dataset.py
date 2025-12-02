import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, json

from Neuroprobe.neuroprobe.config import *
from Neuroprobe.neuroprobe.braintreebank_subject import BrainTreebankSubject

# Defining the names of evaluations and preparing them for downstream processing
single_float_variables_name_remapping = {
    "pitch": "enhanced_pitch", #"pitch",
    "volume": "rms", #"rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length"
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos"
}
new_pitch_variables = ['enhanced_pitch', 'enhanced_volume', 'delta_enhanced_pitch', 'delta_enhanced_volume', 'raw_pitch', 'raw_volume', 'delta_raw_pitch', 'delta_raw_volume']
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys()) + new_pitch_variables
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index"] + classification_variables


class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, binary_tasks=False,
                 start_neural_data_before_word_onset=START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE, end_neural_data_after_word_onset=END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE,
                 lite=True, nano=False, random_seed=NEUROPROBE_GLOBAL_RANDOM_SEED, output_dict=False, max_samples=None, always_cache_full_subject=False,
                 binning_thresholds=None):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
            lite (bool, optional): if True, the eval is Neuroprobe (the default), otherwise it is Neuroprobe-Full
            nano (bool, optional): if True, the eval is Neuroprobe-Nano, otherwise it is Neuroprobe-Lite
            output_indices (bool, optional): whether to output indices instead of data
            binary_tasks (bool, optional): if True, the tasks will all be binary (default).
            output_dict (bool, optional): if True, outputs a dictionary with metadata.
            start_neural_data_before_word_onset (int, optional): samples before word onset
            end_neural_data_after_word_onset (int, optional): samples after word onset
            random_seed (int, optional): seed for random operations
            max_samples (int, optional): max samples to include
            always_cache_full_subject (bool, optional): cache full neural data
            binning_thresholds (dict, optional): pre-calculated thresholds for binning continuous variables. 
                                                 If None, thresholds are calculated from this dataset (Train mode).
                                                 If provided, these are applied to this dataset (Test mode).
        """

        # Set up a local random state with the provided seed
        self.rng = np.random.RandomState(random_seed)
        
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.binary_tasks = binary_tasks
        self.output_indices = output_indices
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset
        self.lite = lite
        self.nano = nano
        self.output_dict = output_dict
        self.max_samples = max_samples
        self.always_cache_full_subject = always_cache_full_subject
        self.binning_thresholds = binning_thresholds

        if self.nano:
            nano_electrodes = NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in nano_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_NANO_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_NANO_SUBJECT_TRIALS"
        elif self.lite:
            lite_electrodes = NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in lite_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_LITE_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_LITE_SUBJECT_TRIALS"
        else:
            # use all electrode labels and indices
            self.electrode_indices_subset = np.arange(len(subject.electrode_labels))
            self.electrode_labels = subject.electrode_labels
        self.electrode_coordinates = subject.get_electrode_coordinates()[self.electrode_indices_subset]

        eval_name_remapped = eval_name
        if eval_name in single_float_variables_name_remapping: eval_name_remapped = single_float_variables_name_remapping[eval_name]
        if eval_name in classification_variables_name_remapping: eval_name_remapped = classification_variables_name_remapping[eval_name]
        self.eval_name_remapped = eval_name_remapped

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)

        self.movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"{self.subject.subject_identifier}_{self.trial_id}"]
        
        # Add the original features from braintreebank to the all_words_df
        transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/features.csv')
        original_features_df = pd.read_csv(transcript_file_format.format(self.movie_name)).set_index('Unnamed: 0')
        # Add new columns from words_df using original_index mapping
        new_columns = [col for col in original_features_df.columns if col not in self.all_words_df.columns]
        for col in new_columns:
            self.all_words_df[col] = self.all_words_df['original_index'].map(original_features_df[col])
        
        if eval_name in single_float_variables:
            # Grab the new pitch volume features if they exist
            if self.eval_name_remapped in new_pitch_variables:
                pitch_volume_features_path = os.path.join(PITCH_VOLUME_FEATURES_DIR, f"{self.movie_name}_pitch_volume_features.json")
                with open(pitch_volume_features_path, 'r') as f:
                    raw_pitch_volume_features = json.load(f)

                TARGET_DP_FOR_KEYS = 5  # Standard number of decimal places
                normalized_pvf = {}
                for k_str, v_val in raw_pitch_volume_features.items():
                    k_float = float(k_str)
                    normalized_key = f"{k_float:.{TARGET_DP_FOR_KEYS}f}"
                    normalized_pvf[normalized_key] = v_val
                pitch_volume_features = normalized_pvf

                start_times = self.all_words_df['start'].to_list()
                all_labels = []
                for start_time_val in start_times:
                    lookup_key = f"{start_time_val:.{TARGET_DP_FOR_KEYS}f}"
                    label = pitch_volume_features[lookup_key][self.eval_name_remapped]
                    all_labels.append(label)
                all_labels = np.array(all_labels)
            else:
                all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()

            # --- CHANGED: Strict Train/Test Separation Logic ---
            # If thresholds are NOT provided, we are the training set. Calculate raw value cutoffs.
            if self.binning_thresholds is None:
                self.binning_thresholds = {
                    'p25': np.percentile(all_labels, 25),
                    'p50': np.percentile(all_labels, 50),
                    'p75': np.percentile(all_labels, 75)
                }
            
            # Apply thresholds (either learned just now, or passed in from training set)
            p25 = self.binning_thresholds['p25']
            p50 = self.binning_thresholds['p50']
            p75 = self.binning_thresholds['p75']

            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(all_labels > p75)[0],  # "Loud" / Top quartile
                    0: np.where(all_labels < p25)[0]   # "Quiet" / Bottom quartile
                }
            else:
                self.label_indices = {
                    3: np.where(all_labels >= p75)[0],
                    2: np.where((all_labels < p75) & (all_labels >= p50))[0],
                    1: np.where((all_labels < p50) & (all_labels >= p25))[0],
                    0: np.where(all_labels < p25)[0]
                }
            # ---------------------------------------------------

        elif eval_name in ["onset", "speech"]:
            self.label_indices = {
                1: np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.all_words_df)), # positive indices
                0: np.arange(len(self.nonverbal_df)) # negative indices
            }
        elif eval_name == "face_num":
            face_nums = self.all_words_df["face_num"].to_numpy().astype(int)
            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(face_nums > 0)[0],
                    0: np.where(face_nums == 0)[0]
                }
            else:
                self.label_indices = {
                    2: np.where(face_nums > 1)[0],
                    1: np.where(face_nums == 1)[0],
                    0: np.where(face_nums == 0)[0]
                }
        elif eval_name == "word_index":
            word_indices = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(word_indices == 0)[0],
                    0: np.where(word_indices == 1)[0]
                }
            else:
                self.label_indices = {
                    2: np.where(word_indices >= 2)[0],
                    1: np.where(word_indices == 1)[0],
                    0: np.where(word_indices == 0)[0]
                }
        elif eval_name == "word_head_pos":
            head_pos = self.all_words_df[self.eval_name_remapped].to_numpy().astype(int)
            self.label_indices = {
                1: np.where(head_pos == 0)[0],
                0: np.where(head_pos == 1)[0]
            }
        elif eval_name == "word_part_speech":
            pos = self.all_words_df[self.eval_name_remapped].to_numpy()  
            if self.binary_tasks: 
                self.label_indices = {
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0]
                }
            else:
                self.label_indices = {
                    4: np.where(pos == "ADV")[0],
                    3: np.where(pos == "ADJ")[0],
                    3: np.where(pos == "DET")[0],
                    2: np.where(pos == "PRON")[0],
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0]
                }
        elif eval_name == "word_gap":
            # For word_gap, we also want to respect the binning logic if thresholds are passed.
            # Calculating raw gap distribution:
            gaps = []
            valid_indices = []
            
            # Map valid gap indices to their gap values
            gap_map = {} 

            for i in range(1, len(self.all_words_df)):
                if self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']: continue
                gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                gaps.append(gap)
                valid_indices.append(i)
                gap_map[i] = gap
            
            gaps = np.array(gaps)

            # --- CHANGED: Strict Train/Test Separation Logic for Gap ---
            if self.binning_thresholds is None:
                self.binning_thresholds = {
                    'p25': np.percentile(gaps, 25),
                    'p50': np.percentile(gaps, 50),
                    'p75': np.percentile(gaps, 75)
                }
            
            p25 = self.binning_thresholds['p25']
            p50 = self.binning_thresholds['p50']
            p75 = self.binning_thresholds['p75']

            positive_indices = []
            negative_indices = []
            middle1_indices = []
            middle2_indices = []

            for idx in valid_indices:
                gap_val = gap_map[idx]
                if gap_val >= p75:
                    positive_indices.append(idx)
                elif gap_val >= p50:
                    middle2_indices.append(idx)
                elif gap_val >= p25:
                    middle1_indices.append(idx)
                else:
                    negative_indices.append(idx)
            
            if self.binary_tasks:
                self.label_indices = {
                    1: positive_indices,
                    0: negative_indices
                }
            else:
                self.label_indices = {
                    2: positive_indices,
                    1: middle2_indices,
                    2: middle1_indices,
                    0: negative_indices
                }
             # ---------------------------------------------------
        else:
            raise ValueError(f"Invalid eval_name: {eval_name}")

        self.n_classes = len(self.label_indices)
        n_samples_each = min([len(self.label_indices[label]) for label in self.label_indices])
        if self.lite: 
            n_samples_each = min(n_samples_each, NEUROPROBE_LITE_MAX_SAMPLES//self.n_classes)
        elif self.nano:
            n_samples_each = min(n_samples_each, NEUROPROBE_NANO_MAX_SAMPLES//self.n_classes)
        
        for label in list(self.label_indices.keys()):
            # Important: For Test sets, we should usually NOT be balancing/downsampling if we want a true eval, 
            # but for this specific pipeline (which seems to require balanced batches), we proceed. 
            # Ideally, test sets are not balanced artificially.
            # Assuming we stick to the existing pipeline logic of balancing all datasets:
            self.label_indices[label] = np.sort(self.rng.choice(self.label_indices[label], size=n_samples_each, replace=False))
            if self.max_samples is not None: 
                self.label_indices[label] = self.label_indices[label][:self.max_samples//self.n_classes]
        
        self.n_samples = sum([len(self.label_indices[label]) for label in self.label_indices])

        self.cache_window_from = None
        self.cache_window_to = None
        if not self.always_cache_full_subject:
            n_try_indices = self.n_classes 
            window_indices = []
            # Safety check if dataset is empty
            if self.n_samples > 0:
                indices_to_check = list(range(min(n_try_indices, self.n_samples))) + \
                                   list(range(max(0, self.n_samples-n_try_indices), self.n_samples))
                for i in indices_to_check:
                    (window_from, window_to), _ = self.__getitem__(i, force_output_indices=True)
                    window_indices.append(window_from)
                    window_indices.append(window_to)
                
                if window_indices:
                    self.cache_window_from = np.min(window_indices)
                    self.cache_window_to = np.max(window_indices)
        
    def get_binning_thresholds(self):
        """Returns the thresholds calculated on this dataset (if it was a training set)."""
        return self.binning_thresholds

    def _get_neural_data(self, window_from, window_to, force_output_indices=False):
        self.subject.load_neural_data(self.trial_id, cache_window_from=self.cache_window_from, cache_window_to=self.cache_window_to)
        if not self.output_indices and not force_output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            if self.lite or self.nano:
                input = input[self.electrode_indices_subset]
            return input.to(dtype=self.dtype)
        else:
            return window_from, window_to # just return the window indices

    def _positive_negative_getitem__(self, idx, force_output_indices=False):
        # even indices are positive samples, odd indices are negative samples
        current_label = (idx+1) % self.n_classes
        word_index = self.label_indices[current_label][idx//self.n_classes]
        if self.eval_name in ["onset", "speech"] and (current_label == 0): # for onset and speech, we need to get the nonverbal data
            row = self.nonverbal_df.iloc[word_index]
        else:
            row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = est_idx + int(self.start_neural_data_before_word_onset) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
        return input, current_label
        
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx, force_output_indices=False):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n_samples}")
        input, label = self._positive_negative_getitem__(idx, force_output_indices=force_output_indices)
        
        if self.output_dict:
            return {
                "data": input, 
                "label": label, 
                "electrode_labels": self.electrode_labels,
                "electrode_coordinates": self.electrode_coordinates,
                "metadata": {
                    "dataset_identifier": "braintreebank",
                    "subject_id": self.subject.subject_id,
                    "trial_id": self.trial_id,
                    "sampling_rate": 2048,
                }
            }
        else:
            return input, label