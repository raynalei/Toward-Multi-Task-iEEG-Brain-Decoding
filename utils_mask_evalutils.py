# utils_mask_evalutils.py

import torch
from torch.utils.data import Dataset

from examples.eval_utils import preprocess_data
from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.config as neuroprobe_config


def patchify_grid(mag: torch.Tensor, t_patch=16, f_patch=16, t_stride=16, f_stride=16):
    """
    Convert a 2D feature map (time x freq) into a grid of flattened patches.

    Args:
        mag:      Tensor of shape (T, F)
        t_patch:  Patch size along time dimension
        f_patch:  Patch size along frequency dimension
        t_stride: Stride along time; defaults to t_patch
        f_stride: Stride along freq; defaults to f_patch

    Returns:
        Tensor of shape (num_patches, t_patch * f_patch)
    """
    T, F = mag.shape
    t_stride = t_patch if t_stride is None else t_stride
    f_stride = f_patch if f_stride is None else f_stride

    patches = []
    for ti in range(0, T - t_patch + 1, t_stride):
        for fi in range(0, F - f_patch + 1, f_stride):
            patches.append(mag[ti:ti + t_patch, fi:fi + f_patch].reshape(-1))

    return torch.stack(patches, 0)   # (num_patches, t_patch * f_patch)


def make_mask(L: int, p: float = 0.3):
    """
    Bernoulli mask generator.

    Args:
        L: Sequence length
        p: Probability of masking each position

    Returns:
        Bool tensor of shape (L,) where True = masked
    """
    return torch.rand(L) < p


class EEGMaskDataset(Dataset):
    """
    Masked-patch EEG dataset using BrainTreebankSubject internally.

    This dataset:
        - Iterates over multiple subjects and trials
        - Loads raw iEEG from BrainTreebank HDF5 files
        - Slices each (subject, trial) recording into temporal windows
        - Applies Neuroprobe-style preprocessing (laplacian, STFT, etc.)
        - Converts per-window features to 2D (T, F) spectrograms
        - Patchifies them and samples a binary mask over patches

    Each __getitem__ returns:
        {
            "cont_feats": (num_patches, patch_dim),
            "labels":     (num_patches,),    # bool mask
            "meta":       {...}              # optional metadata (subject, trial, window)
        }

    Args:
        subject_ids: list[int] or int. Subject IDs to include.
        trial_ids:   list[int] or int. Trial IDs (per subject) to include.
        preprocess:  Preprocessing type string, e.g. "laplacian-stft_abs".
        preprocess_parameters: Dict with STFT and other parameters (same
                               structure as used in eval_population.py).
        mask_prob:   Probability of masking each patch.
        t_patch,f_patch: Patch sizes in time/freq.
        window_size_seconds: Temporal window length in seconds.
        stride_seconds:      Stride between windows in seconds
                             (defaults to window_size_seconds).
        lite:        If True, use the official "lite" electrode subset per subject.
        electrodes:  "all" | "lite" | list[str]. Additional electrode filter.
        cache:       Whether to use BrainTreebankSubject caching.
        dtype:       Torch dtype for raw data.
    """

    def __init__(
        self,
        subject_ids,
        trial_ids,
        preprocess: str,
        preprocess_parameters: dict,
        mask_prob: float = 0.3,
        t_patch: int = 2,
        f_patch: int = 8,
        window_size_seconds: float = 1.0,
        stride_seconds: float = None,
        lite: bool = True,
        electrodes="lite",
        cache: bool = True,
        dtype=torch.float16,
    ):
        super().__init__()

        # Normalize subject_ids and trial_ids to lists
        if isinstance(subject_ids, int):
            subject_ids = [subject_ids]
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids]

        self.subject_ids = subject_ids
        self.trial_ids = trial_ids
        self.preprocess = preprocess
        self.preprocess_parameters = preprocess_parameters
        self.mask_prob = mask_prob
        self.t_patch = t_patch
        self.f_patch = f_patch

        self.samples = []   

        sr = neuroprobe_config.SAMPLING_RATE
        window_size = int(window_size_seconds * sr)
        if stride_seconds is None:
            stride_seconds = window_size_seconds
        stride = int(stride_seconds * sr)

        self._subject_cache = {}
        self._dtype = dtype
        self._cache_flag = cache
        self._lite = lite
        self._electrodes_arg = electrodes

        for sid in self.subject_ids:

            subject = BrainTreebankSubject(
                sid,
                allow_corrupted=False,
                allow_missing_coordinates=False,
                cache=self._cache_flag,
                dtype=dtype,
                coordinates_type="cortical",
            )

            if lite:
                subject.set_electrode_subset(subject.get_lite_electrodes())

            if isinstance(electrodes, list):
                subject.set_electrode_subset(electrodes)
            elif electrodes == "all":
                pass
            elif electrodes == "lite":
                subject.set_electrode_subset(subject.get_lite_electrodes())
            else:
                raise ValueError(f"Invalid electrodes argument: {electrodes}")

            subj_electrode_labels = subject.electrode_labels

            for tid in self.trial_ids:
                try:

                    all_elec_data = subject.get_all_electrode_data(tid)
                except FileNotFoundError:
                    continue

                n_samples_total = all_elec_data.shape[1]
                if n_samples_total < window_size:
                    continue

                window_starts = range(0, n_samples_total - window_size + 1, stride)

                for w_start in window_starts:
                    w_end = w_start + window_size

                    self.samples.append(
                        {
                            "subject_id": sid,
                            "trial_id": tid,
                            "window_start": w_start,
                            "window_end": w_end,
                            "electrode_labels": subj_electrode_labels,
                        }
                    )

        self.num_windows = len(self.samples)

    def __len__(self):
        return self.num_windows
    
    def _get_subject(self, sid):

        if sid not in self._subject_cache:
            subject = BrainTreebankSubject(
                sid,
                allow_corrupted=False,
                allow_missing_coordinates=False,
                cache=self._cache_flag,
                dtype=self._dtype,
                coordinates_type="cortical",
            )
            if self._lite:
                subject.set_electrode_subset(subject.get_lite_electrodes())

            if isinstance(self._electrodes_arg, list):
                subject.set_electrode_subset(self._electrodes_arg)
            elif self._electrodes_arg == "all":
                pass
            elif self._electrodes_arg == "lite":
                subject.set_electrode_subset(subject.get_lite_electrodes())
            else:
                raise ValueError(f"Invalid electrodes argument: {self._electrodes_arg}")

            self._subject_cache[sid] = subject
        return self._subject_cache[sid]
    
    def __getitem__(self, idx):
        """
        Single sample pipeline:

            raw_window (n_elec, T_raw)
              → preprocess_data (Neuroprobe pipeline)
              → select one electrode
              → ensure 2D (T, F)
              → patchify_grid
              → sample binary mask over patches
        """
        sample = self.samples[idx]
        sid = sample["subject_id"]
        tid = sample["trial_id"]
        w_start = sample["window_start"]
        w_end = sample["window_end"]
        electrode_labels = sample["electrode_labels"]

        subject = self._get_subject(sid)
        all_elec_data = subject.get_all_electrode_data(tid)   # (n_elec, total_samples)
        window_data = all_elec_data[:, w_start:w_end]         # (n_elec, window_size)

        if not isinstance(window_data, torch.Tensor):
            raw_data = torch.tensor(window_data, dtype=torch.float32)
        else:
            raw_data = window_data.to(dtype=torch.float32)

        proc = preprocess_data(
            raw_data,
            electrode_labels,
            preprocess=self.preprocess,
            preprocess_parameters=self.preprocess_parameters,
        )
        # STFT case: (1, n_elec, T, F)
        # Other cases: (1, n_elec, T_samples), etc.

        # 2) Remove batch dimension if present
        if proc.ndim == 4:
            # (1, n_elec, T, F) -> (n_elec, T, F)
            proc = proc[0]
        elif proc.ndim == 3:
            # (1, n_elec, T) -> (n_elec, T)
            proc = proc[0]
        else:
            raise ValueError(f"Unexpected preprocessed shape: {proc.shape}")

        # 3) Take the first electrode for now
        feat = proc[0]  # (T, F) or (T,)

        # 4) Ensure 2D feature map
        if feat.ndim == 1:
            # Only time dimension -> add freq dim of size 1
            feat = feat.unsqueeze(-1)  # (T,) -> (T,1)
        elif feat.ndim != 2:
            raise ValueError(f"Expected 2D (T, F) feature map, got shape {feat.shape}")

        mag = feat.to(dtype=torch.float32)  # (T, F)

        # 5) Patchify and sample mask
        patches = patchify_grid(mag, self.t_patch, self.f_patch)
        mask = make_mask(patches.size(0), self.mask_prob)

        return {
            "cont_feats": patches,   # (num_patches, patch_dim)
            "labels": mask,          # (num_patches,)
            "meta": {
                "subject_id": sample["subject_id"],
                "trial_id": sample["trial_id"],
                "window_start": sample["window_start"],
                "window_end": sample["window_end"],
            },
        }


def collate_mask(batch):
    """
    Collate function for masked EEG patch reconstruction.

    Each dataset item contains:
        "cont_feats": Tensor of shape (Pi, D)
        "labels":     Tensor of shape (Pi,)
        "meta":       dict with subject/trial/window info (optional for model)

    Different samples may have different patch counts Pi.
    This collate pads all sequences in the batch to max(Pi) so they can form
    a proper (B, Pmax, D) tensor.

    Returns:
        {
            "cont_feats": (B, Pmax, D) float32   # padded patch sequences
            "labels":     (B, Pmax) bool         # masked positions
            "attn_mask":  (B, Pmax) bool         # True for real patches, False for pads
            "lengths":    list[int]              # original Pi for each sample
            "meta":       list[dict]             # metadata per sample
        }
    """

    B = len(batch)

    # Number of patches per sample
    lengths = [item["cont_feats"].size(0) for item in batch]
    Pmax = max(lengths)
    D = batch[0]["cont_feats"].size(1)

    # Allocate padded tensors
    cont_feats = torch.zeros(B, Pmax, D, dtype=torch.float32)
    labels = torch.zeros(B, Pmax, dtype=torch.bool)
    attn_mask = torch.zeros(B, Pmax, dtype=torch.bool)

    metas = []

    # Fill tensors
    for i, item in enumerate(batch):
        P = lengths[i]
        cont_feats[i, :P] = item["cont_feats"]
        labels[i, :P]     = item["labels"]
        attn_mask[i, :P]  = True
        metas.append(item.get("meta", {}))

    return {
        "cont_feats": cont_feats,
        "labels": labels,
        "attn_mask": attn_mask,
        "lengths": lengths,
        "meta": metas,
    }
