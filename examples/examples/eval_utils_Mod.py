import Neuroprobe.neuroprobe.config as neuroprobe_config

from sklearn.metrics import roc_auc_score
import torch, numpy as np
import time, psutil
import torch.nn as nn
import math

verbose = True # print logs

############## LOGGING ###############

def model_name_from_classifier_type(classifier_type):
    if classifier_type == 'linear':
        return "Logistic Regression"
    elif classifier_type == 'cnn':
        return "CNN"
    elif classifier_type == 'transformer':
        return "Transformer"
    elif classifier_type == 'mlp':
        return "MLP"
    elif classifier_type == 'mamba':
        return "Mamba"
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")

def log(message, priority=0, indent=0):
    max_log_priority = -1 if not verbose else 4
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


############## ELECTRODE SUBSET ###############

def subset_electrodes(subject, lite=False, nano=False):
    all_electrode_labels = subject.electrode_labels
    if lite:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
    elif nano:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    return all_electrode_labels


############## DATA PREPROCESSING ###############

from scipy import signal
import numpy as np

def preprocess_stft(data, sampling_rate=2048, preprocess="stft_abs", 
                    preprocess_parameters={"stft": {"nperseg": 512, "poverlap": 0.75, "window": "hann", "max_frequency": 150, "min_frequency": 0}}):
    was_tensor = isinstance(data, torch.Tensor)
    x = torch.from_numpy(data) if not was_tensor else data

    if len(x.shape) == 2: # if it is only (n_electrodes, n_samples)
        x = x.unsqueeze(0)
    # data is of shape (batch_size, n_electrodes, n_samples)
    batch_size, n_electrodes, n_samples = x.shape

    # convert to float32 and reshape for STFT
    x = x.to(dtype=torch.float32)
    x = x.reshape(batch_size * n_electrodes, -1)

    # STFT parameters
    nperseg = preprocess_parameters["stft"]["nperseg"]
    poverlap = preprocess_parameters["stft"]["poverlap"]
    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap

    if preprocess_parameters["stft"]["window"] == "hann":
        window = torch.hann_window(nperseg, device=x.device)
    elif preprocess_parameters["stft"]["window"] == "boxcar":
        window = torch.ones(nperseg, device=x.device)
    else:
        raise ValueError(f"Invalid window type: {preprocess_parameters['stft']['window']}")
    
    max_frequency = preprocess_parameters["stft"]["max_frequency"]
    min_frequency = preprocess_parameters["stft"]["min_frequency"]

    # Compute STFT
    x = torch.stft(x,
                    n_fft=nperseg, 
                    hop_length=hop_length,
                    win_length=nperseg,
                    window=window,
                    return_complex=True,
                    normalized=False,
                    center=True)
    # Get frequency bins
    freqs = torch.fft.rfftfreq(nperseg, d=1.0/sampling_rate) # 2048Hz sampling rate
    x = x[:, (freqs >= min_frequency) & (freqs <= max_frequency)]

    if preprocess == "stft_absangle":
        # Split complex values into magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        # Stack magnitude and phase along a new axis
        x = torch.stack([magnitude, phase], dim=-2)
    elif preprocess == "stft_realimag":
        real = torch.real(x)
        imag = torch.imag(x)
        x = torch.stack([real, imag], dim=-2)
    elif preprocess == "stft_abs":   
        x = torch.abs(x)
    else:
        raise ValueError(f"Invalid preprocess type: {preprocess}")

    # Reshape back
    _, n_freqs, n_times = x.shape
    x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
    x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)
    
    # Z-score normalization
    # NOTE: skipping batch norm here because in the regression pipeline, StandardScaler is used anyway,
    # and we would like to avoid batch effects in case input items are processed one by one. TODO: find a better idea here
    # x = x - x.mean(dim=[0, 2], keepdim=True)
    # x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

    return x.numpy() if not was_tensor else x

def downsample(data, fs=2048, downsample_rate=200):
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    # Apply downsampling
    result = signal.resample_poly(data_np, up=downsample_rate, down=fs, axis=-1)
    
    # Convert back to tensor if input was a tensor
    if was_tensor:
        result = torch.from_numpy(result).to(device)
    
    return result
def remove_line_noise(data, fs=2048, line_freq=60):
    """Remove line noise (60 Hz and harmonics) from neural data.
    
    Args:
        data (numpy.ndarray or torch.Tensor): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
        line_freq (int): Fundamental line frequency in Hz (typically 60 Hz in the US)
        
    Returns:
        numpy.ndarray or torch.Tensor: Filtered data with the same shape as input (same type as input)
    """
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        filtered_data = data.cpu().numpy().copy()
    else:
        filtered_data = data.copy()
    
    # Define the width of the notch filter (5 Hz on each side)
    bandwidth = 5.0
    
    # Calculate the quality factor Q
    Q = line_freq / bandwidth
    
    # Apply notch filters for the fundamental frequency and harmonics
    # We'll filter up to the 5th harmonic (60, 120, 180, 240, 300 Hz)
    for harmonic in range(1, 6):
        harmonic_freq = line_freq * harmonic
        
        # Skip if the harmonic frequency is above the Nyquist frequency
        if harmonic_freq > fs/2:
            break
            
        # Create and apply a notch filter
        b, a = signal.iirnotch(harmonic_freq, Q, fs)
        
        # Apply the filter along the time dimension
        if filtered_data.ndim == 2:  # (n_channels, n_samples)
            filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        elif filtered_data.ndim == 3:  # (batch_size, n_channels, n_samples)
            for i in range(filtered_data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i], axis=1)
    
    # Convert back to tensor if input was a tensor
    if was_tensor:
        filtered_data = torch.from_numpy(filtered_data).to(device)
    
    return filtered_data

def laplacian_rereference_neural_data(electrode_data, electrode_labels, remove_non_laplacian=True):
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)
    inputs:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes from the data; if false, keep them without rereferencing
    outputs:
        rereferenced_data: torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
        rereferenced_labels: list of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """
    def get_all_laplacian_electrodes(electrode_labels):
        """
            Get all laplacian electrodes for a given subject. This function is originally from
            https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """
        def stem_electrode_name(name):
            #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            #names look like 'T1b2
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return ''.join(reversed(stem)), int(''.join(reversed(num)))
        def has_neighbors(stem, stems):
            (x,y) = stem
            return ((x,y+1) in stems) or ((x,y-1) in stems)
        def get_neighbors(stem, stems):
            (x,y) = stem
            return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)] if (x, y) in stems]
        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes}
        return electrodes, neighbors

    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)

    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]

    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(electrode_labels)
    laplacian_neighbor_indices = {laplacian_electrode_label: [electrode_labels.index(neighbor_label) for neighbor_label in neighbors] for laplacian_electrode_label, neighbors in laplacian_neighbors.items()}

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    if was_tensor:
        rereferenced_data = torch.zeros((batch_size, rereferenced_n_electrodes, n_samples), dtype=electrode_data.dtype, device=electrode_data.device)
    else:
        rereferenced_data = np.zeros((batch_size, rereferenced_n_electrodes, n_samples), dtype=electrode_data.dtype)

    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i] - electrode_data[:, laplacian_neighbor_indices[electrode_label]].mean(axis=1)
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian: 
                continue # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1
                
    if batch_unsqueeze:
        if was_tensor:
            rereferenced_data = rereferenced_data.squeeze(0)
        else:
            rereferenced_data = rereferenced_data.squeeze(0)

    return rereferenced_data, laplacian_electrodes if remove_non_laplacian else electrode_labels, original_electrode_indices

def preprocess_data(data, electrode_labels, preprocess, preprocess_parameters):
    for preprocess_option in preprocess.split('-'):
        if preprocess_option.lower() in ['stft_absangle', 'stft_realimag', 'stft_abs']:
            data = preprocess_stft(data, preprocess=preprocess_option, preprocess_parameters=preprocess_parameters)
        elif preprocess_option.lower() == 'remove_line_noise':
            data = remove_line_noise(data)
        elif preprocess_option.lower() == 'downsample_200':
            data = downsample(data, downsample_rate=200)
        elif preprocess_option.lower() == 'downsample_500':
            data = downsample(data, downsample_rate=500)
        elif preprocess_option.lower() == 'laplacian':
            data, electrode_labels, original_electrode_indices = laplacian_rereference_neural_data(data, electrode_labels, remove_non_laplacian=False)
    return data



############## CLASSIFICATION ###############


import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score

class TransformerClassifier:
    """
    A Transformer-based classifier for EEG downstream tasks.
    This version requires an explicit validation set (X_val, y_val) to ensure
    apple-to-apple comparison across models. No internal train/val split is performed.
    """

    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001,
                 tol=1e-4, patience=10, d_model=64, nhead=8, dim_feedforward=256,
                 dropout=0.1, num_layers=3):

        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.patience = patience

        # Always run on GPU for speed
        self.device = torch.device('cuda')

        # Model hyperparameters
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers

    def _create_model(self, input_shape, n_classes):
        """
        Create the Transformer model. Handles 2D and 3D inputs:
        (channels, time) or (channels, freq, time)
        """

        class PositionalEncoding(torch.nn.Module):
            """Standard sinusoidal positional encoding."""
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                     * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class Transformer(torch.nn.Module):
            """
            Minimal Transformer encoder + average pooling + linear classifier.
            """
            def __init__(self, input_shape, n_classes, d_model, nhead,
                         dim_feedforward, dropout, num_layers):
                super().__init__()
                self.d_model = d_model

                # Case 1: (channels, time)
                if len(input_shape) == 2:
                    C, T = input_shape
                    self.input_proj = torch.nn.Linear(C, d_model)
                    self.pos_encoder = PositionalEncoding(d_model, max_len=T)

                # Case 2: (channels, freq, time)
                else:
                    C, F, T = input_shape
                    self.input_proj = torch.nn.Linear(C * F, d_model)
                    self.pos_encoder = PositionalEncoding(d_model, max_len=T)

                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

                # Final classifier
                self.fc = torch.nn.Linear(d_model, n_classes)

            def forward(self, x):
                # Input reshape depending on input dimensions
                if len(x.shape) == 3:  # (B, C, T)
                    x = x.transpose(1, 2)  # (B, T, C)
                    x = self.input_proj(x)  # (B, T, d_model)
                else:  # (B, C, F, T)
                    B, C, F, T = x.shape
                    x = x.transpose(1, 3)  # (B, T, C, F)
                    x = x.reshape(B, T, C * F)
                    x = self.input_proj(x)

                x = self.pos_encoder(x)
                x = self.encoder(x)

                # Global average pooling over time dimension
                x = x.mean(dim=1)

                return self.fc(x)

        return Transformer(
            input_shape, n_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the Transformer classifier using an explicit validation set.
        - X_train, y_train: training data
        - X_val, y_val: validation data (must be provided)
        """

        # Convert to numpy → torch
        X_train = torch.FloatTensor(np.asarray(X_train))
        y_train = torch.LongTensor(np.asarray(y_train))
        X_val = torch.FloatTensor(np.asarray(X_val))
        y_val = torch.LongTensor(np.asarray(y_val))

        # Class list (for prediction mapping and AUROC computation)
        self.classes_ = np.unique(y_train.numpy())
        n_classes = len(self.classes_)

        # Create model
        input_shape = X_train.shape[1:]
        self.model = self._create_model(input_shape, n_classes).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TransformerClassifier] Trainable params: {n_params} (~{n_params/1e6:.2f}M)")#268868

        # Optimizer + loss
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_auroc = 0.0
        best_state = None
        patience_counter = 0

        # ------------------------
        # Training Loop
        # ------------------------
        for epoch in range(self.max_iter):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, len(X_train), self.batch_size):
                bx = X_train[i:i + self.batch_size].to(self.device)
                by = y_train[i:i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * bx.size(0)
                _, preds = torch.max(logits, 1)
                total += by.size(0)
                correct += (preds == by).sum().item()

            train_loss = epoch_loss / total
            train_acc = correct / total

            # ------------------------
            # Validation
            # ------------------------
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val.to(self.device))
                val_loss = criterion(val_logits, y_val.to(self.device)).item()

                probs = torch.nn.functional.softmax(val_logits, dim=1).cpu().numpy()
                y_np = y_val.cpu().numpy()

                # One-hot labels for AUROC
                y_onehot = np.zeros((len(y_np), n_classes))
                y_onehot[np.arange(len(y_np)), y_np] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_onehot, probs, multi_class="ovr", average="macro"
                    )
                else:
                    val_auroc = roc_auc_score(y_onehot, probs)

            # Print training info
            print(
                f"[Epoch {epoch+1}/{self.max_iter}] "
                f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f}, AUROC={val_auroc:.4f}"
            )

            # ------------------------
            # Early stopping
            # ------------------------
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Load best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_val_auroc = best_val_auroc
        print(f"Training completed. Best val AUROC = {best_val_auroc:.4f}")

        return self

    def predict_proba(self, X):
        """Return class probabilities."""
        self.model.eval()
        X = torch.FloatTensor(np.asarray(X))

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                bx = X[i:i + self.batch_size].to(self.device)
                logits = self.model(bx)
                probs = torch.nn.functional.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        """Return predicted class indices."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Return accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)



class CNNClassifier:
    """
    A CNN-based classifier for EEG downstream tasks.
    This version requires an explicit validation set (X_val, y_val) to ensure
    apple-to-apple comparison across baselines.
    No internal validation split is performed.
    """

    def __init__(self, random_state=42, max_iter=100, batch_size=128,
                 learning_rate=0.0001, tol=1e-4, patience=10):

        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.patience = patience

        # Always run on GPU
        self.device = torch.device("cuda")

        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0

    def _create_model(self, input_shape, n_classes):
        """
        Create a simple CNN classifier.
        Supports:
        - (channels, time)
        - (channels, freq, time)
        """

        class CNN(torch.nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()

                # Case 1: 1D CNN (channels, time)
                if len(input_shape) == 2:
                    C, T = input_shape
                    self.conv1 = torch.nn.Conv1d(C, 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool1d(2)
                    self.dropout = torch.nn.Dropout(0.5)

                    # Downsampled by 2×2×2 = 8 in time
                    conv_output = (T // 8) * 128

                # Case 2: 2D CNN (channels, freq, time)
                else:
                    C, F, T = input_shape
                    self.conv1 = torch.nn.Conv2d(C, 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2)
                    self.dropout = torch.nn.Dropout(0.5)

                    # Downsample freq/time each by 2×2×2 = 8
                    conv_output = (F // 8) * (T // 8) * 128

                self.relu = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(conv_output, 256)
                self.fc2 = torch.nn.Linear(256, n_classes)

                self.input_is_1d = (len(input_shape) == 2)

            def forward(self, x):
                if self.input_is_1d:
                    # x: (B, C, T)
                    x = self.relu(self.conv1(x))
                    x = self.pool(x)
                    x = self.relu(self.conv2(x))
                    x = self.pool(x)
                    x = self.relu(self.conv3(x))
                    x = self.pool(x)
                else:
                    # x: (B, C, F, T)
                    x = self.relu(self.conv1(x))
                    x = self.pool(x)
                    x = self.relu(self.conv2(x))
                    x = self.pool(x)
                    x = self.relu(self.conv3(x))
                    x = self.pool(x)

                # Flatten
                x = x.reshape(x.size(0), -1)

                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)

        return CNN(input_shape, n_classes)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the classifier using an explicit validation set.
        No internal split is performed.
        """

        # Convert numpy → torch
        X_train = torch.FloatTensor(np.asarray(X_train))
        y_train = torch.LongTensor(np.asarray(y_train))
        X_val = torch.FloatTensor(np.asarray(X_val))
        y_val = torch.LongTensor(np.asarray(y_val))

        # Class mapping
        self.classes_ = np.unique(y_train.numpy())
        n_classes = len(self.classes_)

        # Create model
        input_shape = X_train.shape[1:]
        self.model = self._create_model(input_shape, n_classes).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[CNNClassifier] Trainable params: {n_params} (~{n_params/1e6:.2f}M)")#387204

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_state = None
        best_val_auroc = 0.0
        patience_counter = 0

        # ------------------------
        # Training Loop
        # ------------------------
        for epoch in range(self.max_iter):
            self.model.train()
            total = 0
            correct = 0
            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                bx = X_train[i:i+self.batch_size].to(self.device)
                by = y_train[i:i+self.batch_size].to(self.device)

                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * bx.size(0)
                _, pred = torch.max(logits, dim=1)
                total += by.size(0)
                correct += (pred == by).sum().item()

            train_loss = epoch_loss / total
            train_acc = correct / total

            # ------------------------
            # Validation
            # ------------------------
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val.to(self.device))
                val_loss = criterion(val_logits, y_val.to(self.device)).item()

                probs = torch.nn.functional.softmax(val_logits, dim=1).cpu().numpy()
                y_np = y_val.cpu().numpy()

                # One-hot for AUROC
                y_onehot = np.zeros((len(y_np), n_classes))
                y_onehot[np.arange(len(y_np)), y_np] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_onehot, probs, multi_class="ovr", average="macro"
                    )
                else:
                    val_auroc = roc_auc_score(y_onehot, probs)

            print(
                f"[Epoch {epoch+1}/{self.max_iter}] "
                f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f}, AUROC={val_auroc:.4f}"
            )

            # ------------------------
            # Early Stopping
            # ------------------------
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Load best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_val_auroc = best_val_auroc
        print(f"Training completed. Best val AUROC = {best_val_auroc:.4f}")

        return self

    def predict_proba(self, X):
        """Return class probabilities."""
        self.model.eval()
        X = torch.FloatTensor(np.asarray(X))
        probs_list = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                bx = X[i:i+self.batch_size].to(self.device)
                logits = self.model(bx)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list, axis=0)

    def predict(self, X):
        """Return predicted class index."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Return accuracy."""
        return np.mean(self.predict(X) == y)
    
class MLPClassifier:
    """
    A simple MLP classifier for EEG downstream tasks.
    Requires an explicit validation set (X_val, y_val) for early stopping
    to ensure apple-to-apple comparison across models.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 100,
        batch_size: int = 200,
        learning_rate: float = 1e-5,
        hidden_dims=None,
        tol: float = 1e-8,
        patience: int = 100,
    ):
        """
        Args:
            hidden_dims: list of integers specifying hidden layer sizes.
                         - [] or None → pure linear classifier (logistic regression)
                         - e.g. [1024, 1024] → 2 hidden layers with 1024 units each
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.patience = patience
        self.hidden_dims = hidden_dims if hidden_dims is not None else []

        # Always run on GPU for consistency with other baselines
        self.device = torch.device("cuda")

        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0

    def _create_model(self, input_size: int, n_classes: int):
        """
        Build an MLP with optional hidden layers.
        """

        class MLP(torch.nn.Module):
            def __init__(self, input_size, n_classes, hidden_dims):
                super().__init__()
                layers = []

                if len(hidden_dims) == 0:
                    # Pure linear classifier (logistic regression)
                    layers.append(torch.nn.Linear(input_size, n_classes))
                else:
                    # MLP with one or more hidden layers
                    prev_dim = input_size
                    for h in hidden_dims:
                        layers.append(torch.nn.Linear(prev_dim, h))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Dropout(0.2))
                        prev_dim = h
                    # Output layer
                    layers.append(torch.nn.Linear(prev_dim, n_classes))

                self.network = torch.nn.Sequential(*layers)

            def forward(self, x):
                # Flatten all non-batch dimensions: (B, *) → (B, D)
                x = x.reshape(x.size(0), -1)
                return self.network(x)

        return MLP(input_size, n_classes, self.hidden_dims)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the MLP with an explicit validation set.
        No internal train/val split is performed.

        Args:
            X_train: training data, shape (N_train, ...)
            y_train: training labels, shape (N_train,)
            X_val: validation data, shape (N_val, ...)
            y_val: validation labels, shape (N_val,)
        """

        # Convert to numpy → torch
        X_train = torch.FloatTensor(np.asarray(X_train))
        y_train = torch.LongTensor(np.asarray(y_train))
        X_val = torch.FloatTensor(np.asarray(X_val))
        y_val = torch.LongTensor(np.asarray(y_val))

        # Classes and number of classes
        self.classes_ = np.unique(y_train.cpu().numpy())
        n_classes = len(self.classes_)

        model_type = "Linear" if len(self.hidden_dims) == 0 else f"MLP{self.hidden_dims}"
        log(
            f"Training {model_type} with {len(X_train)} train samples and "
            f"{len(X_val)} val samples",
            priority=3,
            indent=2,
        )

        # Determine input dimensionality after flattening
        input_size = int(np.prod(X_train.shape[1:]))
        self.model = self._create_model(input_size, n_classes).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)#281660
        print(f"[MLPClassifier] Trainable params: {n_params} (~{n_params/1e6:.2f}M)")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_model_state = None
        best_val_auroc = 0.0
        patience_counter = 0

        # ------------------------
        # Training loop
        # ------------------------
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0

            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i + self.batch_size].to(self.device)
                batch_y = y_train[i:i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(logits, dim=1)
                train_total += batch_y.size(0)
                train_correct += (preds == batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ------------------------
            # Compute training AUROC
            # ------------------------
            self.model.eval()
            with torch.no_grad():
                train_probs_list = []
                y_train_np = y_train.cpu().numpy()

                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i + self.batch_size].to(self.device)
                    logits = self.model(batch_X)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    train_probs_list.append(probs.cpu().numpy())

                train_probs = np.concatenate(train_probs_list, axis=0)

                # One-hot encode labels for multi-class AUROC
                y_train_onehot = np.zeros((len(y_train_np), n_classes))
                y_train_onehot[np.arange(len(y_train_np)), y_train_np] = 1

                if n_classes > 2:
                    train_auroc = roc_auc_score(
                        y_train_onehot,
                        train_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                else:
                    train_auroc = roc_auc_score(y_train_onehot, train_probs)

            # ------------------------
            # Validation AUROC
            # ------------------------
            with torch.no_grad():
                val_probs_list = []
                y_val_np = y_val.cpu().numpy()

                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i + self.batch_size].to(self.device)
                    logits = self.model(batch_X)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    val_probs_list.append(probs.cpu().numpy())

                val_probs = np.concatenate(val_probs_list, axis=0)

                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                y_val_onehot[np.arange(len(y_val_np)), y_val_np] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_val_onehot,
                        val_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)

            log(
                f"Epoch {epoch+1}/{self.max_iter}: "
                f"Train loss: {train_loss:.8f}, "
                f"Train acc: {train_acc:.4f}, "
                f"Train AUROC: {train_auroc:.4f}, "
                f"Val AUROC: {val_auroc:.4f}",
                priority=3,
                indent=2,
            )

            # ------------------------
            # Early stopping (based on val AUROC)
            # ------------------------
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                log(
                    f"New best model saved with val AUROC: {best_val_auroc:.4f}",
                    priority=3,
                    indent=2,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(
                        f"Early stopping triggered after {epoch+1} epochs",
                        priority=3,
                        indent=2,
                    )
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.best_val_auroc = best_val_auroc
        log(
            f"Training complete. Best validation AUROC: {best_val_auroc:.4f}",
            priority=3,
            indent=2,
        )
        return self

    def predict_proba(self, X):
        """
        Return class probabilities for input X.
        """
        self.model.eval()
        X = torch.FloatTensor(np.asarray(X))

        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size].to(self.device)
                logits = self.model(batch_X)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list, axis=0)

    def predict(self, X):
        """
        Return predicted class indices.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """
        Return accuracy on given data.
        """
        y = np.asarray(y)
        preds = self.predict(X)
        return np.mean(preds == y)

# ==== NEW: MambaClassifier ====
class MambaClassifier:
    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        val_size: float = 0.2,  # kept for backward compatibility, NOT used anymore
        tol: float = 1e-4,
        patience: int = 10,
        d_model: int = 64,
        d_intermediate: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        ssm_cfg=None,
        rms_norm: bool = False,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        """
        Mamba-based classifier for EEG downstream tasks.

        Args:
            d_model: Feature embedding dimension (similar to d_model in Transformer).
            d_intermediate: Hidden dimension of the MLP inside each block.
            n_layers: Number of stacked Mamba blocks.
            ssm_cfg: Configuration dict passed to Mamba, typical keys:
                     - 'layer': 'Mamba2' or 'Mamba1'
                     - 'd_state': state size, e.g., 16 / 32
                     - 'd_conv' : conv kernel size, e.g., 4 / 8
                     - 'expand' : channel expansion factor, e.g., 2
            val_size: kept for compatibility but NOT used anymore
                      (validation set must be provided explicitly to fit()).
        """

        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience

        # For consistency with other baselines we force CUDA here
        self.device = torch.device("cuda")

        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0

        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layers = n_layers
        self.dropout = dropout
        self.rms_norm = rms_norm
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        # Default stable Mamba2 configuration (can be overridden via ssm_cfg)
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {
            "layer": "Mamba2",
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        }

    def _create_model(self, input_shape, n_classes):
        """
        Build the bidirectional Mamba model defined in bidir_mamba.py.

        input_shape: shape of a single sample, e.g. (channels, time) or (channels, freq, time)
        n_classes : number of output classes
        """
        from Neuroprobe.examples.bidir_mamba import Mamba

        return Mamba(
            input_shape=input_shape,
            n_classes=n_classes,
            d_model=self.d_model,
            d_intermediate=self.d_intermediate,
            n_layers=self.n_layers,
            dropout=self.dropout,
            ssm_cfg=self.ssm_cfg,
            rms_norm=self.rms_norm,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32,
            alternating_directions=True,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the Mamba classifier using an explicit validation set.

        Args:
            X_train: training data, shape (N_train, ...)
            y_train: training labels, shape (N_train,)
            X_val:   validation data, shape (N_val, ...)
            y_val:   validation labels, shape (N_val,)
        """

        # Convert numpy → torch
        X_train = torch.FloatTensor(np.asarray(X_train))
        y_train = torch.LongTensor(np.asarray(y_train))
        X_val = torch.FloatTensor(np.asarray(X_val))
        y_val = torch.LongTensor(np.asarray(y_val))

        # Classes and number of classes
        self.classes_ = np.unique(y_train.cpu().numpy())
        n_classes = len(self.classes_)

        log(
            f"Training MambaClassifier with {len(X_train)} train samples and "
            f"{len(X_val)} val samples",
            priority=3,
            indent=2,
        )

        # Build model using input shape (excluding batch dimension)
        input_shape = X_train.shape[1:]
        self.model = self._create_model(input_shape, n_classes).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[MambaClassifier] Trainable params: {n_params} (~{n_params/1e6:.2f}M)")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0

        # ------------------------
        # Training loop
        # ------------------------
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i + self.batch_size].to(self.device)
                batch_y = y_train[i:i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ------------------------
            # Validation
            # ------------------------
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss = criterion(val_outputs, y_val.to(self.device)).item()

                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.cpu().numpy()

                # One-hot labels for multi-class AUROC
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                y_val_onehot[np.arange(len(y_val_np)), y_val_np] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_val_onehot,
                        val_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)

            log(
                f"Epoch {epoch+1}/{self.max_iter}: "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
                f"Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}",
                priority=3,
                indent=2,
            )

            # ------------------------
            # Early stopping based on val AUROC
            # ------------------------
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                log(
                    f"New best model saved with val AUROC: {best_val_auroc:.4f}",
                    priority=3,
                    indent=2,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(
                        f"Early stopping triggered after {epoch+1} epochs",
                        priority=3,
                        indent=2,
                    )
                    break

        # Restore best checkpoint
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.best_val_auroc = best_val_auroc
        log(
            f"Training complete. Best validation AUROC: {best_val_auroc:.4f}",
            priority=3,
            indent=2,
        )
        return self

    def predict_proba(self, X):
        """
        Return class probabilities for input X.
        """
        self.model.eval()
        X = torch.FloatTensor(np.asarray(X))

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        """
        Return predicted class indices.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """
        Return accuracy on given data.
        """
        y = np.asarray(y)
        preds = self.predict(X)
        return np.mean(preds == y)
    
############## REGION AVERAGING (FOR DS/DM SPLITS) ###############

def get_region_labels(subject):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    return subject.get_all_electrode_metadata()['DesikanKilliany'].to_numpy()

def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)
    
    d_model_dimension_unsqueezed = False
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        d_model_dimension_unsqueezed = True

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)

    if d_model_dimension_unsqueezed: # remove the dummy dimension
        X_train_regions = X_train_regions[:, :, :, 0]
        X_test_regions = X_test_regions[:, :, :, 0]
    
    return X_train_regions, X_test_regions, common_regions

class MambaEvalWrapper:
    """
    Wrapper for loading fine-tuned Mamba model in eval_single_electrode.py.
    Note:
    - No training
    - Only performs forward inference
    - Adapts to sklearn-style interface: fit / predict_proba / predict / score
    """
    def __init__(self, ckpt_path, device="cuda"):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        # Load necessary information saved during fine-tuning
        self.in_dim = ckpt["in_dim"]
        self.num_labels = ckpt["num_labels"]
        self.mamba_pretrained = ckpt["mamba_pretrained"]
        from mixer_seq_simple import MambaLMHeadModel
        # Construct model (structure must match fine-tuning setup)
        self.model = EEGMambaForEval(
            mamba_pretrained=self.mamba_pretrained,
            in_dim=self.in_dim,
            num_labels=self.num_labels,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        # Label mapping (eval_utils convention)
        self.classes_ = np.arange(self.num_labels)
    
    def fit(self, X, y):
        """
        No training during evaluation phase, just return self.
        """
        return self
    
    @torch.no_grad()
    def predict_proba(self, X):
        """
        Args:
            X: numpy array of shape (N, C, T) or (N, C, F, T)
        Returns:
            Probabilities of shape (N, num_labels)
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X)      # (N, num_labels)
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=-1)]
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


class EEGMambaForEval(nn.Module):
    """
    Mamba classification model structure consistent with training setup,
    but without training logic. Only used for forward pass during evaluation.
    """
    def __init__(self, mamba_pretrained, in_dim, num_labels):
        super().__init__()

        from Neuroprobe.examples.mixer_seq_simple import MambaLMHeadModel

        self.mamba = MambaLMHeadModel.from_pretrained(mamba_pretrained)
        d_model = self.mamba.config.d_model
        # Input projection + classifier
        self.input_proj = nn.Linear(in_dim, d_model)
        self.classifier = nn.Linear(d_model, num_labels)
        # Freeze all backbone parameters
        for p in self.mamba.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T) or (B, C, F, T)
        Returns:
            logits: Classification logits of shape (B, num_labels)
        """
        # Convert input to sequence format
        if x.ndim == 3:
            # (B, C, T) → (B, T, C)
            x = x.permute(0, 2, 1)
        elif x.ndim == 4:
            # (B, C, F, T) → (B, T, C*F)
            x = x.permute(0, 3, 1, 2)
            B, T, C, F = x.shape
            x = x.reshape(B, T, C * F)
        else:
            raise ValueError("Unexpected X shape")
        
        # Project to model dimension
        h = self.input_proj(x)
        
        # Forward through backbone (skip embedding layer)
        backbone = self.mamba.backbone
        hidden_states = h
        residual = None
        for layer in backbone.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)
        
        # Apply final normalization
        if not backbone.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
        else:
            from Neuroprobe.examples.mixer_seq_simple import layer_norm_fn
            hidden_states = layer_norm_fn(
                hidden_states,
                backbone.norm_f.weight,
                backbone.norm_f.bias,
                eps=backbone.norm_f.eps,
            )
        
        # Pool and classify
        pooled = hidden_states.mean(dim=1)  # Average pooling over time
        logits = self.classifier(pooled)
        return logits