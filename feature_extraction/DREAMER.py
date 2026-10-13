import numpy as np
import pickle
import scipy.io
from scipy import stats
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.fft import fft
import warnings
from scipy import signal
import neurokit2 as nk
np.random.seed(2025)
warnings.filterwarnings('ignore')
# Valence dimension by default. For Arousal, use score_arousal instead of score_valence in the neutral-rating check
# and in the call to process_subject_data() in __main__.

def process_signal_segments(signal_data, feature_extractor, fs=128, **kwargs):
   
    num_samples, num_channels, _ = signal_data.shape
    all_features = []

    for i in range(num_samples):
        sample_features = []
        
        for ch in range(num_channels):
            signal_segment = signal_data[i, ch, :]
            features = feature_extractor(signal_segment, fs=fs, **kwargs)
            sample_features.extend(features)
        
        all_features.append(sample_features)

    return np.array(all_features)
# 1. EEG
def extract_eeg_features(eeg_signal, fs=128, normalize=True):
    eeg_signal = np.nan_to_num(eeg_signal)
    
    if len(eeg_signal) == 0 or np.all(eeg_signal == 0) or np.isnan(eeg_signal).all():
        return [0.0, 0.0, 0.0, 0.0]
    eeg_signal = signal.detrend(eeg_signal)
    b, a = signal.butter(4, [1, 48], btype='bandpass', fs=fs)
    eeg_signal = signal.filtfilt(b, a, eeg_signal)
    
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 47)
    }
    try:
        nperseg = len(eeg_signal)
        f, Pxx_den = signal.welch(eeg_signal, fs=fs, nperseg=nperseg, 
                                 window='hann', scaling='density')
        band_powers = []
        total_power = 0
        for band_name, (low, high) in bands.items():
            band_mask = (f >= low) & (f <= high)
            if np.sum(band_mask) > 0:
                band_power = np.trapz(Pxx_den[band_mask], f[band_mask])
            else:
                band_power = 0.0
            band_powers.append(band_power)
            total_power += band_power
        if total_power > 0:
            band_powers = [(power / total_power) * 100 for power in band_powers]
        band_powers = [np.log1p(power) for power in band_powers] 
        
        return band_powers
        
    except Exception as e:
        return [0.0, 0.0, 0.0, 0.0]
# 2. ECG
def extract_ecg_features(ecg_signal, fs=256):
    ecg_signal = np.nan_to_num(ecg_signal)
    if len(ecg_signal) == 0 or np.all(ecg_signal == 0) or np.isnan(ecg_signal).all():
        return [0.0, 0.0]
    try:
        ecg_signal = signal.detrend(ecg_signal)
        b, a = signal.butter(4, [0.5, 40], btype='bandpass', fs=fs)
        ecg_signal = signal.filtfilt(b, a, ecg_signal)
    except:
        pass
    try:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        if len(rpeaks["ECG_R_Peaks"]) > 1:
            rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs * 1000
            mean_hr = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        else:
            mean_hr = 0.0
    except:
        mean_hr = 0.0
    try:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        if len(rpeaks["ECG_R_Peaks"]) > 2:
            rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs * 1000
            sdnn = np.std(rr_intervals)
        else:
            sdnn = 0.0
    except:
        sdnn = 0.0
    return [mean_hr, sdnn]
# Overall feature extraction function
def extract_all_features(data_dict, fs=128, baseline_correction=True, window_size=1):
    """
    Extract features of all signal types
    """
    all_features = {}
    feature_extractors = {
        'modality1': (extract_eeg_features, 4),  # EEG
        'modality2': (extract_ecg_features, 2)   # ECG
    }
    baseline_features_dict = {}
    if baseline_correction and 'baseline' in data_dict:
        for signal_name, baseline_data in data_dict['baseline'].items():
            if signal_name in feature_extractors:
                extractor, num_features = feature_extractors[signal_name]
                signal_fs = 128 if signal_name == 'modality1' else 256  # EEG: 128Hz, ECG: 256Hz
                baseline_samples_per_window = window_size * signal_fs
                if baseline_data.shape[2] >= baseline_samples_per_window:
                    last_baseline_data = baseline_data[:, :, -baseline_samples_per_window:]
                    baseline_features = process_signal_segments(last_baseline_data, extractor, fs=signal_fs)
                    if len(baseline_features) > 0:
                        baseline_features_dict[signal_name] = np.mean(baseline_features, axis=0)
                    else:
                        baseline_features_dict[signal_name] = np.zeros(num_features)
                else:

                    baseline_features = process_signal_segments(baseline_data, extractor, fs=signal_fs)
                    if len(baseline_features) > 0:
                        baseline_features_dict[signal_name] = np.mean(baseline_features, axis=0)
                    else:
                        baseline_features_dict[signal_name] = np.zeros(num_features)

    for signal_name, signal_data in data_dict.items():
        if signal_name in feature_extractors and signal_name != 'baseline':
            extractor, num_features = feature_extractors[signal_name]
            signal_fs = 128 if signal_name == 'modality1' else 256  # EEG: 128Hz, ECG: 256Hz
            experiment_features = process_signal_segments(signal_data, extractor, fs=signal_fs)
            
            if baseline_correction and signal_name in baseline_features_dict:
                baseline_mean = baseline_features_dict[signal_name]
                if baseline_mean.shape[0] == experiment_features.shape[1]:
                    corrected_features = experiment_features - baseline_mean
                else:
                    corrected_features = experiment_features
                
                all_features[signal_name] = corrected_features
                print(f"{signal_name} feature: {experiment_features.shape} -> corrected: {corrected_features.shape}")
            else:
                all_features[signal_name] = experiment_features
                print(f"{signal_name} feature: {experiment_features.shape}")
    return all_features

# Data preprocessing function
def data_preprocess(eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, baseline_duration=3, 
                   eeg_sample_rate=128, ecg_sample_rate=256):
    """
    Data preprocessing: Separating baseline data and experimental data
    """
    
    eeg_baseline_samples = baseline_duration * eeg_sample_rate
    ecg_baseline_samples = baseline_duration * ecg_sample_rate
    
    eeg_baseline_processed = eeg_baseline[-eeg_baseline_samples:] if len(eeg_baseline) >= eeg_baseline_samples else eeg_baseline
    ecg_baseline_processed = ecg_baseline[-ecg_baseline_samples:] if len(ecg_baseline) >= ecg_baseline_samples else ecg_baseline

    eeg_experiment_processed = eeg_stimuli
    ecg_experiment_processed = ecg_stimuli
    
    return (eeg_baseline_processed, eeg_experiment_processed, 
            ecg_baseline_processed, ecg_experiment_processed)
def process_subject_data(eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, score_valence, 
                        window_size, eeg_sample_rate=128, ecg_sample_rate=256):
    """
    Process individual subject data, separate baseline and experimental data
    """
    # data preprocessing
    eeg_baseline_proc, eeg_experiment_proc, ecg_baseline_proc, ecg_experiment_proc = data_preprocess(
        eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, 
        baseline_duration=window_size,  
        eeg_sample_rate=eeg_sample_rate, ecg_sample_rate=ecg_sample_rate
    )
    # Create label
    label_val = 0 if score_valence[0][0] < 3 else 1 # Label from the rating passed in (valence by default)
    eeg_window_length = window_size * eeg_sample_rate
    ecg_window_length = window_size * ecg_sample_rate
    
    # Process EEG baseline data (the last window_size s has been captured through data_preprocessing)
    eeg_baseline_windows = []
    if len(eeg_baseline_proc) >= eeg_window_length:
        window = eeg_baseline_proc
        eeg_baseline_windows.append(window)
 
    ecg_baseline_windows = []
    if len(ecg_baseline_proc) >= ecg_window_length:
        window = ecg_baseline_proc
        ecg_baseline_windows.append(window)
    
    eeg_experiment_windows = []
    num_eeg_windows = len(eeg_experiment_proc) // eeg_window_length
    for i in range(num_eeg_windows):
        start_idx = i * eeg_window_length
        end_idx = start_idx + eeg_window_length
        if end_idx <= len(eeg_experiment_proc):
            window = eeg_experiment_proc[start_idx:end_idx]
            eeg_experiment_windows.append(window)
    
    ecg_experiment_windows = []
    num_ecg_windows = len(ecg_experiment_proc) // ecg_window_length
    for i in range(num_ecg_windows):
        start_idx = i * ecg_window_length
        end_idx = start_idx + ecg_window_length
        if end_idx <= len(ecg_experiment_proc):
            window = ecg_experiment_proc[start_idx:end_idx]
            ecg_experiment_windows.append(window)
    
    min_windows = min(len(eeg_experiment_windows), len(ecg_experiment_windows))
    eeg_experiment_windows = eeg_experiment_windows[:min_windows]
    ecg_experiment_windows = ecg_experiment_windows[:min_windows]
    
    labels = [label_val] * min_windows
    
    return (eeg_baseline_windows, eeg_experiment_windows, 
            ecg_baseline_windows, ecg_experiment_windows, labels)

def create_complete_dataset(all_modality_baseline, all_modality_experiment, all_labels, window_size, sample_rates):
    """
    Create a complete dataset, including baseline calibration
    """
    data_dict = {}
    
    modality_names = ['modality1', 'modality2']  # EEG and ECG
    baseline_dict = {}
    for i, name in enumerate(modality_names):
        if len(all_modality_baseline[i]) > 0:
            data_array = np.array(all_modality_baseline[i])
            if i == 0:  # EEG
                data_array = data_array.transpose(0, 2, 1)
            else:  # ECG
                data_array = data_array.transpose(0, 2, 1)
            baseline_dict[name] = data_array
    data_dict['baseline'] = baseline_dict
    
    # Prepare experimental data
    for i, name in enumerate(modality_names):
        if len(all_modality_experiment[i]) > 0:
            data_array = np.array(all_modality_experiment[i])
            if i == 0:  # EEG
                data_array = data_array.transpose(0, 2, 1)
            else:  # ECG
                data_array = data_array.transpose(0, 2, 1)
            data_dict[name] = data_array
    features = extract_all_features(data_dict, fs=128, baseline_correction=True, window_size=window_size)
    features['label'] = np.array(all_labels).reshape(-1, 1)
    features['sample_ids'] = np.arange(len(all_labels)).reshape(-1, 1)
    
    return features

def save_complete_dataset(dataset, output_path):
    """
    Save the complete dataset to a file
    """
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Total sample size: {len(dataset['label'])}")
    for key, value in dataset.items():
        if key not in ['label', 'sample_ids']:
            print(f"{key}: {value.shape}")
if __name__ == '__main__':
    window_size = 3  
    eeg_sample_rate = 128
    ecg_sample_rate = 256
    txt_path = '.txt'
    output_path = '.pkl'
    
    all_modality_baseline = [[] for _ in range(2)]  # EEG and ECG
    all_modality_experiment = [[] for _ in range(2)]
    all_labels = []

    with open(txt_path, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    # Process each participant
    total_subjects = len(file_list)
    print(f"Start processing data for {total_subjects} subjects...")
    print(f"Window size: {window_size}s")
    print(f"EEG sampling rate: {eeg_sample_rate}Hz, ECGsampling rate: {ecg_sample_rate}Hz")
    
    for i, mat_path in enumerate(file_list):
        print(f"Process the {i+1}/{total_subjects} th subject: {mat_path}")
        try:
            mat_cont = scipy.io.loadmat(mat_path)
            mat_data = mat_cont['combined_data']
            for j in range(mat_data.shape[0]):
                eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, score_valence, score_arousal,score_dominance= mat_data[j]
                if score_valence[0][0] == 3: # Skip neutral ratings (3). For Arousal, use score_arousal[0][0]
                    continue
                (eeg_baseline_win, eeg_experiment_win, 
                 ecg_baseline_win, ecg_experiment_win, labels) = process_subject_data(
                    eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, score_valence,  # For Arousal, pass score_arousal
                    window_size, eeg_sample_rate, ecg_sample_rate
                )
                if len(eeg_baseline_win) > 0:
                    all_modality_baseline[0].extend(eeg_baseline_win)
                if len(ecg_baseline_win) > 0:
                    all_modality_baseline[1].extend(ecg_baseline_win)
                if len(eeg_experiment_win) > 0:
                    all_modality_experiment[0].extend(eeg_experiment_win)
                if len(ecg_experiment_win) > 0:
                    all_modality_experiment[1].extend(ecg_experiment_win)
                
                all_labels.extend(labels)
                
        except Exception as e:
            print(f"Error occurred while processing file {mat_path}: {e}")
            continue
    
    for i, data in enumerate(all_modality_baseline):
        if len(data) > 0:
            data_array = np.array(data)
            print(f"modal {i+1} baseline: {data_array.shape}")

    for i, data in enumerate(all_modality_experiment):
        if len(data) > 0:
            data_array = np.array(data)
            print(f"modal {i+1} experiment: {data_array.shape}")
    
    print(f"\nStart extracting features (using the last{window_size} seconds of the baseline period for correction)...")
    complete_dataset = create_complete_dataset(
        all_modality_baseline, all_modality_experiment, all_labels, 
        window_size, {'eeg': eeg_sample_rate, 'ecg': ecg_sample_rate}
    )
    save_complete_dataset(complete_dataset, output_path)
    # Final statistical information
    print(f"Window size: {window_size}s")
    print(f"Total sample size: {len(complete_dataset['label'])}")
    if len(complete_dataset['label']) > 0:
        print(f"Positive sample size (arousal > 3): {np.sum(complete_dataset['label'] == 1)}")
        print(f"Negative sample size (arousal < 3): {np.sum(complete_dataset['label'] == 0)}")
    print(f"Data saving path: {output_path}")
