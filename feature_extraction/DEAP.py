import numpy as np
import pickle
import scipy.io
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.fft import fft
import warnings
from scipy import signal
np.random.seed(2025)
warnings.filterwarnings('ignore')

# Valence dimension by default. For Arousal, replace mat_label1[0] with mat_label1[1]
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


# 2. EOG
def extract_eog_features(eog_signal, fs=128):
    eog_signal = np.nan_to_num(eog_signal)
    if len(eog_signal) == 0 or np.all(eog_signal == 0):
        return [0.0, 0.0]
    
    mean_amp = np.mean(np.abs(eog_signal))
    
    try:
        threshold = np.mean(np.abs(eog_signal)) + 2 * np.std(np.abs(eog_signal))
        blink_positions, _ = find_peaks(np.abs(eog_signal), height=threshold, distance=fs//2)
        blink_rate = len(blink_positions) / (len(eog_signal) / fs) * 60
    except:
        blink_rate = 0.0
    
    return [mean_amp, blink_rate]

# 3. EMG
def extract_emg_features(emg_signal, fs=128):
    emg_signal = np.nan_to_num(emg_signal)
    if len(emg_signal) == 0:
        return [0.0, 0.0]
    
    rms = np.sqrt(np.mean(emg_signal**2))
    
    try:
        f, psd = welch(emg_signal, fs=fs, nperseg=min(fs*2, len(emg_signal)))
        dominant_freq = f[np.argmax(psd)] if len(psd) > 0 else 0.0
    except:
        dominant_freq = 0.0
    
    return [rms, dominant_freq]

# 4. GSR
def extract_gsr_features(gsr_signal, fs=128):
    gsr_signal = np.nan_to_num(gsr_signal)
    if len(gsr_signal) == 0:
        return [0.0, 0.0]
    
    mean_val = np.mean(gsr_signal)
    diff_val = np.mean(np.abs(np.diff(gsr_signal)))
    return [mean_val, diff_val]

# 5. Respiration
def extract_respiration_features(resp_signal, fs=128):
    resp_signal = np.nan_to_num(resp_signal)
    if len(resp_signal) == 0:
        return [0.0, 0.0]
    
    try:
        peaks, _ = find_peaks(resp_signal, distance=fs//3, height=np.mean(resp_signal))
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            resp_rate = 60 / np.mean(rr_intervals)
        else:
            resp_rate = 0.0
    except:
        resp_rate = 0.0
    
    amplitude = np.max(resp_signal) - np.min(resp_signal)
    return [resp_rate, amplitude]

# 6. Plethysmograph
def extract_plethysmograph_features(pleth_signal, fs=128):
    pleth_signal = np.nan_to_num(pleth_signal)
    if len(pleth_signal) == 0:
        return [0.0, 0.0]
    
    try:
        peaks, _ = find_peaks(pleth_signal, distance=fs*0.5, height=np.mean(pleth_signal))
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            heart_rate = 60 / np.mean(rr_intervals)
            hrv = np.std(rr_intervals)
        else:
            heart_rate, hrv = 0.0, 0.0
    except:
        heart_rate, hrv = 0.0, 0.0
    
    return [heart_rate, hrv]

# 7. Temperature
def extract_temperature_features(temp_signal, fs=128):
    temp_signal = np.nan_to_num(temp_signal)
    if len(temp_signal) == 0:
        return [0.0, 0.0]
    
    mean_temp = np.mean(temp_signal)
    range_temp = np.max(temp_signal) - np.min(temp_signal)
    return [mean_temp, range_temp]

# Overall feature extraction function
def extract_all_features(data_dict, fs=128, baseline_correction=True, window_size=1):
    all_features = {}
    
    feature_extractors = {
        'modality1': (extract_eeg_features, 4),
        'modality2': (extract_eog_features, 2),
        'modality3': (extract_emg_features, 2),
        'modality4': (extract_gsr_features, 2),
        'modality5': (extract_respiration_features, 2),
        'modality6': (extract_plethysmograph_features, 2),
        'modality7': (extract_temperature_features, 2)
    }
    baseline_features_dict = {}
    if baseline_correction and 'baseline' in data_dict:
        for signal_name, baseline_data in data_dict['baseline'].items():
            if signal_name in feature_extractors:
                extractor, num_features = feature_extractors[signal_name]
                baseline_samples_per_window = window_size * fs
                if baseline_data.shape[2] >= baseline_samples_per_window:
                 
                    last_baseline_data = baseline_data[:, :, -baseline_samples_per_window:]
                    
                    # extract features
                    baseline_features = process_signal_segments(last_baseline_data, extractor, fs=fs)
                    
                    if len(baseline_features) > 0:
                        baseline_features_dict[signal_name] = np.mean(baseline_features, axis=0)
                    else:
                        baseline_features_dict[signal_name] = np.zeros(num_features)
                else:
                    baseline_features = process_signal_segments(baseline_data, extractor, fs=fs)
                    if len(baseline_features) > 0:
                        baseline_features_dict[signal_name] = np.mean(baseline_features, axis=0)
                    else:
                        baseline_features_dict[signal_name] = np.zeros(num_features)
    
    for signal_name, signal_data in data_dict.items():
        if signal_name in feature_extractors and signal_name != 'baseline':
            extractor, num_features = feature_extractors[signal_name]
            
            experiment_features = process_signal_segments(signal_data, extractor, fs=fs)
            
            if baseline_correction and signal_name in baseline_features_dict:
                baseline_mean = baseline_features_dict[signal_name]
                corrected_features = experiment_features - baseline_mean
                
                all_features[signal_name] = corrected_features
                print(f"{signal_name} feature: {experiment_features.shape} -> corrected: {corrected_features.shape}")
            else:
                all_features[signal_name] = experiment_features
                print(f"{signal_name} feature: {experiment_features.shape}")
    
    return all_features

# Data preprocessing function
def data_preprocess(mat_data, baseline_duration=3, experiment_duration=60, sample_rate=128):
    """
    Data preprocessing: Separate baseline data for the first 3 seconds and experimental data for the last 60 seconds
    """
    baseline_samples = baseline_duration * sample_rate
    experiment_samples = experiment_duration * sample_rate
    
    baseline_data = mat_data[:baseline_samples]
    experiment_data = mat_data[baseline_samples:baseline_samples+experiment_samples]

    return baseline_data, experiment_data

def process_subject_data(mat_data1, mat_label1, window_size, sample_rate):
    """
    Process individual subject data, separate baseline and experimental data
    """
    modality_baseline = [[] for _ in range(7)]
    modality_experiment = [[] for _ in range(7)]
    label_data = []

    # data preprocessing
    baseline_data, experiment_data = data_preprocess(mat_data1, baseline_duration=3, 
                                                   experiment_duration=60, sample_rate=sample_rate)
    for k in range(baseline_data.shape[0]):
        modality_baseline[0].append(baseline_data[k][:32])    # EEG
        modality_baseline[1].append(baseline_data[k][32:34])  # EOG
        modality_baseline[2].append(baseline_data[k][34:36])  # EMG
        modality_baseline[3].append([baseline_data[k][36]])    # GSR
        modality_baseline[4].append([baseline_data[k][37]])    # Respiration
        modality_baseline[5].append([baseline_data[k][38]])    # Plethysmograph
        modality_baseline[6].append([baseline_data[k][39]])    # Temperature

 
    for k in range(experiment_data.shape[0]):
        modality_experiment[0].append(experiment_data[k][:32])    # EEG
        modality_experiment[1].append(experiment_data[k][32:34])  # EOG
        modality_experiment[2].append(experiment_data[k][34:36])  # EMG
        modality_experiment[3].append([experiment_data[k][36]])    # GSR
        modality_experiment[4].append([experiment_data[k][37]])    # Respiration
        modality_experiment[5].append([experiment_data[k][38]])    # Plethysmograph
        modality_experiment[6].append([experiment_data[k][39]])    # Temperature


    windowed_baseline = [[] for _ in range(7)]
    baseline_window_length = window_size * sample_rate
    
    for mod_idx in range(7):
        data_length = len(modality_baseline[mod_idx])
       
        start_idx = max(0, data_length - baseline_window_length)
        if start_idx + baseline_window_length <= data_length:
            window = modality_baseline[mod_idx][start_idx:start_idx + baseline_window_length]
            windowed_baseline[mod_idx].append(window)


    windowed_experiment = [[] for _ in range(7)]
    experiment_window_length = window_size * sample_rate
    
    data_length = len(modality_experiment[0])
    for i in range(0, data_length, experiment_window_length):
        for mod_idx in range(7):
            if i + experiment_window_length <= len(modality_experiment[mod_idx]):
                window = modality_experiment[mod_idx][i:i + experiment_window_length]
                windowed_experiment[mod_idx].append(window)
        # Valence dimension by default. For Arousal, replace mat_label1[0] with mat_label1[1]
        label_val = 0 if mat_label1[0] <= 5 else 1  
        label_data.append(label_val)

    return windowed_baseline, windowed_experiment, label_data

def create_complete_dataset(all_modality_baseline, all_modality_experiment, all_labels, window_size, sample_rate):
    """
    Create a complete dataset, including baseline calibration
    """
    data_dict = {}
    
    modality_names = ['modality1', 'modality2', 'modality3', 'modality4', 
                     'modality5', 'modality6', 'modality7']
    baseline_dict = {}
    for i, name in enumerate(modality_names):
        if len(all_modality_baseline[i]) > 0:
            data_array = np.array(all_modality_baseline[i])
            
            if i == 0:  # EEG
                data_array = data_array.transpose(0, 2, 1)
            elif i in [1, 2]:  # EOG, EMG
                data_array = data_array.transpose(0, 2, 1)
            else:  
                data_array = data_array.reshape(len(data_array), 1, -1)
            
            baseline_dict[name] = data_array
    data_dict['baseline'] = baseline_dict
    
    # Prepare experimental data
    for i, name in enumerate(modality_names):
        if len(all_modality_experiment[i]) > 0:
            data_array = np.array(all_modality_experiment[i])
            
            if i == 0:  # EEG
                data_array = data_array.transpose(0, 2, 1)
            elif i in [1, 2]:  # EOG, EMG
                data_array = data_array.transpose(0, 2, 1)
            else:  
                data_array = data_array.reshape(len(data_array), 1, -1)
            data_dict[name] = data_array
    # Extract features (including baseline correction, using the baseline of the last window_size s)
    features = extract_all_features(data_dict, fs=sample_rate, baseline_correction=True, window_size=window_size)
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
    sample_rate = 128
    txt_path = '.txt'
    output_path = '.pkl'
    
    if window_size not in [1, 2, 3]:
        print("Warning: It is recommended to set the window size to 1, 2, or 3 seconds.")
    all_modality_baseline = [[] for _ in range(7)]
    all_modality_experiment = [[] for _ in range(7)]
    all_labels = []
    with open(txt_path, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    # Process each participant
    total_subjects = len(file_list)
    print(f"Start processing data for {total_subjects} subjects...")
    print(f"Window size: {window_size}s, sampling rate: {sample_rate}Hz")
    print(f"Baseline correction use: data from the last{window_size}s of the baseline period")
    
    for i, mat_path in enumerate(file_list):
        print(f"Process the {i+1}/{total_subjects} th subject: {mat_path}")
        
        try:
            mat_cont = scipy.io.loadmat(mat_path)
            mat_data = np.transpose(mat_cont['data'], (0, 2, 1))
            mat_label = mat_cont['labels']
            
            for j in range(mat_data.shape[0]):
                trial_data = mat_data[j]
                trial_label = mat_label[j]
                
                modality_baseline, modality_experiment, labels = process_subject_data(trial_data, trial_label, window_size, sample_rate)
                
                for mod_idx in range(7):
                    if len(modality_baseline[mod_idx]) > 0:
                        all_modality_baseline[mod_idx].extend(modality_baseline[mod_idx])
                    if len(modality_experiment[mod_idx]) > 0:
                        all_modality_experiment[mod_idx].extend(modality_experiment[mod_idx])
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
    complete_dataset = create_complete_dataset(all_modality_baseline, all_modality_experiment, all_labels, window_size, sample_rate)
    
    save_complete_dataset(complete_dataset, output_path)
    
    # Final statistical information
    print(f"Window size: {window_size}s")
    print(f"Total sample size: {len(complete_dataset['label'])}")
    if len(complete_dataset['label']) > 0:
        print(f"Positive sample size (arousal > 5): {np.sum(complete_dataset['label'] == 1)}")
        print(f"Negative sample size (arousal <= 5): {np.sum(complete_dataset['label'] == 0)}")
    print(f"Data saving path: {output_path}")