import numpy as np
import pickle
import os
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.fft import fft
import warnings
from scipy import signal
import pywt

np.random.seed(2025)
warnings.filterwarnings('ignore')
""" Binary classification (non-stress vs. stress) by default. For ternary classification (baseline / stress / amusement),
    replace the label mapping in process_wesad_data() with the following:
    # tag mapping
    label_mapped = []
    for i in range(len(label)):
        if label[i] == 1:
            label_mapped.append(0)  
        elif label[i] == 2:
            label_mapped.append(1) 
        elif label[i] == 3:
            label_mapped.append(2)  
        else:
            label_mapped.append(-1)  
    """
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

# 1. BVP
def extract_bvp_features(pleth_signal, fs=128):
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

# 2. EDA
def extract_eda_features(eda_signal, fs=4):
   
    eda_signal = np.nan_to_num(eda_signal)
    if len(eda_signal) == 0 or np.all(eda_signal == 0):
        return [0.0, 0.0]
    mean_val = np.mean(eda_signal)
    diff_val = np.mean(np.abs(np.diff(eda_signal)))
    return [mean_val, diff_val]
# 3.TEMP
def extract_temp_features(temp_signal, fs=4):
    temp_signal = np.nan_to_num(temp_signal)
    if len(temp_signal) == 0:
        return [0.0, 0.0]
    
    mean_temp = np.mean(temp_signal)
    range_temp = np.max(temp_signal) - np.min(temp_signal)
    return [mean_temp, range_temp]
# Overall feature extraction function
def extract_all_features(data_dict, fs=128):
    
    all_features = {}
    feature_extractors = {
        'bvp': (extract_bvp_features, 2),  # BVP
        'eda': (extract_eda_features, 2),   # EDA
        'temp': (extract_temp_features, 2)  # TEMP
    }
    
    for signal_name, signal_data in data_dict.items():
        if signal_name in feature_extractors:
            extractor, expected_features = feature_extractors[signal_name]
            if signal_name == 'bvp':
                signal_fs = 64
            else:  # eda, temp
                signal_fs = 4
            features = process_signal_segments(signal_data, extractor, fs=signal_fs)
            all_features[signal_name] = features
            print(f"{signal_name} features: {features.shape}")
    
    return all_features

def process_wesad_data(pkl_data, window_seconds=10):
    """
    Process WESAD data and extract wrist BVP, EDA, and TEMP signals
    """
    bvp_signal = pkl_data[b'signal'][b'wrist'][b'BVP']  # 64 Hz
    eda_signal = pkl_data[b'signal'][b'wrist'][b'EDA']  # 4 Hz
    temp_signal = pkl_data[b'signal'][b'wrist'][b'TEMP']  # 4 Hz
    label = pkl_data[b'label']  # 700 Hz
    print(f"Original signal shape - BVP: {bvp_signal.shape}, EDA: {eda_signal.shape}, TEMP: {temp_signal.shape}")
    
    label_mapped = []
    for i in range(len(label)):
        if label[i] == 1:
            label_mapped.append(0)  
        elif label[i] == 2:
            label_mapped.append(1)  
        elif label[i] == 3:
            label_mapped.append(0)  
        else:
            label_mapped.append(-1)  
    label_mapped = np.array(label_mapped)
    
    """ 
    # tag mapping
    label_mapped = []
    for i in range(len(label)):
        if label[i] == 1:
            label_mapped.append(0)  
        elif label[i] == 2:
            label_mapped.append(1) 
        elif label[i] == 3:
            label_mapped.append(2)  
        else:
            label_mapped.append(-1)  
    """

    bvp_segments = []
    eda_segments = []
    temp_segments = []
    label_segments = []
    
    bvp_points_per_window = 64 * window_seconds
    eda_points_per_window = 4 * window_seconds
    temp_points_per_window = 4 * window_seconds
    label_points_per_window = 700 * window_seconds
    
    total_seconds = len(label) // 700
    num_windows = total_seconds // window_seconds
    
    for window_idx in range(num_windows):
        start_second = window_idx * window_seconds
        start_sample_label = start_second * 700
        label_segment = label_mapped[start_sample_label:start_sample_label + label_points_per_window]
        unique, counts = np.unique(label_segment, return_counts=True)
        main_label = unique[np.argmax(counts)]
        if main_label == -1:
            continue
        bvp_start = start_second * 64
        eda_start = start_second * 4
        temp_start = start_second * 4
        if (bvp_start + bvp_points_per_window > len(bvp_signal) or
            eda_start + eda_points_per_window > len(eda_signal) or
            temp_start + temp_points_per_window > len(temp_signal)):
            continue
        bvp_segment = bvp_signal[bvp_start:bvp_start + bvp_points_per_window]
        eda_segment = eda_signal[eda_start:eda_start + eda_points_per_window]
        temp_segment = temp_signal[temp_start:temp_start + temp_points_per_window]
        
        bvp_segments.append(bvp_segment)
        eda_segments.append(eda_segment)
        temp_segments.append(temp_segment)
        label_segments.append(main_label)
    return bvp_segments, eda_segments, temp_segments, label_segments

def create_complete_dataset(all_bvp, all_eda, all_temp, all_labels, window_seconds=10):
    """
    Create a complete dataset
    """
    bvp_array = np.array(all_bvp).reshape(len(all_bvp), 1, -1)
    eda_array = np.array(all_eda).reshape(len(all_eda), 1, -1)
    temp_array = np.array(all_temp).reshape(len(all_temp), 1, -1)
    
    print(f"Data Shape - BVP: {bvp_array.shape}, EDA: {eda_array.shape}, TEMP: {temp_array.shape}")
    data_dict = {
        'bvp': bvp_array,
        'eda': eda_array,
        'temp': temp_array
    }
    
    # extract features
    features = extract_all_features(data_dict)
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

    window_seconds = 3  
    txt_path = '.txt'
    output_path = '.pkl'

    all_bvp = []
    all_eda = []
    all_temp = []
    all_labels = []
 
    with open(txt_path, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    # Process each subject.
    total_subjects = len(file_list)
    print(f"Start processing the WESAD wrist data of {total_subjects} subjects...")
    print(f"time window: {window_seconds}s")
    print(f"Signal sampling rate: BVP=64Hz, EDA=4Hz, TEMP=4Hz")
    
    for i, pkl_path in enumerate(file_list):
        
        try:
            with open(pkl_path, 'rb') as pkl_file:
                pkl_data = pickle.load(pkl_file, encoding='bytes')
            
            # Processing WESAD data
            bvp_segments, eda_segments, temp_segments, labels = process_wesad_data(pkl_data, window_seconds)
            all_bvp.extend(bvp_segments)
            all_eda.extend(eda_segments)
            all_temp.extend(temp_segments)
            all_labels.extend(labels)  
        except Exception as e:
            continue
    print(f"\nData statistics:")
    print(f"BVP samples: {len(all_bvp)}")
    print(f"EDA samples: {len(all_eda)}")
    print(f"TEMP samples: {len(all_temp)}")
    print(f"Labels: {len(all_labels)}")
    if len(all_bvp) == 0:
        exit(1)
    complete_dataset = create_complete_dataset(all_bvp, all_eda, all_temp, all_labels, window_seconds)
    
    # Save the dataset
    save_complete_dataset(complete_dataset, output_path)

    # Final statistics
    print("\n=== Data Processing Completed ===")
    print(f"Time window: {window_seconds} seconds")
    print(f"Total number of samples: {len(complete_dataset['label'])}")

    if len(complete_dataset['label']) > 0:
        label_counts = np.bincount(complete_dataset['label'].flatten())
        print("Label distribution:")
        for label_val, count in enumerate(label_counts):
            if label_val == 0:
                print(f"  Non-stress (0): {count} samples")
            elif label_val == 1:
                print(f"  Stress (1): {count} samples")
            elif label_val == 2:
                print(f"  Amusement (2): {count} samples")

    print(f"Dataset saved to: {output_path}")
