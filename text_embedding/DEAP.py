import os
import json
import pickle
import numpy as np


# === Definition of EEG channel names ===
EEG_channels_geneva = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                      'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

# === Load data ===
with open('.pkl', 'rb') as f:
    data = pickle.load(f)

features = [data['modality1'], data['modality2'], data['modality3'],
            data['modality4'], data['modality5'], data['modality6'],
            data['modality7']]

labels = data['label']
num_samples = len(labels)

# Check the actual feature dimensions
feature_names = ['EEG', 'EOG', 'EMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature']
for i, (name, feature_array) in enumerate(zip(feature_names, features)):
    if feature_array is not None and len(feature_array) > 0:
        print(f"{name}: shape {feature_array.shape} -> {feature_array.shape[1]} features per sample")
    else:
        print(f"{name}: no data")

# === Adjust formatting functions based on actual feature dimensions ===
def format_eeg_features(eeg_feature_vector, channels):
    """Format EEG features - 4 frequency-band features per channel"""
    features_per_channel = 4  # Theta, Alpha, Beta, Gamma
    if len(eeg_feature_vector) != len(channels) * features_per_channel:
        actual_features_per_channel = len(eeg_feature_vector) // len(channels)
        features_per_channel = actual_features_per_channel
    formatted_features = []
    for i, channel in enumerate(channels):
        start_idx = i * features_per_channel
        end_idx = start_idx + features_per_channel
        if end_idx <= len(eeg_feature_vector):
            channel_features = eeg_feature_vector[start_idx:end_idx]
            feature_str = (
                f"{channel}: "
                f"Theta={channel_features[0]:.4f}, "
                f"Alpha={channel_features[1]:.4f}, "
                f"Beta={channel_features[2]:.4f}, "
                f"Gamma={channel_features[3]:.4f}"
            )
            formatted_features.append(feature_str)
    
    return formatted_features

def format_eog_features(eog_feature_vector):
    num_features = len(eog_feature_vector)
    if num_features >= 4:
        return f"EOG: Channel1 Mean Amplitude={eog_feature_vector[0]:.2f}, Channel1 Blink Rate={eog_feature_vector[1]:.2f}, Channel2 Mean Amplitude={eog_feature_vector[2]:.2f}, Channel2 Blink Rate={eog_feature_vector[3]:.2f}"
    elif num_features >= 2:
        return f"EOG: Mean Amplitude={eog_feature_vector[0]:.2f}, Blink Rate={eog_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(eog_feature_vector)])
        return f"EOG: {features_str}"

def format_emg_features(emg_feature_vector):
    num_features = len(emg_feature_vector)
    if num_features >= 4:
        return f"EMG: Channel1 Root Mean Square={emg_feature_vector[0]:.2f}, Channel1 Dominant Frequency={emg_feature_vector[1]:.2f}, Channel2 Root Mean Square={emg_feature_vector[2]:.2f}, Channel2 Dominant Frequency={emg_feature_vector[3]:.2f}"
    elif num_features >= 2:
        return f"EMG: Root Mean Square={emg_feature_vector[0]:.2f}, Dominant Frequency={emg_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(emg_feature_vector)])
        return f"EMG: {features_str}"
def format_gsr_features(gsr_feature_vector):
    num_features = len(gsr_feature_vector)
    if num_features >= 2:
        return f"GSR: Mean Conductance={gsr_feature_vector[0]:.2f}, Mean Absolute Derivative={gsr_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(gsr_feature_vector)])
        return f"GSR: {features_str}"

def format_respiration_features(resp_feature_vector):
    num_features = len(resp_feature_vector)
    if num_features >= 2:
        return f"RESP: Respiration Rate={resp_feature_vector[0]:.2f}, Amplitude Range={resp_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(resp_feature_vector)])
        return f"RESP: {features_str}"

def format_plethysmograph_features(pleth_feature_vector):
    num_features = len(pleth_feature_vector)
    if num_features >= 2:
        return f"PPG: Heart Rate={pleth_feature_vector[0]:.2f}, Heart Rate Variability Standard Deviation={pleth_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(pleth_feature_vector)])
        return f"PPG: {features_str}"

def format_temperature_features(temp_feature_vector):
    num_features = len(temp_feature_vector)
    if num_features >= 2:
        return f"TEMP: Mean Temperature={temp_feature_vector[0]:.2f}, Temperature Range={temp_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(temp_feature_vector)])
        return f"TEMP: {features_str}"

# === Optimized prompt template ===
def get_instruction_template():
    return (
        "You are an expert in emotion recognition using physiological signals. "
        "Analyze the baseline-corrected physiological features and classify the emotional state on the valence dimension as: "
        "Negative emotion (label: 0) or Positive emotion (label: 1). "
        "Baseline Correction: "
        "All features are calculated as the difference from the baseline features (i.e., subtracted from baseline). Positive values indicate increase from baseline, negative values indicate decrease."
    )
    
def get_input_template(sample_idx, eeg_display, eog_formatted, emg_formatted, 
                    gsr_formatted, resp_formatted, pleth_formatted, temp_formatted):
 
    return (
        "EEG: EEG Spectral Analysis (32 channels; PSD-based relative band power; Frequency bands: Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-47Hz)):\n"
        f"{eeg_display}\n"
        f"{eog_formatted}\n"
        f"{emg_formatted}\n"
        f"{gsr_formatted}\n"
        f"{resp_formatted}\n"
        f"{pleth_formatted}\n"
        f"{temp_formatted}\n"
        "Based on these features, your task is to classify the emotion of the given sample as either Negative (label: 0) or Positive (label: 1). "
        "Final Output Format (must follow exactly): "
        "Identify_emotion: <emotion label>\n"
        "Do not output any other text or explanation."  
    )
    
# The template above is for the Valence dimension. For Arousal, replace it with the template below and update the input/output paths.
""" 
def get_instruction_template():
    return (
        "You are an expert in emotion recognition using physiological signals. "
        "Analyze the baseline-corrected physiological features and classify the emotional state on the arousal dimension as: "
        "Low emotion (label: 0) or High emotion (label: 1). "
        "Baseline Correction: "
        "All features are calculated as the difference from the baseline features (i.e., subtracted from baseline). Positive values indicate increase from baseline, negative values indicate decrease."
  
    )
def get_input_template(sample_idx, eeg_display, eog_formatted, emg_formatted, 
                      gsr_formatted, resp_formatted, pleth_formatted, temp_formatted):
    
    return (
        "EEG: EEG Spectral Analysis (32 channels; PSD-based relative band power; Frequency bands: Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-47Hz)):\n"
        f"{eeg_display}\n"
        f"{eog_formatted}\n"
        f"{emg_formatted}\n"
        f"{gsr_formatted}\n"
        f"{resp_formatted}\n"
        f"{pleth_formatted}\n"
        f"{temp_formatted}\n"
        "Based on these features, your task is to classify the emotion of the given sample as either Low (label: 0) or High (label: 1). "
        "Final Output Format (must follow exactly): "
        "Identify_emotion: <emotion label>\n"
        "Do not output any other text or explanation." 
    )
"""
output_list = []
for i in range(num_samples): 
    try:
        eeg_formatted = format_eeg_features(features[0][i], EEG_channels_geneva)
        eog_formatted = format_eog_features(features[1][i])
        emg_formatted = format_emg_features(features[2][i])
        gsr_formatted = format_gsr_features(features[3][i])
        resp_formatted = format_respiration_features(features[4][i])
        pleth_formatted = format_plethysmograph_features(features[5][i])
        temp_formatted = format_temperature_features(features[6][i])
        eeg_groups = [eeg_formatted[j:j+8] for j in range(0, len(eeg_formatted), 8)]
        eeg_display = "\n".join([f"  - " + "\n  - ".join(group) for group in eeg_groups])
        instruction = get_instruction_template()
        input_text = get_input_template(i, eeg_display, eog_formatted, emg_formatted, 
                                          gsr_formatted, resp_formatted, pleth_formatted, temp_formatted)
        
        output = {
            "instruction": instruction,
            "input": input_text,
            "output": f"Identify_emotion: {int(labels[i][0])}."
        }
        output_list.append(output)
    except Exception as e:
        continue

# === Save ===
save_path = 's.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=4, ensure_ascii=False)

