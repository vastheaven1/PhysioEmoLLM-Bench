import os
import json
import pickle
import numpy as np
# === Definition of EEG channel names ===
EEG_channels_geneva = ['AF3', 'F7', 'F3','FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# === Load data  ===
with open('.pkl', 'rb') as f:
    data = pickle.load(f)

features = [data['modality1'], data['modality2']]

labels = data['label']
num_samples = len(labels)

# Check the actual feature dimensions
feature_names = ['EEG', 'ECG']
for i, (name, feature_array) in enumerate(zip(feature_names, features)):
    if feature_array is not None and len(feature_array) > 0:
        print(f"{name}: shape {feature_array.shape} -> {feature_array.shape[1]} features per sample")
    else:
        print(f"{name}: no data")

def format_eeg_features(eeg_feature_vector, channels):
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
def format_ecg_features(ecg_feature_vector):
    num_features = len(ecg_feature_vector)
    
    if num_features >= 4:
        return (f"ECG: Channel1 Mean Heart Rate={ecg_feature_vector[0]:.2f}, "
                f"Channel1 Heart Rate Variability={ecg_feature_vector[1]:.2f}, "
                f"Channel2 Mean Heart Rate={ecg_feature_vector[2]:.2f}, "
                f"Channel2 Heart Rate Variability={ecg_feature_vector[3]:.2f}")
    elif num_features >= 2:
        return f"ECG: Mean Heart Rate={ecg_feature_vector[0]:.2f}, Heart Rate Variability={ecg_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(ecg_feature_vector)])
        return f"ECG: {features_str}"
    
# === Optimized prompt template ===
def get_instruction_template():
    """Simplified instruction template."""
    return (
        "You are an expert in emotion recognition using physiological signals. "
        "Analyze the baseline-corrected physiological features and classify the emotional state on the valence dimension as: "
        "Negative emotion (label: 0) or Positive emotion (label: 1). "
        "Baseline Correction: "
        "All features are calculated as the difference from the baseline features (i.e., subtracted from baseline). Positive values indicate increase from baseline, negative values indicate decrease."
  
    )
def get_input_template(sample_idx, eeg_display, ecg_formatted):
    return (
        "EEG: EEG Spectral Analysis (14 channels; PSD-based relative band power; Frequency bands: Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-47Hz)):\n"
        f"{eeg_display}\n"
        f"{ecg_formatted}\n"
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

def get_input_template(sample_idx, eeg_display, ecg_formatted):
    return (
        "EEG: EEG Spectral Analysis (14 channels; PSD-based relative band power; Frequency bands: Theta(4-8Hz), Alpha(8-13Hz), Beta(13-30Hz), Gamma(30-47Hz)):\n"
        f"{eeg_display}\n"
        f"{ecg_formatted}\n"
        "Based on these features, your task is to classify the emotion of the given sample as either Low (label: 0) or High (label: 1). "
        "Final Output Format (must follow exactly): "
        "Identify_emotion: <emotion label>\n"
        "Do not output any other text or explanation." 
    )
"""

# === Construct JSON data ===
output_list = []
for i in range(num_samples):  
    try:
        # Format various modal features
        eeg_formatted = format_eeg_features(features[0][i], EEG_channels_geneva)
        ecg_formatted = format_ecg_features(features[1][i])
     
    
        eeg_groups = [eeg_formatted[j:j+8] for j in range(0, len(eeg_formatted), 8)]
        eeg_display = "\n".join([f"  - " + "\n  - ".join(group) for group in eeg_groups])
        instruction = get_instruction_template()
        input_text = get_input_template(i, eeg_display, ecg_formatted)
        
        output = {
            "instruction": instruction,
            "input": input_text,
            "output": f"Identify_emotion: {int(labels[i][0])}."
        }
        output_list.append(output)   
    except Exception as e:
        continue

# === Save ===
save_path = '.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=4, ensure_ascii=False)

