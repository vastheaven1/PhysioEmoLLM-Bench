import os
import json
import pickle
import numpy as np

# === Load data ===
with open('.pkl', 'rb') as f:
    data = pickle.load(f)

features = [data['bvp'], data['eda'], data['temp']]  
labels = data['label']
num_samples = len(labels)

# Check the actual feature dimensions

feature_names = ['BVP', 'EDA', 'Temperature'] 
for i, (name, feature_array) in enumerate(zip(feature_names, features)):
    if feature_array is not None and len(feature_array) > 0:
        print(f"{name}: shape {feature_array.shape} -> {feature_array.shape[1]} features per sample")
    else:
        print(f"{name}: no data")
def format_bvp_features(pleth_feature_vector):
    num_features = len(pleth_feature_vector)
    if num_features >= 2:
        return f"BVP: Heart Rate={pleth_feature_vector[0]:.2f}, Heart Rate Variability Standard Deviation={pleth_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(pleth_feature_vector)])
        return f"BVP: {features_str}"

def format_eda_features(gsr_feature_vector):
    num_features = len(gsr_feature_vector)
    if num_features >= 2:
        return f"EDA: Mean Conductance={gsr_feature_vector[0]:.2f}, Mean Absolute Derivative={gsr_feature_vector[1]:.2f}"
    else:
        features_str = ", ".join([f"Feature {i}={val:.2f}" for i, val in enumerate(gsr_feature_vector)])
        return f"EDA: {features_str}"


def format_temp_features(temp_feature_vector):
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
        "Analyze the original physiological features (without baseline correction) and classify the emotional state as: "
        "Non-stress emotion (label: 0) or Stress emotion (label: 1). "
        "Baseline Correction: "
        "No baseline correction has been performed on the features."

    )
   
def get_input_template(sample_idx, bvp_formatted, 
                    eda_formatted, temp_formatted):
    return (
        f"{bvp_formatted}\n"
        f"{eda_formatted}\n"
        f"{temp_formatted}\n"
        "Based on these features, your task is to classify the emotion of the given sample as either Non-stress (label: 0) or Stress (label: 1). "
        "Final Output Format (must follow exactly): "
        "Identify_emotion: <emotion label>\n"
        "Do not output any other text or explanation."  
    )
    
# The template above is for the 2-class setting. For the 3-class setting, replace it with the template below and update the input/output paths.
""" 
def get_instruction_template():
    return (
        "You are an expert in emotion recognition using physiological signals. "
        "Analyze the original physiological features (without baseline correction) and classify the emotional state as: "
        "Baseline emotion (label: 0), Stress emotion (label: 1), or Amusement emotion (label: 2). "
        "Baseline Correction: "
        "No baseline correction has been performed on the features."

    )
   
def get_input_template(sample_idx, bvp_formatted, 
                    eda_formatted, temp_formatted):
  
    return (
        f"{bvp_formatted}\n"
        f"{eda_formatted}\n"
        f"{temp_formatted}\n"
        "Based on these features, your task is to classify the emotion of the given sample as either Baseline (label: 0), Stress (label: 1), or Amusement (label: 2). "
        "Final Output Format (must follow exactly): "
        "Identify_emotion: <emotion label>\n"
        "Do not output any other text or explanation." 
    )
"""    
    
# === Construct JSON data ===
output_list = []
for i in range(num_samples): 
    try:
      
        bvp_formatted = format_bvp_features(features[0][i])
        eda_formatted = format_eda_features(features[1][i])
        temp_formatted = format_temp_features(features[2][i])
        instruction = get_instruction_template()
        input_text = get_input_template(i, bvp_formatted,eda_formatted, temp_formatted)
    
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

