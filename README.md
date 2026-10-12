# PhysioEmoLLM-Bench

PhysioEmoLLM-Bench is a benchmark for evaluating the zero-shot emotion recognition ability of large language models (LLMs) on physiological signals. Hand-crafted physiological features are converted into textual prompts, and LLMs are asked to predict emotion labels without any task-specific training.

Three public datasets are supported:

| Dataset | Modalities | Tasks |
| ------- | ---------- | ----- |
| DEAP    | EEG, EOG, EMG, GSR, Respiration, Plethysmograph, Temperature | Valence / Arousal (binary) |
| DREAMER | EEG, ECG | Valence / Arousal (binary) |
| WESAD   | Wrist BVP, EDA, Temperature | Non-stress vs. Stress (2-class); Baseline / Stress / Amusement (3-class) |

## Repository Structure

```
PhysioEmoLLM-Bench/
├── feature_extraction/   # Feature extraction for each dataset
├── text_embedding/       # Convert extracted features into JSON prompt files for each dataset
├── main/                 # Zero-shot evaluation of different LLMs on each dataset
├── requirements.txt
└── README.md
```

Each folder contains one script per dataset (`DEAP.py`, `DREAMER.py`, `WESAD.py`).

## Requirements

```bash
pip install -r requirements.txt
```

A recent version of `transformers` with Qwen3-VL support is required.

## Datasets

The raw datasets are not included in this repository and should be requested from their official sources under the respective license agreements: [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [DREAMER](https://zenodo.org/record/546113), and [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/).

Each feature extraction script reads a `.txt` file (`txt_path`) that lists the path of one subject file per line:

- **DEAP**: the preprocessed MATLAB files (`s01.mat`–`s32.mat`), each containing `data` and `labels`.
- **DREAMER**: one `.mat` file per subject with a `combined_data` variable, where each row is a trial of `[eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, score_valence, score_arousal, score_dominance]`, reorganized from the original `DREAMER.mat`.
- **WESAD**: the original subject files (`S2.pkl`–`S17.pkl`).

## Usage

The pipeline consists of three steps. The input and output paths in `feature_extraction/` and `text_embedding/` are left as placeholders (e.g., `'.pkl'`, `'.json'`) and should be set before running. All commands below are run from the repository root, using DEAP as an example.

1. **Feature extraction**: extract physiological features from the raw data and save them as a `.pkl` file.

   ```bash
   python feature_extraction/DEAP.py
   ```

2. **Prompt construction**: convert the extracted features into a JSON file of instruction–input–output prompts.

   ```bash
   python text_embedding/DEAP.py
   ```

3. **Zero-shot evaluation**: evaluate all LLMs placed under `main/models/` (one sub-folder per model) on the generated JSON file.

   ```bash
   python main/DEAP.py \
       --test_dataset_dir main/datasets_json/deap/valence.json \
       --save_result_dir main/results/deap/valence \
       --llms_dir main/models
   ```

   Optional arguments: `--model_names` to evaluate only the specified models, and `--skip_models` to skip the specified models.

   Each model's output is parsed with both a strict and a lenient parser. Accuracy, precision, recall, and F1-score are reported for both, and the predictions, error logs, and reports are saved under `--save_result_dir`.

## Switching Tasks

By default, the scripts use the Valence dimension for DEAP and DREAMER and the 2-class setting for WESAD. To switch to Arousal (DEAP / DREAMER) or the 3-class setting (WESAD), follow the comments in the corresponding scripts:

- `feature_extraction/`: change the label used to build the dataset (or, for WESAD, the label mapping).
- `text_embedding/`: replace the prompt template with the commented-out alternative.
- `main/`: update the default paths in `parse_for_llm()` and replace `strict_parse_response()` and `lenient_parse_response()` with the commented-out alternatives.

## Note

This repository currently includes the core zero-shot evaluation code. Additional experimental scripts will be released after the paper is accepted.
