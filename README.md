# PhysioEmoLLM-Bench

PhysioEmoLLM-Bench evaluates the zero-shot emotion recognition ability of large language models (LLMs) on physiological signals. Hand-crafted features are converted into textual prompts, and LLMs predict emotion labels without task-specific training. The repository also provides the multi-task LoRA adaptation used for the best-performing model.

| Dataset | Modalities | Tasks |
| ------- | ---------- | ----- |
| DEAP    | EEG, EOG, EMG, GSR, Respiration, Plethysmograph, Temperature | Valence / Arousal (binary) |
| DREAMER | EEG, ECG | Valence / Arousal (binary) |
| WESAD   | Wrist BVP, EDA, Temperature | Non-stress vs. Stress (2-class); Baseline / Stress / Amusement (3-class) |

## Repository Structure

```
PhysioEmoLLM-Bench/
├── feature_extraction/   # Feature extraction (DEAP.py, DREAMER.py, WESAD.py)
│                         # and DREAMER_prepare.py to split DREAMER.mat into per-subject files
├── text_embedding/       # Convert features into JSON prompt files (DEAP.py, DREAMER.py, WESAD.py)
├── main/                 # Zero-shot evaluation (DEAP.py, DREAMER.py, WESAD.py for 2-class, WESAD_3class.py)
├── lora_adaptation/      # Unified multi-task LoRA fine-tuning and evaluation
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

`transformers>=4.57` is required for Qwen3-VL. The Gemma-4 checkpoints require `transformers` 5.x.

## Datasets

The raw datasets are not included and should be requested from their official sources: [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [DREAMER](https://zenodo.org/record/546113), and [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/). Each feature extraction script reads a `.txt` file (`txt_path`) listing one subject file per line:

- **DEAP**: the preprocessed MATLAB files `s01.mat`–`s32.mat`.
- **DREAMER**: per-subject files generated from the original `DREAMER.mat` with
  `python feature_extraction/DREAMER_prepare.py --dreamer_mat /path/to/DREAMER.mat --output_dir /path/to/data_subject`.
- **WESAD**: the original subject files `S2.pkl`–`S17.pkl`.

Subjects are split into training, validation, and test sets with no overlap. Zero-shot evaluation uses only the test subjects.

| Dataset | Training | Validation | Test |
| ------- | -------- | ---------- | ---- |
| DEAP    | s01–s26 | s27–s29 | s30–s32 |
| DREAMER | subject 01–19 | subject 20–21 | subject 22–23 |
| WESAD   | S2–S15 (no S12 in WESAD) | S16 | S17 |

Signals are segmented into non-overlapping 3-second windows. Labels are defined as follows:

- **DEAP**: ratings ≤ 5 → 0, > 5 → 1.
- **DREAMER**: trials rated exactly 3 on the target dimension are excluded; ratings < 3 → 0, > 3 → 1.
- **WESAD**: 2-class uses Non-stress (baseline and amusement, 0) vs. Stress (1); 3-class uses Baseline (0), Stress (1), Amusement (2). Each window takes its majority label, and windows of other conditions are discarded.

For DEAP and DREAMER, baseline correction subtracts the mean features of the last 3-s baseline segments, averaged over all trials in the processed subject list. Run feature extraction separately for each split so that each split uses its own baseline statistics. WESAD features are not baseline-corrected.

## Usage

Set the placeholder paths (`'.txt'`, `'.pkl'`, `'.json'`) in `feature_extraction/` and `text_embedding/` before running. Commands are run from the repository root, using DEAP as an example.

```bash
python feature_extraction/DEAP.py   # features of one subject list -> .pkl
python text_embedding/DEAP.py       # .pkl -> JSON prompts
python main/DEAP.py \
    --test_dataset_dir main/datasets_json/deap/valence.json \
    --save_result_dir main/results/deap/valence \
    --llms_dir main/models          # one sub-folder per model
```

`--model_names` and `--skip_models` select or skip models. Use `main/WESAD.py` for WESAD 2-class and `main/WESAD_3class.py` for WESAD 3-class.

Text-only LLMs are loaded with `AutoModelForCausalLM`, while Qwen3-VL and LLaVA are loaded with their vision-language classes and used in text-only mode. Each model uses its default generation configuration, so models that sample by default may produce slightly different outputs across runs. Outputs are parsed with a strict and a lenient parser. Parsing failures count as errors: binary tasks assign the opposite label, and WESAD 3-class assigns a random incorrect label. Accuracy, precision, recall, F1 (macro-averaged for WESAD 3-class), parsing error rate, and inference time are saved under `--save_result_dir`.

**Switching tasks.** The defaults are Valence for DEAP and DREAMER and 2-class for WESAD. For Arousal or WESAD 3-class data, follow the comments in `feature_extraction/` (label selection) and `text_embedding/` (alternative prompt template). For Arousal evaluation, update the default paths in `parse_for_llm()` of `main/DEAP.py` or `main/DREAMER.py` and switch to the commented-out Arousal parsers.

## Multi-task LoRA Adaptation

`lora_adaptation/main_unified_multitask_lora.py` jointly fine-tunes Qwen3-1.7B on the training data of all six tasks and evaluates it on each task with the same strict and lenient parsers (greedy decoding).

1. Generate the JSON files of each split and place them as `lora_adaptation/datasets_json/{deap,dreamer}/{valence,arousal}_{train,valid,test}.json` and `lora_adaptation/datasets_json/wesad/{class2,class3}_{train,valid,test}.json`.
2. Place the base model in `lora_adaptation/models/Qwen3-1.7B`.
3. Run `python lora_adaptation/main_unified_multitask_lora.py`.

The adapter and results are saved under `lora_adaptation/models/` and `lora_adaptation/results/`.

LoRA (r = 8, α = 16, dropout = 0.1) is applied to all attention and feed-forward projections. The model is trained for 1 epoch with a learning rate of 1e-5, and the checkpoint with the lowest validation loss is kept. DEAP and DREAMER use 50% of their training samples, and WESAD uses the full training set. All other settings are listed in `CONFIG` and `TrainingArguments` in the script.

## Evaluated LLMs

The 40 lightweight LLMs (< 10B parameters) evaluated in the benchmark, obtained from Hugging Face (LLaVA-1.5-7B: `llava-hf/llava-1.5-7b-hf`):

- **Base**: DeepSeek-LLM-7B-Base, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, Gemma-3-1B-PT, Gemma-3-4B-PT, Gemma-4-E2B, Gemma-4-E4B, Hunyuan-0.5B-Pretrain, Hunyuan-1.8B-Pretrain, Hunyuan-4B-Pretrain, Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B, Mistral-7B-v0.1, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B.
- **Instruct**: DeepSeek-LLM-7B-Chat, Gemma-3-1B-IT, Gemma-3-4B-IT, Gemma-4-E2B-IT, Gemma-4-E4B-IT, Hunyuan-0.5B-Instruct, Hunyuan-1.8B-Instruct, Hunyuan-4B-Instruct, LLaVA-1.5-7B, Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3, Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct.

## Note

The ablation studies (input duration, modality combination, and prompt sensitivity) reuse this pipeline with modified settings; their scripts are not included.
