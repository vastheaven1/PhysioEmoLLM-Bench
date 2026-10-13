import datetime
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Paths are relative to this folder: datasets_json/{deap,dreamer,wesad}/*_{train,valid,test}.json,
# the base model in models/Qwen3-1.7B, and outputs in models/unified_multitask and results/unified_multitask.
LORA_ROOT = Path(__file__).resolve().parent
DATA_ROOT = LORA_ROOT / "datasets_json"
MODEL_STORE_ROOT = LORA_ROOT / "models"
RESULTS_STORE_ROOT = LORA_ROOT / "results"

CONFIG = {
    "tasks": {
        "deap_valence": {
            "train_json": str(DATA_ROOT / "deap" / "valence_train.json"),
            "valid_json": str(DATA_ROOT / "deap" / "valence_valid.json"),
            "test_json":  str(DATA_ROOT / "deap" / "valence_test.json"),
            "train_ratio": 0.5,
            "sample_strategy": "subject",
        },
        "deap_arousal": {
            "train_json": str(DATA_ROOT / "deap" / "arousal_train.json"),
            "valid_json": str(DATA_ROOT / "deap" / "arousal_valid.json"),
            "test_json":  str(DATA_ROOT / "deap" / "arousal_test.json"),
            "train_ratio": 0.5,
            "sample_strategy": "subject",
        },
        "dreamer_valence": {
            "train_json": str(DATA_ROOT / "dreamer" / "valence_train.json"),
            "valid_json": str(DATA_ROOT / "dreamer" / "valence_valid.json"),
            "test_json":  str(DATA_ROOT / "dreamer" / "valence_test.json"),
            "train_ratio": 0.5,
            "sample_strategy": "subject",
        },
        "dreamer_arousal": {
            "train_json": str(DATA_ROOT / "dreamer" / "arousal_train.json"),
            "valid_json": str(DATA_ROOT / "dreamer" / "arousal_valid.json"),
            "test_json":  str(DATA_ROOT / "dreamer" / "arousal_test.json"),
            "train_ratio": 0.5,
            "sample_strategy": "subject",
        },
        "wesad_class2": {
            "train_json": str(DATA_ROOT / "wesad" / "class2_train.json"),
            "valid_json": str(DATA_ROOT / "wesad" / "class2_valid.json"),
            "test_json":  str(DATA_ROOT / "wesad" / "class2_test.json"),
            "train_ratio": 1.0,
            "sample_strategy": "subject",
        },
        "wesad_class3": {
            "train_json": str(DATA_ROOT / "wesad" / "class3_train.json"),
            "valid_json": str(DATA_ROOT / "wesad" / "class3_valid.json"),
            "test_json":  str(DATA_ROOT / "wesad" / "class3_test.json"),
            "train_ratio": 1.0,
            "sample_strategy": "subject",
        }
    },
    "model_root": str(MODEL_STORE_ROOT / "unified_multitask"),
    "results_root": str(RESULTS_STORE_ROOT / "unified_multitask"),
    "base_model_path": str(MODEL_STORE_ROOT / "Qwen3-1.7B"),
    "base_model_name": "Qwen3-1.7B",
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 1e-5,
    "num_train_epochs": 1,
    "max_length": 2048,
    "max_output_tokens": 10,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "per_device_eval_batch_size": 1,
    "seed": 2025
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])
torch.cuda.empty_cache()

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_training_data(train_samples: List[Dict], ratio: float, strategy: str) -> List[Dict]:
    if ratio >= 1.0: return train_samples
    if strategy == "instance":
        return random.sample(train_samples, int(len(train_samples) * ratio))
    elif strategy == "subject":
        key = "subject_id" if "subject_id" in train_samples[0] else "participant_id"
        if key in train_samples[0]:
            subjects = list(set(s[key] for s in train_samples))
            selected_subjects = set(random.sample(subjects, max(1, int(len(subjects) * ratio))))
            return [s for s in train_samples if s[key] in selected_subjects]
        else:
            chunks = [train_samples[i:i+800] for i in range(0, len(train_samples), 800)]
            selected = random.sample(chunks, max(1, int(len(chunks) * ratio)))
            return [item for sublist in selected for item in sublist]

def sample_balanced_valid_data(all_valid_datasets: Dict[str, List[Dict]], samples_per_task: int = 400) -> List[Dict]:
    balanced_valid = []
    for task_name, v_data in all_valid_datasets.items():
        num_to_take = min(len(v_data), samples_per_task)
        balanced_valid.extend(random.sample(v_data, num_to_take))
        print(f"   => Balanced validation sampling: selected {num_to_take} samples from {task_name}")
    random.shuffle(balanced_valid)
    print(f"Balanced validation set ready. Total samples: {len(balanced_valid)}")
    return balanced_valid

def load_multi_task_data(tasks_config: Dict) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    unified_train_samples = []
    all_valid_datasets = {}
    all_test_datasets = {}

    for task_name, cfg in tasks_config.items():
        if not os.path.exists(cfg["train_json"]):
            print(f"Dataset for task {task_name} was not found. Skipping.")
            continue
            
        print(f"Loading and processing task: {task_name} ...")
        train_raw = load_json(cfg["train_json"])
        valid_raw = load_json(cfg["valid_json"])
        test_raw = load_json(cfg["test_json"])

        sampled_train = sample_training_data(train_raw, cfg["train_ratio"], cfg["sample_strategy"])
        unified_train_samples.extend(sampled_train)
        all_valid_datasets[task_name] = valid_raw
        all_test_datasets[task_name] = test_raw
        print(f"   => Training samples for {task_name}: {len(sampled_train)}")

    random.shuffle(unified_train_samples)
    print(f"\nMultitask data mixing complete. Unified training samples: {len(unified_train_samples)}")
    return unified_train_samples, all_valid_datasets, all_test_datasets

def build_prompt(sample: Dict, include_label: bool, tokenizer: AutoTokenizer) -> str:
    custom_user_content = f"### Input: \n{sample['input']}\n### Output:\n"
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": custom_user_content}
    ]
    if include_label:
        messages.append({"role": "assistant", "content": sample["output"].strip()})
        
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not include_label, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not include_label)

def make_dataset(samples: List[Dict], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _to_id_list(token_obj):
        if isinstance(token_obj, dict):
            token_obj = token_obj.get("input_ids", token_obj)
        elif hasattr(token_obj, "data") and "input_ids" in token_obj:
            token_obj = token_obj["input_ids"]

        if isinstance(token_obj, list) and len(token_obj) > 0 and isinstance(token_obj[0], list):
            token_obj = token_obj[0]
        return token_obj

    ids, masks, labels = [], [], []
    skipped_count = 0
    max_output_tokens = CONFIG.get("max_output_tokens", 10)
    max_prompt_length = max(32, max_length - max_output_tokens-1)
    
    for s in tqdm(samples, desc="Tokenizing Dataset", mininterval=2.0, ncols=100):
        custom_user_content = f"### Input: \n{s['input']}\n### Output:\n"
        
        msg_full = [
            {"role": "system", "content": s["instruction"]},
            {"role": "user", "content": custom_user_content},
            {"role": "assistant", "content": s["output"].strip()}
        ]
        msg_prompt = [
            {"role": "system", "content": s["instruction"]},
            {"role": "user", "content": custom_user_content}
        ]
        try:
            p_ids = tokenizer.apply_chat_template(
                msg_prompt,
                tokenize=True,
                max_length=max_prompt_length,
                truncation=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            p_ids = tokenizer.apply_chat_template(
                msg_prompt,
                tokenize=True,
                max_length=max_prompt_length,
                truncation=True,
                add_generation_prompt=True,
            )
        p_ids = _to_id_list(p_ids)

        a_ids = tokenizer(
            s["output"].strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=max_output_tokens,
        )["input_ids"]
        a_ids = _to_id_list(a_ids)

        if len(a_ids) == 0:
            skipped_count += 1
            continue

        f_ids = p_ids + a_ids
        if tokenizer.eos_token_id is not None and len(f_ids) < max_length:
            f_ids = f_ids + [tokenizer.eos_token_id]

        p_len = len(p_ids)
        if p_len >= len(f_ids):
            skipped_count += 1
            continue

        # Mask prompt tokens so only answer tokens contribute to the loss.
        f_ids = f_ids[:max_length]
        labels_ids = [-100] * p_len + f_ids[p_len:]
        if len(labels_ids) != len(f_ids):
            skipped_count += 1
            continue

        ids.append(f_ids)
        masks.append([1] * len(f_ids))
        labels.append(labels_ids)
        
    print(f"Samples skipped because no valid supervised label remained: {skipped_count} / {len(samples)}")
    return Dataset.from_dict({"input_ids": ids, "attention_mask": masks, "labels": labels})

class LoraZeroShotEvaluator:
    def __init__(self, model, tokenizer, data: List[Dict], save_dir: str, split_name: str, max_length: int, max_new_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.save_dir = save_dir
        self.split_name = split_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_patterns_and_map(self):
        if "valence" in self.split_name:
            p_all = r'\b(negative|positive|0|1)\b'
            p_strict = r"Identify_emotion:\s*\b(negative|positive|0|1)\b"
            e_map = {"negative": 0, "0": 0, "positive": 1, "1": 1}
        elif "arousal" in self.split_name:
            p_all = r'\b(low|high|0|1)\b'
            p_strict = r"Identify_emotion:\s*\b(low|high|0|1)\b"
            e_map = {"low": 0, "0": 0, "high": 1, "1": 1}
        elif "class2" in self.split_name:
            p_all = r'\b(non-stress|stress|0|1)\b'
            p_strict = r"Identify_emotion:\s*\b(non-stress|stress|0|1)\b"
            e_map = {"non-stress": 0, "0": 0, "stress": 1, "1": 1}
        elif "class3" in self.split_name:
            p_all = r'\b(baseline|stress|amusement|0|1|2)\b'
            p_strict = r"Identify_emotion:\s*\b(baseline|stress|amusement|0|1|2)\b"
            e_map = {"baseline": 0, "0": 0, "stress": 1, "1": 1, "amusement": 2, "2": 2}
        else:
            raise ValueError(f"Unknown task name: {self.split_name}")
        return p_all, p_strict, e_map

    def strict_parse_response(self, response: str) -> Optional[int]:
        response_parts = response.split("assistant\n", 1)
        if len(response_parts) > 1:
            response_content = response_parts[1]
        else:
            return None
            
        p_all, p_strict, e_map = self._get_patterns_and_map()
        all_matches = re.findall(p_all, response_content, re.IGNORECASE)
        if len(all_matches) != 1:
            return None
            
        match = re.search(p_strict, response_content.strip(), re.IGNORECASE)
        if match:
            return e_map.get(match.group(1).lower())
        return None

    def lenient_parse_response(self, response: str) -> Optional[int]:
        response_parts = response.split("assistant\n", 1)
        if len(response_parts) > 1:
            response_content = response_parts[1].lower()
        else:
            return None
            
        if "valence" in self.split_name:
            has_0 = bool(re.search(r'\b(negative|0)\b', response_content))
            has_1 = bool(re.search(r'\b(positive|1)\b', response_content))
        elif "arousal" in self.split_name:
            has_0 = bool(re.search(r'\b(low|0)\b', response_content))
            has_1 = bool(re.search(r'\b(high|1)\b', response_content))
        elif "class2" in self.split_name:
            has_0 = bool(re.search(r'\b(non-stress|0)\b', response_content))
            has_1 = bool(re.search(r'\b(stress|1)\b', response_content))
        elif "class3" in self.split_name:
            has_0 = bool(re.search(r'\b(baseline|0)\b', response_content))
            has_1 = bool(re.search(r'\b(stress|1)\b', response_content))
            has_2 = bool(re.search(r'\b(amusement|2)\b', response_content))
            if has_0 and not has_1 and not has_2: return 0
            if has_1 and not has_0 and not has_2: return 1
            if has_2 and not has_0 and not has_1: return 2
            return None

        if "class3" not in self.split_name:
            if has_0 and not has_1: return 0
            elif has_1 and not has_0: return 1
        return None

    def calculate_metrics(self, y_true, y_pred, dataset_type):
        if not y_true: return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": dataset_type, "sample_count": 0}
        
        accuracy = float(accuracy_score(y_true, y_pred))
        
        if "class3" in self.split_name:
            precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        else:
            if len(set(y_true)) == 1:
                precision = 1.0 if y_true[0] == 1 and sum(y_pred) == len(y_pred) else 0.0
                recall = 1.0 if y_true[0] == 1 and sum(y_pred) == len(y_pred) else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
                recall = float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
                f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
                
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "dataset_type": dataset_type, "sample_count": len(y_true)}

    def run(self):
        y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict = [],[], [],[]
        strict_parse_error_indices, strict_error_log = [],[]
        y_true_all_lenient, y_pred_all_lenient, y_true_valid_lenient, y_pred_valid_lenient =[], [], [],[]
        lenient_parse_error_indices, lenient_error_log = [],[]
        inference_times, raw_responses, parse_differences = [],[],[]

        possible_labels = [0, 1, 2] if "class3" in self.split_name else [0, 1]

        for idx, sample in tqdm(enumerate(self.data), total=len(self.data), desc=f"Evaluating[{self.split_name}]", mininterval=2.0, maxinterval=5.0, ncols=100):
            try:
                full_prompt = build_prompt(sample, include_label=False, tokenizer=self.tokenizer)
                inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=10, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id, 
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                inference_times.append(time.perf_counter() - start_time)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_responses.append(response)
                
                true_label = int(sample["output"].split(": ")[-1].strip("."))
                strict_pred = self.strict_parse_response(response)
                lenient_pred = self.lenient_parse_response(response)
                
                y_true_all_strict.append(true_label)
                if strict_pred is not None and strict_pred in set(possible_labels):
                    y_pred_all_strict.append(strict_pred)
                    y_true_valid_strict.append(true_label)
                    y_pred_valid_strict.append(strict_pred)
                else:
                    choices = [l for l in possible_labels if l != true_label]
                    error_pred = random.choice(choices) if choices else 0
                    y_pred_all_strict.append(error_pred)
                    strict_parse_error_indices.append(idx)
                    strict_error_log.append({"index": idx, "true_label": true_label, "assigned_pred": error_pred, "raw_response": response, "error_type": "strict_parse_error"})
                
                y_true_all_lenient.append(true_label)
                if lenient_pred is not None and lenient_pred in set(possible_labels):
                    y_pred_all_lenient.append(lenient_pred)
                    y_true_valid_lenient.append(true_label)
                    y_pred_valid_lenient.append(lenient_pred)
                else:
                    choices = [l for l in possible_labels if l != true_label]
                    error_pred = random.choice(choices) if choices else 0
                    y_pred_all_lenient.append(error_pred)
                    lenient_parse_error_indices.append(idx)
                    lenient_error_log.append({"index": idx, "true_label": true_label, "assigned_pred": error_pred, "raw_response": response, "error_type": "lenient_parse_error"})
                
                if strict_pred != lenient_pred:
                    parse_differences.append({"index": idx, "true_label": true_label, "strict_pred": strict_pred, "lenient_pred": lenient_pred, "raw_response": response})
                    
            except Exception as e:
                true_label = int(sample["output"].split(": ")[-1].strip("."))
                inference_times.append(0)
                raw_responses.append("")
                
                choices = [l for l in possible_labels if l != true_label]
                error_pred = random.choice(choices) if choices else 0
                
                y_true_all_strict.append(true_label)
                y_pred_all_strict.append(error_pred)
                strict_parse_error_indices.append(idx)
                strict_error_log.append({"index": idx, "error": str(e), "true_label": true_label, "error_type": "exception"})
                
                y_true_all_lenient.append(true_label)
                y_pred_all_lenient.append(error_pred)
                lenient_parse_error_indices.append(idx)
                lenient_error_log.append({"index": idx, "error": str(e), "true_label": true_label, "error_type": "exception"})

        metrics_all_strict = self.calculate_metrics(y_true_all_strict, y_pred_all_strict, f"Strict parsing - full dataset ({self.split_name})")
        metrics_valid_strict = self.calculate_metrics(y_true_valid_strict, y_pred_valid_strict, f"Strict parsing - valid predictions ({self.split_name})") if y_true_valid_strict else {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": f"Strict parsing - valid predictions ({self.split_name})", "sample_count": 0}
        
        metrics_all_lenient = self.calculate_metrics(y_true_all_lenient, y_pred_all_lenient, f"Lenient parsing - full dataset ({self.split_name})")
        metrics_valid_lenient = self.calculate_metrics(y_true_valid_lenient, y_pred_valid_lenient, f"Lenient parsing - valid predictions ({self.split_name})") if y_true_valid_lenient else {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": f"Lenient parsing - valid predictions ({self.split_name})", "sample_count": 0}

        self.save_results(
            y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict, 
            strict_error_log, strict_parse_error_indices, y_true_all_lenient, y_pred_all_lenient, 
            y_true_valid_lenient, y_pred_valid_lenient, lenient_error_log, lenient_parse_error_indices, 
            inference_times, raw_responses, parse_differences, metrics_all_strict, metrics_valid_strict, 
            metrics_all_lenient, metrics_valid_lenient
        )
        
        self.save_detailed_report(
            metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient, 
            inference_times, strict_error_log, lenient_error_log, strict_parse_error_indices, 
            lenient_parse_error_indices, parse_differences, len(self.data)
        )
        
        self.print_metrics(metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient)
        
        return metrics_all_strict

    def save_results(self, y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict, 
                     strict_error_log, strict_parse_error_indices, y_true_all_lenient, y_pred_all_lenient, 
                     y_true_valid_lenient, y_pred_valid_lenient, lenient_error_log, lenient_parse_error_indices, 
                     inference_times, raw_responses, parse_differences, metrics_all_strict, metrics_valid_strict, 
                     metrics_all_lenient, metrics_valid_lenient):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(os.path.join(self.save_dir, f"predictions_{timestamp}.json"), "w") as f:
            json.dump({
                "strict_parsing": {
                    "true_labels_all": y_true_all_strict,
                    "pred_labels_all": y_pred_all_strict,
                    "true_labels_valid": y_true_valid_strict,
                    "pred_labels_valid": y_pred_valid_strict,
                    "parse_error_indices": strict_parse_error_indices,
                    "parse_error_count": len(strict_parse_error_indices)
                },
                "lenient_parsing": {
                    "true_labels_all": y_true_all_lenient,
                    "pred_labels_all": y_pred_all_lenient,
                    "true_labels_valid": y_true_valid_lenient,
                    "pred_labels_valid": y_pred_valid_lenient,
                    "parse_error_indices": lenient_parse_error_indices,
                    "parse_error_count": len(lenient_parse_error_indices)
                },
                "shared_data": {
                    "inference_times": inference_times,
                    "raw_responses": raw_responses,
                    "total_samples": len(y_true_all_strict)
                },
                "parse_differences": parse_differences,
                "parse_difference_count": len(parse_differences)
            }, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.save_dir, f"strict_error_log_{timestamp}.json"), "w") as f: 
            json.dump(strict_error_log, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.save_dir, f"lenient_error_log_{timestamp}.json"), "w") as f: 
            json.dump(lenient_error_log, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.save_dir, f"parse_differences_{timestamp}.json"), "w") as f: 
            json.dump(parse_differences, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(self.save_dir, f"metrics_{timestamp}.json"), "w") as f: 
            json.dump({
                "strict_parsing_metrics": {"metrics_all": metrics_all_strict, "metrics_valid": metrics_valid_strict},
                "lenient_parsing_metrics": {"metrics_all": metrics_all_lenient, "metrics_valid": metrics_valid_lenient}
            }, f, indent=2, ensure_ascii=False)

    def save_detailed_report(self, metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient, 
                             inference_times, strict_error_log, lenient_error_log, strict_parse_error_indices, 
                             lenient_parse_error_indices, parse_differences, total_samples):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            param_ratio = 100 * trainable_params / all_param
        except Exception:
            trainable_params, all_param, param_ratio = 0, 0, 0
            
        with open(os.path.join(self.save_dir, f"detailed_report_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"            Unified Multitask Model Performance Report ({self.split_name})\n")
            f.write("=" * 80 + "\n\n")
            f.write("Model parameter statistics:\n")
            f.write(f"Total parameters (Base + LoRA): {all_param:,}\n")
            f.write(f"Trainable parameters (LoRA): {trainable_params:,}\n")
            f.write(f"Trainable parameter ratio: {param_ratio:.4f}%\n\n")
            f.write("Dataset statistics:\n")
            f.write(f"Task name: {self.split_name}\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Average inference time: {sum(inference_times)/len(inference_times):.4f}s\n\n")
            f.write("Strict parsing metrics:\n")
            f.write(f"Accuracy: {metrics_all_strict['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics_all_strict['precision']:.4f}\n")
            f.write(f"Recall: {metrics_all_strict['recall']:.4f}\n")
            f.write(f"F1 score: {metrics_all_strict['f1']:.4f}\n")
            f.write(f"Parse errors: {len(strict_error_log)}\n\n")
            f.write("Lenient parsing metrics:\n")
            f.write(f"Accuracy: {metrics_all_lenient['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics_all_lenient['precision']:.4f}\n")
            f.write(f"Recall: {metrics_all_lenient['recall']:.4f}\n")
            f.write(f"F1 score: {metrics_all_lenient['f1']:.4f}\n")
            f.write(f"Parse errors: {len(lenient_error_log)}\n")
            f.write("=" * 80 + "\n")
            
    def print_metrics(self, metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient):
        print(f"\n{' Performance Report ('+self.split_name+') ':=^80}")
        print(f"{'Parsing method / dataset':<45} {'Samples':<8} {'Accuracy':<8} {'Precision':<9} {'Recall':<8} {'F1':<8}")
        print("-" * 80)
        print(f"{metrics_all_strict['dataset_type']:<45} {metrics_all_strict['sample_count']:<8} {metrics_all_strict['accuracy']:.4f}    {metrics_all_strict['precision']:.4f}     {metrics_all_strict['recall']:.4f}    {metrics_all_strict['f1']:.4f}")
        print(f"{metrics_all_lenient['dataset_type']:<45} {metrics_all_lenient['sample_count']:<8} {metrics_all_lenient['accuracy']:.4f}    {metrics_all_lenient['precision']:.4f}     {metrics_all_lenient['recall']:.4f}    {metrics_all_lenient['f1']:.4f}")
        print("=" * 80)

def main() -> None:
    run_id = f"unified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_model_dir = os.path.join(CONFIG["model_root"], f"{CONFIG['base_model_name']}_lora_{run_id}")
    run_result_dir = os.path.join(CONFIG["results_root"], run_id)
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_result_dir, exist_ok=True)
    config_save_path = os.path.join(run_result_dir, f"experiment_config_{run_id}.txt")
    with open(config_save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("          Unified Multitask Fine-tuning Configuration\n")
        f.write(f"          Run ID: {run_id}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration (CONFIG)\n")
        f.write(json.dumps(CONFIG, indent=4, ensure_ascii=False))
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Additional training components\n")
        f.write("Label Smoothing: 0.05\n")
        f.write("LoRA Dropout: 0.1\n")
        f.write("=" * 60 + "\n")
    print(f"Experiment configuration saved to: {config_save_path}")

    base_model_path = CONFIG["base_model_path"]
    unified_train_samples, all_valid_datasets, all_test_datasets = load_multi_task_data(CONFIG["tasks"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        attn_implementation="sdpa", 
        device_map="auto",
        trust_remote_code=True,
    )

    # Keep all major projection layers trainable for multitask adaptation.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"\nTrainable parameter ratio: {100 * trainable_params / all_param:.4f}%\n")
    
    train_ds = make_dataset(unified_train_samples, tokenizer, CONFIG["max_length"])
    if len(train_ds) == 0:
        raise ValueError("The tokenized training set is empty. Increase max_length or shorten the input text.")
    
    unified_valid_samples = sample_balanced_valid_data(all_valid_datasets, samples_per_task=400)
    
    valid_ds = make_dataset(unified_valid_samples, tokenizer, CONFIG["max_length"])
    if len(valid_ds) == 0:
        raise ValueError("The tokenized validation set is empty. Increase max_length or shorten the input text.")
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    
    training_args = TrainingArguments(
        output_dir=run_model_dir,
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=0.001,
        label_smoothing_factor=0.05,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"], 
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps", 
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=20,
        
        bf16=torch.cuda.is_available(),
        report_to=["none"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nStarting unified multitask LoRA fine-tuning...")
    trainer.train()

    adapter_dir = os.path.join(run_model_dir, "best_lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nTraining complete. Starting per-task evaluation...")
    tokenizer.padding_side = "left"

    all_summaries = {"run_id": run_id, "config": CONFIG}

    for task_name in CONFIG["tasks"].keys():
        print(f"\n{'='*60}\nEvaluating task: {task_name}\n{'='*60}")
        if task_name not in all_valid_datasets or len(all_valid_datasets[task_name]) == 0:
            continue
            
        valid_evaluator = LoraZeroShotEvaluator(
            model=trainer.model, tokenizer=tokenizer, data=all_valid_datasets[task_name], 
            save_dir=os.path.join(run_result_dir, task_name, "valid"), split_name=task_name,
            max_length=CONFIG["max_length"], max_new_tokens=10
        )
        v_metrics = valid_evaluator.run()

        test_evaluator = LoraZeroShotEvaluator(
            model=trainer.model, tokenizer=tokenizer, data=all_test_datasets[task_name], 
            save_dir=os.path.join(run_result_dir, task_name, "test"), split_name=task_name,
            max_length=CONFIG["max_length"], max_new_tokens=10
        )
        t_metrics = test_evaluator.run()
        
        all_summaries[task_name] = {"valid": v_metrics, "test": t_metrics}

    with open(os.path.join(run_result_dir, "unified_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\nAll tasks are complete. The unified multitask model and evaluation results have been saved.")

if __name__ == "__main__":
    main()
