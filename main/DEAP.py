import os
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
import datetime
from datasets import load_dataset
import re
import time
from tqdm import tqdm

torch.cuda.empty_cache()
def set_seed(seed=2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

set_seed(2025)

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["DISABLE_EXLLAMA_KERNELS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Valence dimension by default. For Arousal, change the default paths in parse_for_llm() and
# replace strict_parse_response() and lenient_parse_response() with the Arousal versions commented out below.
def parse_for_llm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_dir', help='path to the test dataset (json file)',
                        default=os.path.join(os.path.dirname(__file__), 'datasets_json','deap','valence.json'))
    parser.add_argument('--save_result_dir', help='path to save the prediction resutls',
                        default=os.path.join(os.path.dirname(__file__), 'results','deap','valence'))
    parser.add_argument('--llms_dir', help='path to pre-train LLM',
                        default=os.path.join(os.path.dirname(__file__), 'models'))
    parser.add_argument('--device_map', help='GPU to use',
                        default={'gpu':[0]})
    parser.add_argument('--model_names', nargs='+', help='specific model names to test (if not provided, test all models)',
                        default=None)
    parser.add_argument('--skip_models', nargs='+', help='model names to skip',
                        default=[])
    args = parser.parse_args()
    return args


def get_available_models(models_base_dir, specific_models=None, skip_models=None):
    if not os.path.exists(models_base_dir):
        return []
    all_models = []
    for item in os.listdir(models_base_dir):
         all_models.append(item)
    
    if specific_models:
        all_models = [model for model in all_models if model in specific_models]
    if skip_models:
        all_models = [model for model in all_models if model not in skip_models]
    
    return sorted(all_models)

class ZeroShotTester:
    def __init__(self, test_dataset_dir: str, save_result_dir: str, 
                 llms_dir: str, device_map: str = "auto",
                 temperature: float = 0.1, top_p: float = 0.9, top_k: float = 50, batch_size: int = 4):
        self.test_dataset_dir = test_dataset_dir 
        self.save_result_dir = save_result_dir 
        self.llms_dir = llms_dir 
        self.device_map = device_map 
        # Prompt template
        self.prompt_template = """{instruction}
        ### Input: 
        {input}
        ### Output:
        """
    
    def __call__(self):
        self.zero_shot_test()
        
    
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llms_dir) 
        # Vision-language models (Qwen3-VL, LLaVA) are loaded with their own classes and used in text-only mode
        model_type = AutoConfig.from_pretrained(self.llms_dir).model_type
        if model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
        elif model_type == "llava":
            from transformers import LlavaForConditionalGeneration
            model_class = LlavaForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(
            self.llms_dir,
            dtype=torch.bfloat16,
            device_map="auto"
        ).eval() 
        return model, tokenizer

    def load_test_data(self):
        with open(self.test_dataset_dir) as f:
            return json.load(f) 
 
    def strict_parse_response(self, response: str) -> int:
        """Strict parser"""
        response_parts = response.split("### Output:", 1)
        if len(response_parts) > 1:
            response_content = response_parts[1]
        else:
            return None
        pattern_all = r'\b(negative|positive|0|1)\b'
        all_matches = re.findall(pattern_all, response_content, re.IGNORECASE)
        if len(all_matches) != 1:
            return None
        pattern = r"Identify_emotion:\s*\b(negative|positive|0|1)\b"
        match = re.search(pattern, response_content.strip(), re.IGNORECASE)
        
        if match:
            emotion_str = match.group(1).lower()
            emotion_map = {
                "negative": 0, "0": 0,
                "positive": 1, "1": 1
            }
            return emotion_map.get(emotion_str)
        return None

    def lenient_parse_response(self, response: str) -> int:
        """Lenient parser"""
        response_parts = response.split("### Output:", 1)
        if len(response_parts) > 1:
            response_content = response_parts[1]
        else:
            return None  
    
        negative_pattern = r'\b(negative|0)\b'
        positive_pattern = r'\b(positive|1)\b'
        
        has_negative = bool(re.search(negative_pattern, response_content, re.IGNORECASE))
        has_positive = bool(re.search(positive_pattern, response_content, re.IGNORECASE))
        
        if has_negative and not has_positive:
            return 0
        elif has_positive and not has_negative:
            return 1
        else:
            return None
    # Arousal dimension: uncomment the two parsers below to replace the Valence versions above
    # def strict_parse_response(self, response: str) -> int:
    #     """Strict parser"""
    #     response_parts = response.split("### Output:", 1)
    #     if len(response_parts) > 1:
    #         response_content = response_parts[1]
    #     else:
    #         return None

    #     pattern_all = r'\b(low|high|0|1)\b'
    #     all_matches = re.findall(pattern_all, response_content, re.IGNORECASE)
       
    #     if len(all_matches) != 1:
    #         return None

    #     pattern = r"Identify_emotion:\s*\b(low|high|0|1)\b"
    #     match = re.search(pattern, response_content.strip(), re.IGNORECASE)
        
    #     if match:
    #         emotion_str = match.group(1).lower()
    #         emotion_map = {
    #             "low": 0, "0": 0,
    #             "high": 1, "1": 1
    #         }
    #         return emotion_map.get(emotion_str)
    #     return None


    # def lenient_parse_response(self, response: str) -> int:
    #     """Lenient parser"""
    #     response_parts = response.split("### Output:", 1)
    #     if len(response_parts) > 1:
    #         response_content = response_parts[1]
    #     else:
    #         return None  
    #     low_pattern = r'\b(low|0)\b'
    #     high_pattern = r'\b(high|1)\b'
        
    #     has_low = bool(re.search(low_pattern, response_content, re.IGNORECASE))
    #     has_high = bool(re.search(high_pattern, response_content, re.IGNORECASE))
    #     if has_low and not has_high:
    #         return 0
    #     elif has_high and not has_low:
    #         return 1
    #     else:
    #         return None
    def zero_shot_test(self, *args: any, **kwds: any) -> any:
        """ Core zero-shot testing workflow """
        # Create the result directory if it does not exist
        os.makedirs(self.save_result_dir, exist_ok=True)
        
        # Load model and test data
        model, tokenizer = self.load_model()
        test_data = self.load_test_data()
        
        # Initialize containers for strict parsing
        y_true_all_strict = []
        y_pred_all_strict = []
        y_true_valid_strict = []
        y_pred_valid_strict = []
        strict_parse_error_indices = []
        strict_error_log = []
        
        # Initialize containers for lenient parsing
        y_true_all_lenient = []
        y_pred_all_lenient = []
        y_true_valid_lenient = []
        y_pred_valid_lenient = []
        lenient_parse_error_indices = []
        lenient_error_log = []
        
        # Shared data (inference times, raw responses, etc.)
        inference_times = []
        raw_responses = []
        
        # Parsing differences records
        parse_differences = []

        # Process each sample
        for idx, sample in tqdm(enumerate(test_data), total=len(test_data), desc="Testing", unit="sample", ncols=100):
            try:
                # Construct the full prompt
                full_prompt = self.prompt_template.format(
                    instruction=sample["instruction"],
                    input=sample["input"]
                )
                
                # Tokenize and prepare inputs for the model
                inputs = tokenizer(full_prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
                
                # Model inference
                start_time = time.perf_counter()
                with torch.no_grad():  # Disable gradient computation for inference
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,  # Limit generated text length
                        pad_token_id=tokenizer.eos_token_id,  # Use EOS token for padding
                    )
                # Record inference time
                inference_time = time.perf_counter() - start_time
                inference_times.append(inference_time)
                
                # Decode model output
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_responses.append(response)
            
                true_label = int(sample["output"].split(": ")[-1].strip("."))
                
                # Parse the model response
                strict_pred = self.strict_parse_response(response)
                lenient_pred = self.lenient_parse_response(response)
            
                # Strict parsing results
                y_true_all_strict.append(true_label)
                if strict_pred is not None and strict_pred in {0, 1}:
                    y_pred_all_strict.append(strict_pred)
                    y_true_valid_strict.append(true_label)
                    y_pred_valid_strict.append(strict_pred)
                else:
                    error_pred = 1 - true_label
                    y_pred_all_strict.append(error_pred)
                    strict_parse_error_indices.append(idx)
                    strict_error_log.append({
                        "index": idx,
                        "input": sample["input"],
                        "true_label": true_label,
                        "assigned_pred": error_pred,
                        "raw_response": response,
                        "error_type": "strict_parse_error"
                    })
                
                # Lenient parsing results
                y_true_all_lenient.append(true_label)
                if lenient_pred is not None and lenient_pred in {0, 1}:
                    y_pred_all_lenient.append(lenient_pred)
                    y_true_valid_lenient.append(true_label)
                    y_pred_valid_lenient.append(lenient_pred)
                else:
                    error_pred = 1 - true_label
                    y_pred_all_lenient.append(error_pred)
                    lenient_parse_error_indices.append(idx)
                    lenient_error_log.append({
                        "index": idx,
                        "input": sample["input"],
                        "true_label": true_label,
                        "assigned_pred": error_pred,
                        "raw_response": response,
                        "error_type": "lenient_parse_error"
                    })
                
                # Record parsing differences
                if strict_pred != lenient_pred:
                    parse_differences.append({
                        "index": idx,
                        "input": sample["input"],
                        "true_label": true_label,
                        "strict_pred": strict_pred,
                        "lenient_pred": lenient_pred,
                        "raw_response": response
                    })
            except Exception as e:
                # Exception handling
                true_label = int(sample["output"].split(": ")[-1].strip("."))
                
                # Record inference time as 0 due to exception
                inference_times.append(0)
                raw_responses.append("")
                
                # Strict parsing exception handling
                y_true_all_strict.append(true_label)
                error_pred = 1 - true_label
                y_pred_all_strict.append(error_pred)
                strict_parse_error_indices.append(idx)
                strict_error_log.append({
                    "index": idx,
                    "error": str(e),
                    "true_label": true_label,
                    "assigned_pred": error_pred,
                    "error_type": "exception"
                })
                
                # Lenient parsing exception handling
                y_true_all_lenient.append(true_label)
                y_pred_all_lenient.append(error_pred)
                lenient_parse_error_indices.append(idx)
                lenient_error_log.append({
                    "index": idx,
                    "error": str(e),
                    "true_label": true_label,
                    "assigned_pred": error_pred,
                    "error_type": "exception"
                })

        # Calculate metrics for both parsing methods
        metrics_all_strict = self.calculate_metrics(y_true_all_strict, y_pred_all_strict, "Strict Parsing - Full Dataset")
        metrics_valid_strict = self.calculate_metrics(y_true_valid_strict, y_pred_valid_strict, "Strict Parsing - Valid Predictions") if y_true_valid_strict else {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": "Strict Parsing - Valid Predictions", "sample_count": 0
        }
        
        metrics_all_lenient = self.calculate_metrics(y_true_all_lenient, y_pred_all_lenient, "Lenient Parsing - Full Dataset")
        metrics_valid_lenient = self.calculate_metrics(y_true_valid_lenient, y_pred_valid_lenient, "Lenient Parsing - Valid Predictions") if y_true_valid_lenient else {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": "Lenient Parsing - Valid Predictions", "sample_count": 0
        }
        
        # Save results
        self.save_results(
            # Strict parsing results
            y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict,
            strict_error_log, strict_parse_error_indices,
            # Lenient parsing results
            y_true_all_lenient, y_pred_all_lenient, y_true_valid_lenient, y_pred_valid_lenient,
            lenient_error_log, lenient_parse_error_indices,
            # Shared data
            inference_times, raw_responses,
            # Parsing differences
            parse_differences,
            # Metrics
            metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient
        )
        
        # Print metrics summary
        self.print_metrics(
            metrics_all_strict, metrics_valid_strict, 
            metrics_all_lenient, metrics_valid_lenient,
            inference_times,
            strict_error_log, lenient_error_log, 
            strict_parse_error_indices, lenient_parse_error_indices, 
            parse_differences,
            len(test_data)
        )
    
        return metrics_all_strict

    def calculate_metrics(self, y_true, y_pred, dataset_type):
        """ Calculate performance indicators """
        if not y_true:
            return {
                "accuracy": 0, 
                "precision": 0, 
                "recall": 0, 
                "f1": 0, 
                "dataset_type": dataset_type,
                "sample_count": 0
            }
        
        if len(set(y_true)) == 1:
            accuracy = accuracy_score(y_true, y_pred)
            precision = 1.0 if y_true[0] == 1 and sum(y_pred) == len(y_pred) else 0.0
            recall = 1.0 if y_true[0] == 1 and sum(y_pred) == len(y_pred) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
            recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dataset_type": dataset_type,
            "sample_count": len(y_true)
        }
    def save_results(self, 
                 # Strict parsing results
                 y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict,
                 strict_error_log, strict_parse_error_indices,
                 # Lenient parsing results
                 y_true_all_lenient, y_pred_all_lenient, y_true_valid_lenient, y_pred_valid_lenient,
                 lenient_error_log, lenient_parse_error_indices,
                 # Shared data
                 inference_times, raw_responses,
                 # Parsing differences
                 parse_differences,
                 # Metrics
                 metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient):
        """Save results for both parsing methods"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_dir = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(self.save_result_dir, date_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save prediction results
        with open(os.path.join(save_dir, f"predictions_{timestamp}.json"), "w") as f:
            json.dump({
                # Strict parsing
                "strict_parsing": {
                    "true_labels_all": y_true_all_strict,
                    "pred_labels_all": y_pred_all_strict,
                    "true_labels_valid": y_true_valid_strict,
                    "pred_labels_valid": y_pred_valid_strict,
                    "parse_error_indices": strict_parse_error_indices,
                    "parse_error_count": len(strict_parse_error_indices)
                },
                # Lenient parsing
                "lenient_parsing": {
                    "true_labels_all": y_true_all_lenient,
                    "pred_labels_all": y_pred_all_lenient,
                    "true_labels_valid": y_true_valid_lenient,
                    "pred_labels_valid": y_pred_valid_lenient,
                    "parse_error_indices": lenient_parse_error_indices,
                    "parse_error_count": len(lenient_parse_error_indices)
                },
                # Shared data
                "shared_data": {
                    "inference_times": inference_times,
                    "raw_responses": raw_responses,
                    "total_samples": len(y_true_all_strict)
                },
                # Parsing differences
                "parse_differences": parse_differences,
                "parse_difference_count": len(parse_differences)
            }, f, indent=2)
        
        # Save error logs separately
        with open(os.path.join(save_dir, f"strict_error_log_{timestamp}.json"), "w") as f:
            json.dump(strict_error_log, f, indent=2)
            
        with open(os.path.join(save_dir, f"lenient_error_log_{timestamp}.json"), "w") as f:
            json.dump(lenient_error_log, f, indent=2)
            
        # Save parsing differences
        with open(os.path.join(save_dir, f"parse_differences_{timestamp}.json"), "w") as f:
            json.dump(parse_differences, f, indent=2)
            
        # Save metrics
        with open(os.path.join(save_dir, f"metrics_{timestamp}.json"), "w") as f:
            json.dump({
                "strict_parsing_metrics": {
                    "metrics_all": metrics_all_strict,
                    "metrics_valid": metrics_valid_strict
                },
                "lenient_parsing_metrics": {
                    "metrics_all": metrics_all_lenient,
                    "metrics_valid": metrics_valid_lenient
                }
            }, f, indent=2)

        
            
    def print_metrics(self, metrics_all_strict, metrics_valid_strict, 
                  metrics_all_lenient, metrics_valid_lenient,
                  inference_times,
                  strict_error_log, lenient_error_log, 
                  strict_parse_error_indices, lenient_parse_error_indices, 
                  parse_differences, total_samples):
        """Print the performance metrics report for both parsing methods"""
        strict_parse_error_count = len(strict_parse_error_indices)
        lenient_parse_error_count = len(lenient_parse_error_indices)
        parse_difference_count = len(parse_differences)
        
        # Count different types of errors
        strict_exception_count = len([e for e in strict_error_log if e.get("error_type") == "exception"])
        strict_parse_error_count_log = len([e for e in strict_error_log if e.get("error_type") == "strict_parse_error"])
        
        lenient_exception_count = len([e for e in lenient_error_log if e.get("error_type") == "exception"])
        lenient_parse_error_count_log = len([e for e in lenient_error_log if e.get("error_type") == "lenient_parse_error"])
        
        print(f"\n{' Performance Metrics Report ':=^80}")
        print(f"{'Parsing Method / Dataset':<25} {'Samples':<8} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1 Score':<8}")
        print("-" * 80)
        
        # Strict parsing results
        print(f"{metrics_all_strict['dataset_type']:<25} {metrics_all_strict['sample_count']:<8} {metrics_all_strict['accuracy']:.4f}    "
            f"{metrics_all_strict['precision']:.4f}    {metrics_all_strict['recall']:.4f}    {metrics_all_strict['f1']:.4f}")
        print(f"{metrics_valid_strict['dataset_type']:<25} {metrics_valid_strict['sample_count']:<8} {metrics_valid_strict['accuracy']:.4f}    "
            f"{metrics_valid_strict['precision']:.4f}    {metrics_valid_strict['recall']:.4f}    {metrics_valid_strict['f1']:.4f}")
        
        # Lenient parsing results
        print(f"{metrics_all_lenient['dataset_type']:<25} {metrics_all_lenient['sample_count']:<8} {metrics_all_lenient['accuracy']:.4f}    "
            f"{metrics_all_lenient['precision']:.4f}    {metrics_all_lenient['recall']:.4f}    {metrics_all_lenient['f1']:.4f}")
        print(f"{metrics_valid_lenient['dataset_type']:<25} {metrics_valid_lenient['sample_count']:<8} {metrics_valid_lenient['accuracy']:.4f}    "
            f"{metrics_valid_lenient['precision']:.4f}    {metrics_valid_lenient['recall']:.4f}    {metrics_valid_lenient['f1']:.4f}")
        print("=" * 80)
        
        print(f"\n{' Inference Time Statistics ':-^40}")
        print(f"Average inference time: {sum(inference_times)/len(inference_times):.4f}s")
        print(f"Total inference time: {sum(inference_times):.4f}s")
        print(f"Maximum inference time: {max(inference_times):.4f}s")
        print(f"Minimum inference time: {min([t for t in inference_times if t > 0]) if any(t > 0 for t in inference_times) else 0:.4f}s")
        
        print(f"\n{' Error Statistics ':-^40}")
        print("Strict Parsing Errors:")
        print(f"  Parse errors: {strict_parse_error_count_log}/{total_samples} ({strict_parse_error_count_log/total_samples:.4f})")
        print(f"  Exceptions: {strict_exception_count}/{total_samples} ({strict_exception_count/total_samples:.4f})")
        print(f"  Total errors: {len(strict_error_log)}/{total_samples} ({len(strict_error_log)/total_samples:.4f})")
        
        print("Lenient Parsing Errors:")
        print(f"  Parse errors: {lenient_parse_error_count_log}/{total_samples} ({lenient_parse_error_count_log/total_samples:.4f})")
        print(f"  Exceptions: {lenient_exception_count}/{total_samples} ({lenient_exception_count/total_samples:.4f})")
        print(f"  Total errors: {len(lenient_error_log)}/{total_samples} ({len(lenient_error_log)/total_samples:.4f})")
        
        print(f"Samples with parsing differences: {parse_difference_count}/{total_samples} ({parse_difference_count/total_samples:.4f})")
        
        # Save detailed text report
        self.save_detailed_report(
            metrics_all_strict, metrics_valid_strict, 
            metrics_all_lenient, metrics_valid_lenient,
            inference_times,
            strict_error_log, lenient_error_log, 
            strict_parse_error_indices, lenient_parse_error_indices, 
            parse_differences, total_samples
        )

    
    def save_detailed_report(self, metrics_all_strict, metrics_valid_strict, 
                         metrics_all_lenient, metrics_valid_lenient,
                         inference_times,
                         strict_error_log, lenient_error_log, 
                         strict_parse_error_indices, lenient_parse_error_indices, 
                         parse_differences, total_samples):
        """Save a detailed text report"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_dir = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(self.save_result_dir, date_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, f"detailed_report_{timestamp}.txt"), "w") as f:
            f.write("=" * 80 + "\n")
            f.write("         Detailed Report of Dual Parsing Method Performance\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Valid samples (strict parsing): {metrics_valid_strict['sample_count']}\n")
            f.write(f"Valid samples (lenient parsing): {metrics_valid_lenient['sample_count']}\n")
            f.write(f"Errors (strict parsing): {len(strict_error_log)}\n")
            f.write(f"Errors (lenient parsing): {len(lenient_error_log)}\n")
            f.write(f"Samples with parsing differences: {len(parse_differences)}\n\n")
            
            f.write("Strict Parsing Metrics:\n")
            f.write(f"Accuracy (all samples): {metrics_all_strict['accuracy']:.4f}\n")
            f.write(f"Precision (all samples): {metrics_all_strict['precision']:.4f}\n")
            f.write(f"Recall (all samples): {metrics_all_strict['recall']:.4f}\n")
            f.write(f"F1 score (all samples): {metrics_all_strict['f1']:.4f}\n")
            f.write(f"Accuracy (valid predictions): {metrics_valid_strict['accuracy']:.4f}\n")
            f.write(f"Precision (valid predictions): {metrics_valid_strict['precision']:.4f}\n")
            f.write(f"Recall (valid predictions): {metrics_valid_strict['recall']:.4f}\n")
            f.write(f"F1 score (valid predictions): {metrics_valid_strict['f1']:.4f}\n\n")
            
            f.write("Lenient Parsing Metrics:\n")
            f.write(f"Accuracy (all samples): {metrics_all_lenient['accuracy']:.4f}\n")
            f.write(f"Precision (all samples): {metrics_all_lenient['precision']:.4f}\n")
            f.write(f"Recall (all samples): {metrics_all_lenient['recall']:.4f}\n")
            f.write(f"F1 score (all samples): {metrics_all_lenient['f1']:.4f}\n")
            f.write(f"Accuracy (valid predictions): {metrics_valid_lenient['accuracy']:.4f}\n")
            f.write(f"Precision (valid predictions): {metrics_valid_lenient['precision']:.4f}\n")
            f.write(f"Recall (valid predictions): {metrics_valid_lenient['recall']:.4f}\n")
            f.write(f"F1 score (valid predictions): {metrics_valid_lenient['f1']:.4f}\n\n")
            
            f.write("Parsing Method Comparison:\n")
            f.write(f"Strict parsing better than lenient parsing (F1): {metrics_all_strict['f1'] > metrics_all_lenient['f1']}\n")
            f.write(f"F1 score difference: {metrics_all_strict['f1'] - metrics_all_lenient['f1']:.4f}\n")
            f.write(f"Error rate (strict): {len(strict_error_log)/total_samples:.4f}\n")
            f.write(f"Error rate (lenient): {len(lenient_error_log)/total_samples:.4f}\n")
            f.write(f"Error rate difference: {(len(strict_error_log) - len(lenient_error_log))/total_samples:.4f}\n\n")
            
            f.write("Inference Time Statistics:\n")
            f.write(f"Average inference time: {sum(inference_times)/len(inference_times):.4f}s\n")
            f.write(f"Total inference time: {sum(inference_times):.4f}s\n")
            f.write(f"Max inference time: {max(inference_times):.4f}s\n")
            f.write(f"Min inference time: {min([t for t in inference_times if t > 0]) if any(t > 0 for t in inference_times) else 0:.4f}s\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")



def batch_test_models():
    """Batch test all models"""
    args = parse_for_llm()
    
    # Get the list of available models
    available_models = get_available_models(
        args.llms_dir, 
        specific_models=args.model_names,
        skip_models=args.skip_models
    )
    
    print(type(available_models), 'available_models info')
    if not available_models:
        print("No available models found")
        return
    print(f"Found {len(available_models)} available models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    # Store test results for all models
    all_results = {}
    
    # Test each model in sequence
    for model_name in available_models:
        print(f"\n{'='*80}")
        print(f"Starting test for model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Set the path for the current model
            current_llm_dir = os.path.join(args.llms_dir, model_name)
            current_save_dir = os.path.join(args.save_result_dir, model_name)
            
            # Create a tester instance
            tester = ZeroShotTester(
                test_dataset_dir=args.test_dataset_dir,
                save_result_dir=current_save_dir,
                llms_dir=current_llm_dir,
                device_map=args.device_map
            )
            
            result = tester.zero_shot_test()
            if result:
                all_results[model_name] = result
                
        except Exception as e:
            continue
    
    # Save and print the summary results of batch testing
    if all_results:
        save_batch_results(all_results, args.save_result_dir)
        print_batch_summary(all_results)


def save_batch_results(all_results, save_base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = datetime.datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(save_base_dir, date_dir)
    os.makedirs(save_dir, exist_ok=True)

    serializable_results = {}
    for model_name, metrics in all_results.items():
        serializable_results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'sample_count': int(metrics['sample_count'])
        }
    
    with open(os.path.join(save_dir, f"batch_results_{timestamp}.json"), "w") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

def print_batch_summary(all_results):
    """Print summary report for batch testing"""
    print(f"\n{' Batch Test Summary Report ':=^80}")
    print(f"{'Model Name':<30} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1 Score':<8}")
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} {metrics['accuracy']:.4f}    "
              f"{metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")
    
    # Print the best model
    if sorted_results:
        best_model, best_metrics = sorted_results[0]
        print(f"\nBest model: {best_model}")
        print(f"Best F1 score: {best_metrics['f1']:.4f}")
        print(f"Best accuracy: {best_metrics['accuracy']:.4f}")



if __name__ == '__main__':
    batch_test_models()
   