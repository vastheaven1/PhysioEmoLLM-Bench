import os
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM,AutoModel
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
                        default=os.path.join(os.path.dirname(__file__), 'datasets_json','dreamer','valence.json'))
    parser.add_argument('--save_result_dir', help='path to save the prediction resutls',
                        default=os.path.join(os.path.dirname(__file__), 'results','dreamer','valence'))
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
        model = AutoModelForCausalLM.from_pretrained(
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
        
        # Ensure result directory exists
        os.makedirs(self.save_result_dir, exist_ok=True)
        
        # Load model and test data
        model, tokenizer = self.load_model()
        test_data = self.load_test_data()
        
        # Initialize result containers - Strict parsing
        y_true_all_strict, y_pred_all_strict = [], []
        y_true_valid_strict, y_pred_valid_strict = [], []
        strict_parse_error_indices, strict_error_log = [], []
        
        # Initialize result containers - Lenient parsing
        y_true_all_lenient, y_pred_all_lenient = [], []
        y_true_valid_lenient, y_pred_valid_lenient = [], []
        lenient_parse_error_indices, lenient_error_log = [], []
        
        # Shared data
        inference_times, raw_responses = [], []
        
        # Parse differences
        parse_differences = []
        
        # Process each sample
        for idx, sample in tqdm(enumerate(test_data), total=len(test_data), desc="Testing", unit="sample", ncols=100):
            try:
                # Build full prompt
                full_prompt = self.prompt_template.format(
                    instruction=sample["instruction"],
                    input=sample["input"]
                )
                
                # Tokenize
                inputs = tokenizer(full_prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
                
                # Model inference
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(**inputs,
                                            max_new_tokens=10,
                                            pad_token_id=tokenizer.eos_token_id)
                inference_time = time.perf_counter() - start_time
                inference_times.append(inference_time)
                
                # Decode model output
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_responses.append(response)
                
                # True label
                true_label = int(sample["output"].split(": ")[-1].strip("."))
                
                # Parse predictions
                strict_pred = self.strict_parse_response(response)
                lenient_pred = self.lenient_parse_response(response)
                
                # Strict parsing
                y_true_all_strict.append(true_label)
                if strict_pred in {0, 1}:
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
                
                # Lenient parsing
                y_true_all_lenient.append(true_label)
                if lenient_pred in {0, 1}:
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
                # Handle exceptions
                inference_times.append(0)
                raw_responses.append("")
                
                error_pred = 1 - true_label
                
                # Strict parsing exception
                y_true_all_strict.append(true_label)
                y_pred_all_strict.append(error_pred)
                strict_parse_error_indices.append(idx)
                strict_error_log.append({
                    "index": idx,
                    "error": str(e),
                    "true_label": true_label,
                    "assigned_pred": error_pred,
                    "error_type": "exception"
                })
                
                # Lenient parsing exception
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
        
        # Compute metrics
        metrics_all_strict = self.calculate_metrics(y_true_all_strict, y_pred_all_strict, "Strict Parsing - All Samples")
        metrics_valid_strict = self.calculate_metrics(y_true_valid_strict, y_pred_valid_strict, "Strict Parsing - Valid Predictions") if y_true_valid_strict else {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": "Strict Parsing - Valid Predictions", "sample_count": 0
        }
        
        metrics_all_lenient = self.calculate_metrics(y_true_all_lenient, y_pred_all_lenient, "Lenient Parsing - All Samples")
        metrics_valid_lenient = self.calculate_metrics(y_true_valid_lenient, y_pred_valid_lenient, "Lenient Parsing - Valid Predictions") if y_true_valid_lenient else {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "dataset_type": "Lenient Parsing - Valid Predictions", "sample_count": 0
        }
        
        # Save results
        self.save_results(
            y_true_all_strict, y_pred_all_strict, y_true_valid_strict, y_pred_valid_strict,
            strict_error_log, strict_parse_error_indices,
            y_true_all_lenient, y_pred_all_lenient, y_true_valid_lenient, y_pred_valid_lenient,
            lenient_error_log, lenient_parse_error_indices,
            inference_times, raw_responses,
            parse_differences,
            metrics_all_strict, metrics_valid_strict, metrics_all_lenient, metrics_valid_lenient
        )
        
        # Print metrics
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

        
            
    def print_metrics(self, metrics_all_strict, metrics_valid_strict, 
                  metrics_all_lenient, metrics_valid_lenient,
                  inference_times,
                  strict_error_log, lenient_error_log, 
                  strict_parse_error_indices, lenient_parse_error_indices, 
                  parse_differences, total_samples):
        """Print performance metrics for both parsing methods (Strict vs Lenient)"""
        
        strict_parse_error_count = len(strict_parse_error_indices)
        lenient_parse_error_count = len(lenient_parse_error_indices)
        parse_difference_count = len(parse_differences)
        
        # Count different types of errors
        strict_exception_count = len([e for e in strict_error_log if e.get("error_type") == "exception"])
        strict_parse_error_count_log = len([e for e in strict_error_log if e.get("error_type") == "strict_parse_error"])
        
        lenient_exception_count = len([e for e in lenient_error_log if e.get("error_type") == "exception"])
        lenient_parse_error_count_log = len([e for e in lenient_error_log if e.get("error_type") == "lenient_parse_error"])
        
        print(f"\n{' PERFORMANCE REPORT ':=^80}")
        print(f"{'Parsing Method / Dataset':<25} {'#Samples':<8} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1-score':<8}")
        print("-" * 80)
        
        # Strict parsing metrics
        print(f"{metrics_all_strict['dataset_type']:<25} {metrics_all_strict['sample_count']:<8} "
            f"{metrics_all_strict['accuracy']:.4f}    {metrics_all_strict['precision']:.4f}    "
            f"{metrics_all_strict['recall']:.4f}    {metrics_all_strict['f1']:.4f}")
        print(f"{metrics_valid_strict['dataset_type']:<25} {metrics_valid_strict['sample_count']:<8} "
            f"{metrics_valid_strict['accuracy']:.4f}    {metrics_valid_strict['precision']:.4f}    "
            f"{metrics_valid_strict['recall']:.4f}    {metrics_valid_strict['f1']:.4f}")
        
        # Lenient parsing metrics
        print(f"{metrics_all_lenient['dataset_type']:<25} {metrics_all_lenient['sample_count']:<8} "
            f"{metrics_all_lenient['accuracy']:.4f}    {metrics_all_lenient['precision']:.4f}    "
            f"{metrics_all_lenient['recall']:.4f}    {metrics_all_lenient['f1']:.4f}")
        print(f"{metrics_valid_lenient['dataset_type']:<25} {metrics_valid_lenient['sample_count']:<8} "
            f"{metrics_valid_lenient['accuracy']:.4f}    {metrics_valid_lenient['precision']:.4f}    "
            f"{metrics_valid_lenient['recall']:.4f}    {metrics_valid_lenient['f1']:.4f}")
        print("=" * 80)
        
        # Inference time statistics
        print(f"\n{' INFERENCE TIME STATISTICS ':-^40}")
        print(f"Average inference time: {sum(inference_times)/len(inference_times):.4f}s")
        print(f"Total inference time: {sum(inference_times):.4f}s")
        print(f"Maximum inference time: {max(inference_times):.4f}s")
        min_time = min([t for t in inference_times if t > 0], default=0)
        print(f"Minimum inference time: {min_time:.4f}s")
        
        # Error statistics
        print(f"\n{' ERROR STATISTICS ':-^40}")
        print("Strict parsing errors:")
        print(f"  Parsing errors: {strict_parse_error_count_log}/{total_samples} ({strict_parse_error_count_log/total_samples:.4f})")
        print(f"  Exceptions: {strict_exception_count}/{total_samples} ({strict_exception_count/total_samples:.4f})")
        print(f"  Total errors: {len(strict_error_log)}/{total_samples} ({len(strict_error_log)/total_samples:.4f})")
        
        print("Lenient parsing errors:")
        print(f"  Parsing errors: {lenient_parse_error_count_log}/{total_samples} ({lenient_parse_error_count_log/total_samples:.4f})")
        print(f"  Exceptions: {lenient_exception_count}/{total_samples} ({lenient_exception_count/total_samples:.4f})")
        print(f"  Total errors: {len(lenient_error_log)}/{total_samples} ({len(lenient_error_log)/total_samples:.4f})")
        
        print(f"Samples with differing predictions: {parse_difference_count}/{total_samples} ({parse_difference_count/total_samples:.4f})")
        
        # Save detailed report
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
        """Save a detailed text report of zero-shot testing results (strict vs lenient parsing)"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_dir = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(self.save_result_dir, date_dir)
        os.makedirs(save_dir, exist_ok=True)

        report_path = os.path.join(save_dir, f"detailed_report_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("        Detailed Performance Report: Strict vs. Lenient Parsing\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset statistics
            f.write("Dataset Statistics:\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Valid samples (Strict parsing): {metrics_valid_strict['sample_count']}\n")
            f.write(f"Valid samples (Lenient parsing): {metrics_valid_lenient['sample_count']}\n")
            f.write(f"Parsing errors (Strict): {len(strict_error_log)}\n")
            f.write(f"Parsing errors (Lenient): {len(lenient_error_log)}\n")
            f.write(f"Samples with differing predictions: {len(parse_differences)}\n\n")
            
            # Strict parsing metrics
            f.write("Strict Parsing Metrics:\n")
            f.write(f"Full dataset accuracy: {metrics_all_strict['accuracy']:.4f}\n")
            f.write(f"Full dataset precision: {metrics_all_strict['precision']:.4f}\n")
            f.write(f"Full dataset recall: {metrics_all_strict['recall']:.4f}\n")
            f.write(f"Full dataset F1-score: {metrics_all_strict['f1']:.4f}\n")
            f.write(f"Valid predictions accuracy: {metrics_valid_strict['accuracy']:.4f}\n")
            f.write(f"Valid predictions precision: {metrics_valid_strict['precision']:.4f}\n")
            f.write(f"Valid predictions recall: {metrics_valid_strict['recall']:.4f}\n")
            f.write(f"Valid predictions F1-score: {metrics_valid_strict['f1']:.4f}\n\n")
            
            # Lenient parsing metrics
            f.write("Lenient Parsing Metrics:\n")
            f.write(f"Full dataset accuracy: {metrics_all_lenient['accuracy']:.4f}\n")
            f.write(f"Full dataset precision: {metrics_all_lenient['precision']:.4f}\n")
            f.write(f"Full dataset recall: {metrics_all_lenient['recall']:.4f}\n")
            f.write(f"Full dataset F1-score: {metrics_all_lenient['f1']:.4f}\n")
            f.write(f"Valid predictions accuracy: {metrics_valid_lenient['accuracy']:.4f}\n")
            f.write(f"Valid predictions precision: {metrics_valid_lenient['precision']:.4f}\n")
            f.write(f"Valid predictions recall: {metrics_valid_lenient['recall']:.4f}\n")
            f.write(f"Valid predictions F1-score: {metrics_valid_lenient['f1']:.4f}\n\n")
            
            # Comparison between parsing methods
            f.write("Parsing Method Comparison:\n")
            f.write(f"Strict F1 > Lenient F1: {metrics_all_strict['f1'] > metrics_all_lenient['f1']}\n")
            f.write(f"F1 difference: {metrics_all_strict['f1'] - metrics_all_lenient['f1']:.4f}\n")
            f.write(f"Strict parsing error rate: {len(strict_error_log)/total_samples:.4f}\n")
            f.write(f"Lenient parsing error rate: {len(lenient_error_log)/total_samples:.4f}\n")
            f.write(f"Error rate difference: {(len(strict_error_log) - len(lenient_error_log))/total_samples:.4f}\n\n")
            
            # Inference time statistics
            f.write("Inference Time Statistics:\n")
            f.write(f"Average inference time: {sum(inference_times)/len(inference_times):.4f}s\n")
            f.write(f"Total inference time: {sum(inference_times):.4f}s\n")
            f.write(f"Maximum inference time: {max(inference_times):.4f}s\n")
            min_time = min([t for t in inference_times if t > 0], default=0)
            f.write(f"Minimum inference time: {min_time:.4f}s\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

        print(f"Detailed report saved at: {report_path}")



def batch_test_models():
    """Batch test all available models"""
    args = parse_for_llm()
    
    # Get the list of available models
    available_models = get_available_models(
        args.llms_dir, 
        specific_models=args.model_names,
        skip_models=args.skip_models
    )
    
    print(type(available_models), 'Available models list:')
    
    if not available_models:
        print("No available models found.")
        return
    
    print(f"Found {len(available_models)} available models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    
    # Store test results for all models
    all_results = {}
    
    # Test each model sequentially
    for model_name in available_models:
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Set paths for current model
            current_llm_dir = os.path.join(args.llms_dir, model_name)
            current_save_dir = os.path.join(args.save_result_dir, model_name)
            
            # Create tester instance
            tester = ZeroShotTester(
                test_dataset_dir=args.test_dataset_dir,
                save_result_dir=current_save_dir,
                llms_dir=current_llm_dir,
                device_map=args.device_map
            )
            
            # Run zero-shot test
            result = tester.zero_shot_test()
            if result:
                all_results[model_name] = result
                
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            continue
    
    # Save summary of batch testing results
    if all_results:
        save_batch_results(all_results, args.save_result_dir)
        print_batch_summary(all_results)


def save_batch_results(all_results, save_base_dir):
    """Save batch testing summary results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = datetime.datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(save_base_dir, date_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to JSON-serializable format
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
    """Print batch testing summary report"""
    print(f"\n{' Batch Testing Summary Report ':=^80}")
    print(f"{'Model Name':<30} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1 Score':<8}")
    print("-" * 80)
    
    # Sort results by F1 score descending
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} {metrics['accuracy']:.4f}    "
              f"{metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")
    
    print("=" * 80)
    
    # Print best model
    if sorted_results:
        best_model, best_metrics = sorted_results[0]
        print(f"\nBest Model: {best_model}")
        print(f"Best F1 Score: {best_metrics['f1']:.4f}")
        print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")



if __name__ == '__main__':
    batch_test_models()
   