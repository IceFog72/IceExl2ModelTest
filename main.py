import os
import gc
import time
import json
import torch
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import pandas as pd
import matplotlib.pyplot as plt
import sys

from config import *
from model_utils import get_model
from dataset_utils import cache_prompts, load_mmlu, load_winogrande, load_musr, generate_hash
from evaluation_utils import evaluate_model

console = Console()

def run_evaluation(model, tokenizer, datasets, progress, task):
    results = {}
    for dataset_name, dataset in datasets.items():
        sys.stdout.write(f"\rEvaluating on {dataset_name}...")
        sys.stdout.flush()
        answer_logits = [tokenizer.tokenizer.encode(f"The answer is: {ch}")[-1] for ch in "ABCD"]
        if dataset_name in ["Winogrande", "MuSR"]:
            answer_logits = answer_logits[:2]  # Only A and B for Winogrande and MuSR
        score, outputs = evaluate_model(model, tokenizer, dataset["prompts"], dataset["labels"], answer_logits)
        sys.stdout.write(f"\r-- Score: {score:.4f}\n")
        sys.stdout.flush()
        results[dataset_name] = {"score": float(score), "outputs": outputs}  # Ensure score is a float
        progress.update(task, advance=1)
    return results

def main():
    mmlu_hash = generate_hash(MMLU_CATEGORIES, MMLU_QUESTIONS_PER_CATEGORY, SEED_KEY, PROMPT_FORMAT)
    mmlu_filename = os.path.join(CACHE_DIR, f"mmlu_prompts_{mmlu_hash}.json")
    mmlu_prompts = cache_prompts(mmlu_filename, load_mmlu, MMLU_CATEGORIES, MMLU_QUESTIONS_PER_CATEGORY, QA_SET, QA_SPLIT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    winogrande_hash = generate_hash("winogrande_xl", WINOGRANDE_QUESTIONS_COUNT, SEED_KEY, PROMPT_FORMAT)
    winogrande_filename = os.path.join(CACHE_DIR, f"winogrande_prompts_{winogrande_hash}.json")
    winogrande_prompts = cache_prompts(winogrande_filename, load_winogrande, WINOGRANDE_QUESTIONS_COUNT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    musr_hash = generate_hash(MUSR_QUESTIONS_COUNT, SEED_KEY, PROMPT_FORMAT)
    musr_filename = os.path.join(CACHE_DIR, f"musr_prompts_{musr_hash}.json")
    musr_prompts = cache_prompts(musr_filename, load_musr, MUSR_QUESTIONS_COUNT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    datasets = {**mmlu_prompts, "Winogrande": winogrande_prompts, "MuSR": musr_prompts}

    all_results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task("[cyan]Overall Progress", total=len(VARIANTS) * len(datasets))
        for variant in VARIANTS:
            console.print(f"\n[bold green]Evaluating model: {variant}")
            model = None
            cache = None
            tokenizer = None
            
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
            model, cache, tokenizer = get_model(MODEL_BASE, variant, GPU_SPLIT, 1, MODEL_PARAMS)
            
            results = run_evaluation(model, tokenizer, datasets, progress, overall_task)
            all_results[variant] = results


    analyze_results(all_results)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"detailed_outputs_{QA_SPLIT}_{current_datetime}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    console.print(f"[green]Detailed results saved to {output_file}")

def analyze_results(results):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Convert the nested dictionary to a DataFrame with numeric values
    df = pd.DataFrame({model: {dataset: data["score"] for dataset, data in model_data.items()} 
                       for model, model_data in results.items()}).transpose()
    
    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    console.print("\nOverall Results:")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim")
    for dataset in df.columns:
        table.add_column(dataset)
    
    for model, scores in df.iterrows():
        row = [model] + [f"{score:.4f}" if pd.notnull(score) else 'N/A' for score in scores]
        table.add_row(*row)
    
    console.print(table)
    
    # Check if there's any numeric data to plot
    if df.notna().any().any():
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar')
        plt.title("Model Performance Across Datasets")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.legend(title="Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"model_performance_{current_datetime}.png"))
        console.print("[green]Performance plot saved as model_performance.png")
    else:
        console.print("[yellow]Warning: No numeric data available for plotting.")


if __name__ == "__main__":
    main()