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
from dataset_utils import cache_prompts, load_mmlu, load_winogrande, load_musr, load_mmlu_pro, load_hellaswag, generate_hash

from model_utils import get_model
from evaluation_utils import evaluate_model

console = Console()

def run_evaluation(model, tokenizer, datasets, progress, task):
    results = {}
    for dataset_name, dataset in datasets.items():
        sys.stdout.write(f"\rEvaluating on {dataset_name}...")
        sys.stdout.flush()
        
        # Determine the number of answer choices
        if dataset_name == "MMLU-Pro":
            answer_choices = "ABCDEFGHIJKLMNOP"  # 10 choices for MMLU-Pro
        elif dataset_name in ["WinograndeMuSR"]:
            answer_choices = "AB"  # 2 choices for Winogrande and MuSR
        else:
            answer_choices = "ABCD"  # 4 choices for standard MMLU and others
        
        answer_logits = [tokenizer.tokenizer.encode(f"{ch}")[-1] for ch in answer_choices]
        
        # If the dataset is nested (like MMLU or MMLU-Pro with categories)
        if isinstance(dataset, dict) and "prompts" not in dataset:
            category_results = {}
            for category, category_data in dataset.items():
                category_score, category_outputs = evaluate_model(model, tokenizer, category_data["prompts"], category_data["labels"], answer_logits)
                category_results[category] = {"score": float(category_score), "outputs": category_outputs}
                sys.stdout.write(f"\r-- {category} Score: {category_score:.4f}")
                sys.stdout.flush()
            results[dataset_name] = category_results
        else:
            # For datasets without categories (like Winogrande or MuSR)
            score, outputs = evaluate_model(model, tokenizer, dataset["prompts"], dataset["labels"], answer_logits)
            results[dataset_name] = {"score": float(score), "outputs": outputs}
            sys.stdout.write(f"\r-- Score: {score:.4f}")
            sys.stdout.flush()
        
        sys.stdout.write("\n")
        progress.update(task, advance=1)
    
    return results

def main():
    # MMLU
    mmlu_hash = generate_hash(MMLU_CATEGORIES, MMLU_QUESTIONS_PER_CATEGORY, SEED_KEY, PROMPT_FORMAT)
    mmlu_filename = os.path.join(CACHE_DIR, f"mmlu_prompts_{mmlu_hash}.json")
    mmlu_prompts = cache_prompts(mmlu_filename, load_mmlu, MMLU_CATEGORIES, MMLU_QUESTIONS_PER_CATEGORY, QA_SET, QA_SPLIT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    # Winogrande and MuSR 
    winogrande_hash = generate_hash(WINOGRANDE_QUESTIONS_COUNT, SEED_KEY, PROMPT_FORMAT)
    winogrande_filename = os.path.join(CACHE_DIR, f"winogrande_prompts_{winogrande_hash}.json")
    winogrande_prompts = cache_prompts(winogrande_filename, load_winogrande, WINOGRANDE_QUESTIONS_COUNT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    musr_hash = generate_hash(MUSR_QUESTIONS_COUNT, SEED_KEY, PROMPT_FORMAT)
    musr_filename = os.path.join(CACHE_DIR, f"musr_prompts_{musr_hash}.json")
    musr_prompts = cache_prompts(musr_filename, load_musr, MUSR_QUESTIONS_COUNT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    #  MMLU-Pro
    mmlu_pro_hash = generate_hash(MMLU_PRO_CATEGORIES, MMLU_PRO_QUESTIONS_PER_CATEGORY, SEED_KEY, PROMPT_FORMAT)
    mmlu_pro_filename = os.path.join(CACHE_DIR, f"mmlu_pro_prompts_{mmlu_pro_hash}.json")
    mmlu_pro_prompts = cache_prompts(mmlu_pro_filename, load_mmlu_pro, MMLU_PRO_CATEGORIES, MMLU_PRO_QUESTIONS_PER_CATEGORY, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    # HeLLaSWAG
    hellaswag_hash = generate_hash(HELLASWAG_QUESTIONS_COUNT, SEED_KEY, PROMPT_FORMAT)
    hellaswag_filename = os.path.join(CACHE_DIR, f"hellaswag_prompts_{hellaswag_hash}.json")
    hellaswag_prompts = cache_prompts(hellaswag_filename, load_hellaswag, HELLASWAG_QUESTIONS_COUNT, CACHE_DIR, SEED_KEY, PROMPT_FORMAT)

    # Combine all datasets
    datasets = {
        "HeLLaSWAG": hellaswag_prompts,
        "Winogrande": winogrande_prompts,
        "MuSR": musr_prompts,
        "MMLU": mmlu_prompts,
        "MMLU-Pro": mmlu_pro_prompts,
    }

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
            console.print(f"\r\n[bold green]Evaluating model: {variant}")
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
    
    # Helper function to flatten nested results
    def flatten_results(model_data):
        flattened = {}
        for dataset, data in model_data.items():
            if isinstance(data, dict) and "score" in data:
                flattened[dataset] = data["score"]
            elif isinstance(data, dict):
                for category, category_data in data.items():
                    flattened[f"{dataset}/{category}"] = category_data["score"]
        return flattened

    # Create DataFrame
    df = pd.DataFrame({model: flatten_results(model_data) for model, model_data in results.items()}).transpose()
    df = df.apply(pd.to_numeric, errors='coerce')

    console.print("\nDetailed Results:")

    # Function to create and print a table for a specific dataset
    def print_dataset_table(dataset_name, dataset_df):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="dim")
        for category in dataset_df.columns:
            table.add_column(category.split('/')[-1])  # Remove dataset prefix from column names
        
        for model, scores in dataset_df.iterrows():
            row = [model] + [f"{score:.4f}" if pd.notnull(score) else 'N/A' for score in scores]
            table.add_row(*row)
        
        console.print(f"\n{dataset_name} Results:")
        console.print(table)

    # MMLU table
    mmlu_columns = [col for col in df.columns if col.startswith("MMLU/")]
    if mmlu_columns:
        mmlu_df = df[mmlu_columns]
        print_dataset_table("MMLU", mmlu_df)

    # MMLU-Pro table
    mmlu_pro_columns = [col for col in df.columns if col.startswith("MMLU-Pro/")]
    if mmlu_pro_columns:
        mmlu_pro_df = df[mmlu_pro_columns]
        print_dataset_table("MMLU-Pro", mmlu_pro_df)

    # Other datasets (like Winogrande and MuSR)
    other_columns = [col for col in df.columns if '/' not in col]
    if other_columns:
        other_df = df[other_columns]
        print_dataset_table("Other Datasets", other_df)

    # Summary table
    summary_df = pd.DataFrame(index=df.index)
    
    if mmlu_columns:
        summary_df['MMLU (Avg)'] = df[mmlu_columns].mean(axis=1)
    if mmlu_pro_columns:
        summary_df['MMLU-Pro (Avg)'] = df[mmlu_pro_columns].mean(axis=1)
    
    for col in other_columns:
        summary_df[col] = df[col]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim")
    for column in summary_df.columns:
        table.add_column(column)
    
    for model, scores in summary_df.iterrows():
        row = [model] + [f"{score:.4f}" if pd.notnull(score) else 'N/A' for score in scores]
        table.add_row(*row)
    
    console.print("\nSummary Results:")
    console.print(table)

    # Save results to CSV
    results_path = os.path.join(RESULTS_DIR, f"results_model_comparison_{QA_SPLIT}_{current_datetime}.csv")
    df.to_csv(results_path)
    console.print(f"[green]Detailed results saved to {results_path}")

    # Plotting code
    if df.notna().any().any():
        plt.figure(figsize=(14, 8))
        ax = summary_df.plot(kind='bar')
        plt.title("Model Performance Summary", fontsize=16)
        plt.xlabel("Models", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(RESULTS_DIR, f"model_performance_summary_{current_datetime}.png"), dpi=300, bbox_inches='tight')
        console.print("[green]Performance summary plot saved as model_performance_summary.png")
    else:
        console.print("[yellow]Warning: No numeric data available for plotting.")
        

if __name__ == "__main__":
    main()