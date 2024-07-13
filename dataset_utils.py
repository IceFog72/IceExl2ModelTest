# dataset_utils.py

import os
import json
import random
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from urllib.request import urlretrieve
from collections import defaultdict
import zipfile
import re
import unicodedata

def generate_hash(*args):
    hash_obj = hashlib.sha256()
    for arg in args:
        hash_obj.update(str(arg).encode())
    return hash_obj.hexdigest()[:12]

def cache_prompts(filename, loader_func, *args):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        prompts = loader_func(*args)
        with open(filename, "w") as f:
            json.dump(prompts, f, indent=4)
        return prompts

def get_dataset(ds_name, category, split, cache_dir, seed_key):
    print(f"\r\n -- Loading dataset: {ds_name}/{category}...")
    return load_dataset(ds_name, category, split=split, cache_dir=cache_dir).shuffle(seed=seed_key)

def format_mmlu_question(question, options, answer, selected_format):
    clabels = "ABCD"
    if selected_format == "none":
        text = f"Question:\n{clean_unicode(question)}\n\nOptions:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {clean_unicode(o)}\n"
        text += f"\nAnswer is: "
    # Add other format options here if needed
    return text

def load_mmlu(categories, questions_per_category, qa_set, qa_split, cache_dir, seed_key, selected_format):
    prompts_dict = {}
    with tqdm(total=len(categories), desc="Loading MMLU datasets") as pbar:
        for category in categories:
            dataset = get_dataset(qa_set, category, qa_split, cache_dir, seed_key)
            rows = dataset.select(range(questions_per_category))

            prompts = [format_mmlu_question(row["question"], row["choices"], row["answer"], selected_format) for row in rows]
            labels = [row["answer"] for row in rows]

            prompts_dict[category] = {"prompts": prompts, "labels": labels}
            pbar.update(1)
            pbar.set_postfix({"Current category": category})

    return prompts_dict

def format_winogrande_question(sentence, option1, option2, answer, selected_format):
    if selected_format == "none":
        text = "Fill in the blank in the following sentence with correct choice:\n"
        text += clean_unicode(sentence)
        text += "\n\nChoices:\n"
        text += f"A: {option1}\n"
        text += f"B: {option2}\n"
        text += "\nAnswer is:\n"
    # Add other format options here if needed
    return text

def load_winogrande(questions_per_category, cache_dir, seed_key, selected_format):
    print("Loading Winogrande dataset...")
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir, trust_remote_code=True).shuffle(seed=seed_key).select(range(questions_per_category))
    
    prompts = []
    labels = []
    with tqdm(total=len(dataset), desc="Processing Winogrande questions") as pbar:
        for row in dataset:
            prompts.append(format_winogrande_question(row["sentence"], row["option1"], row["option2"], row["answer"], selected_format))
            labels.append(int(row["answer"]) - 1)
            pbar.update(1)

    return {"prompts": prompts, "labels": labels}

def download_musr(cache_dir):
    print(" -- Downloading MuSR dataset...")
    url = "https://github.com/Strong-AI-Lab/Multi-Step-Deductive-Reasoning-Over-Natural-Language/archive/refs/heads/main.zip"
    local_zip_path = os.path.join(cache_dir, "MuSR.zip")
    extract_path = os.path.join(cache_dir, "MuSR")

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        urlretrieve(url, local_zip_path)

        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

def format_musr_question(context, question, answer, selected_format):
    if selected_format == "none":
        text = f"{clean_unicode(context)}\n\nQuestion: This statement - '{clean_unicode(question)}' is:\n\nChoices:\nA: True\nB: False\n\nThink step-by-step and answer only with the correct letter:\n"
    # Add other format options here if needed
    return text

def load_musr(questions_per_category, cache_dir, seed_key, selected_format):
    download_musr(cache_dir)
    print("Loading MuSR dataset...")

    musr_dataset_path = os.path.join(cache_dir, "MuSR", "Multi-Step-Deductive-Reasoning-Over-Natural-Language-main", "dataset")
    files = [f"Depth{i}/PARARULE_Plus_Depth{i}_shuffled_test.jsonl" for i in range(2, 6)]

    combined_questions = []
    with tqdm(total=len(files), desc="Reading MuSR files") as pbar:
        for file in files:
            with open(os.path.join(musr_dataset_path, file), 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    for q in entry["questions"]:
                        combined_questions.append({
                            "context": entry["context"],
                            "question": q["text"],
                            "answer": q["label"]
                        })
            pbar.update(1)
            pbar.set_postfix({"Current file": file})

    random.seed(seed_key)
    random.shuffle(combined_questions)

    prompts = []
    labels = []
    with tqdm(total=min(len(combined_questions), questions_per_category), desc="Processing MuSR questions") as pbar:
        for q in combined_questions[:questions_per_category]:
            prompt = format_musr_question(q["context"], q["question"], q["answer"], selected_format)
            prompts.append(prompt)
            labels.append(0 if q["answer"].lower() == "true" else 1)  # 0 for True, 1 for False
            pbar.update(1)

    return {"prompts": prompts, "labels": labels}

def clean_unicode(text):
    # Step 1: Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Replace specific problematic characters
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
    }
    for code, char in replacements.items():
        text = text.replace(code, char)
    
    # Step 3: Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

def load_mmlu_pro(categories, questions_per_category, cache_dir, seed_key, selected_format):
    prompts_dict = defaultdict(lambda: {"prompts": [], "labels": []})
    
    # Load the entire dataset
    dataset = get_dataset("TIGER-Lab/MMLU-Pro", "default", "test", cache_dir, seed_key)
    dataset = dataset.shuffle(seed=seed_key)  # Shuffle the dataset
    
    # Count questions per category
    category_counts = defaultdict(int)
    
    with tqdm(total=len(dataset), desc="Loading MMLU-Pro datasets") as pbar:
        for row in dataset:
            category = row["category"]
            
            # Skip if we've already collected enough questions for this category
            if category_counts[category] >= questions_per_category:
                pbar.update(1)
                continue
            
            prompt = format_mmlu_pro_question(row["question"], row["options"], row["answer_index"], selected_format)
            label = row["answer_index"]  # Use answer_index instead of answer
            
            prompts_dict[category]["prompts"].append(prompt)
            prompts_dict[category]["labels"].append(label)
            
            category_counts[category] += 1
            pbar.update(1)
            pbar.set_postfix({"Current category": category, "Count": category_counts[category]})
            
            # Break if we've collected enough questions for all categories
            if all(count >= questions_per_category for count in category_counts.values()):
                break
    
    # Remove any categories that didn't meet the required question count
    prompts_dict = {k: v for k, v in prompts_dict.items() if len(v["prompts"]) == questions_per_category}
    
    print(f"Loaded categories: {', '.join(prompts_dict.keys())}")
    print(f"Questions per category: {questions_per_category}")
    
    return dict(prompts_dict)

def format_mmlu_pro_question(question, options, answer, selected_format):
    clabels = "ABCDEFGHIJKLMNOP"
    if selected_format == "none":
        text = f"Question:\n{clean_unicode(question)}\nThink step-by-step and answer only with the correct letter.\nChoices:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {clean_unicode(o)}\n"
        text += f"\nAnswer is:\n"
    # Add other format options here if needed
    return text

def format_hellaswag_question(ctx, endings, selected_format):
    if selected_format == "none":
        text = f"Context: {clean_unicode(ctx)}\n\nComplete the sentence with the most appropriate ending:\n\n"
        for i, ending in enumerate(endings):
            text += f"{chr(65+i)}: {clean_unicode(ending)}\n"
        text += "\nAnswer is: "
    # Add other format options here if needed
    return text

def load_hellaswag(questions_per_category, cache_dir, seed_key, selected_format):
    print("Loading HeLLaSWAG dataset...")
    dataset = load_dataset("hellaswag", split="validation", cache_dir=cache_dir).shuffle(seed=seed_key).select(range(questions_per_category))
    
    prompts = []
    labels = []
    with tqdm(total=len(dataset), desc="Processing HeLLaSWAG questions") as pbar:
        for row in dataset:
            prompts.append(format_hellaswag_question(row["ctx"], row["endings"], selected_format))
            labels.append(int(row["label"]))  # Convert label to integer
            pbar.update(1)

    return {"prompts": prompts, "labels": labels}