# dataset_utils.py

import os
import json
import random
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from urllib.request import urlretrieve
import zipfile

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
    print(f" -- Loading dataset: {ds_name}/{category}...")
    return load_dataset(ds_name, category, split=split, cache_dir=cache_dir).shuffle(seed=seed_key)

def format_mmlu_question(question, options, answer, selected_format):
    clabels = "ABCD"
    if selected_format == "none":
        text = f"Question:\n{question}\n\nChoices:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {o}\n"
        text += f"\nAnswer: "
    # Add other format options here if needed
    return text

def load_mmlu(categories, questions_per_category, qa_set, qa_split, cache_dir, seed_key, selected_format):
    prompts_dict = {}
    for category in tqdm(categories, desc="Loading datasets"):
        dataset = get_dataset(qa_set, category, qa_split, cache_dir, seed_key)
        rows = dataset.select(range(questions_per_category))

        prompts = [format_mmlu_question(row["question"], row["choices"], row["answer"], selected_format) for row in rows]
        labels = [row["answer"] for row in rows]

        prompts_dict[category] = {"prompts": prompts, "labels": labels}

    return prompts_dict

def format_winogrande_question(sentence, option1, option2, answer, selected_format):
    if selected_format == "none":
        text = "Fill in the blank (represented by _) in the following sentence with the best choice:\n"
        text += sentence
        text += "\n\nChoices:\n"
        text += f"A: {option1}\n"
        text += f"B: {option2}\n"
        text += "\nAnswer: "
    # Add other format options here if needed
    return text

def load_winogrande(questions_per_category, cache_dir, seed_key, selected_format):
    print(" -- Loading Winogrande dataset...")
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir, trust_remote_code=True).shuffle(seed=seed_key).select(range(questions_per_category))
    
    prompts = [format_winogrande_question(row["sentence"], row["option1"], row["option2"], row["answer"], selected_format) for row in dataset]
    labels = [int(row["answer"]) - 1 for row in dataset]

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
        text = f"{context}\n\nQuestion: This statement - '{question}' is:\n\nChoices:\nA: True\nB: False\n\nAnswer: "
    # Add other format options here if needed
    return text

def load_musr(questions_per_category, cache_dir, seed_key, selected_format):
    download_musr(cache_dir)
    print(" -- Loading MuSR dataset...")

    musr_dataset_path = os.path.join(cache_dir, "MuSR", "Multi-Step-Deductive-Reasoning-Over-Natural-Language-main", "dataset")
    files = [f"Depth{i}/PARARULE_Plus_Depth{i}_shuffled_test.jsonl" for i in range(2, 6)]

    combined_questions = []
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

    random.seed(seed_key)
    random.shuffle(combined_questions)

    prompts = []
    labels = []
    for q in combined_questions[:questions_per_category]:
        prompt = format_musr_question(q["context"], q["question"], q["answer"], selected_format)
        prompts.append(prompt)
        labels.append(0 if q["answer"].lower() == "true" else 1)  # 0 for True, 1 for False

    return {"prompts": prompts, "labels": labels}