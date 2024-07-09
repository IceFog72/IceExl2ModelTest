import sys
import os
import gc
import json
import hashlib
import torch
from datasets import load_dataset
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm
import random
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(os.path.dirname(script_dir)))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)

# Models to test
model_base = "E:\\models"
variants = ["Storm-7B-4.2bpw","dolphin-2.9.3-mistral-7B-32k-4.2bpw-exl2","IceSakeV12RP-7b-4.2bpw"]
#variants = ["Mistral-7B-Instruct-v0.3-exl2-4.25", "M7-7b-4.0bpw-h6-exl2", "IceSakeV12RP-7b-4.2bpw"]

# Custom parameters for each model variant
model_params = {
    "Storm-7B-4.2bpw": {"temp": 1, "max_seq_len": 2048, "rotary_embedding_base": 40000.0},
    "dolphin-2.9.3-mistral-7B-32k-4.2bpw-exl2": {"temp": 1, "max_seq_len": 2048, "rotary_embedding_base": 100000.0},
    "IceSakeV12RP-7b-4.2bpw": {"temp": 1, "max_seq_len": 2048, "rotary_embedding_base": 40000.0}
    # Add other models with their specific parameters here
}

# Prompt format options
prompt_formats = ["none", "Alpaca", "ChatML", "Llama3"]
selected_format = "none"

gpu_split = None  # auto

qa_set = "cais/mmlu"
qa_split = "test"

mmlu_categories = ["abstract_algebra", "formal_logic", "logical_fallacies", "philosophy"]

# mmlu_categories = [
#     "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
#     "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
#     "college_medicine", "college_physics", "computer_security", "conceptual_physics", 
#     "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
#     "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
#     "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
#     "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
#     "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
#     "high_school_world_history", "human_aging", "human_sexuality", "international_law", 
#     "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
#     "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
#     "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
#     "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
# ]

mmlu_questions_per_category_count = 100
winogrande_questions_count = 100
musr_questions_count = 100

seed_key = 4675

cache_dir = os.path.join(script_dir, "cache_dir")  # Set your custom cache directory here

def get_model(base, variant_, gpu_split_, batch_size_):
    model_dir = os.path.join(base, variant_)

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_batch_size = batch_size_

    # Apply custom parameters
    params = model_params.get(variant_, {})
    for param, value in params.items():
        setattr(config, param, value)

    model_ = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    if gpu_split_:
        model_.load(gpu_split_)
        cache_ = None
    else:
        cache_ = ExLlamaV2Cache_Q4(model_, batch_size=batch_size_, lazy=True)
        model_.load_autosplit(cache_)

    tokenizer_ = ExLlamaV2Tokenizer(config)

    return model_, cache_, tokenizer_

def format_mmlu_question(question, options, answer, ex=False):
    clabels = "ABCD"
    if selected_format == "none":
        text = f"Question:\n{question}\n\nChoices:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {o}\n"
        text += f"\nAnswer: "
    elif selected_format == "Alpaca":
        text = f"### Instruction:\nQuestion:\n{question}\n\n### Input:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {o}\n"
        text += f"\n### Response: "
    elif selected_format == "ChatML":
        text = f"<|im_start|>User\n{question}\n\nOptions:\n"
        for i, o in enumerate(options):
            text += f"{clabels[i]}: {o}\n"
        text += f"\n<|im_end|><|im_start|>Assistant\n"
    elif selected_format == "Llama3":
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}\n\nOptions:\n"
        for i, o in enumerate(options):
            text += f"- {clabels[i]}: {o}\n"
        text += f"\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    return text

def format_winogrande_question(sentence, option1, option2, answer, ex=False):
    clabels = "AB"
    if selected_format == "none":
        text = "Fill in the blank (represented by _) in the following sentence with the best choice:\n"
        text += sentence
        text += "\n\nChoices:\n"
        text += f"A: {option1}\n"
        text += f"B: {option2}\n"
        text += "\nAnswer: "
    elif selected_format == "Alpaca":
        text = f"### Instruction:\nFill in the blank in the following sentence with the best choice:\n\n{sentence}\n\n### Input:\nA: {option1}\nB: {option2}\n\n### Response: "
    elif selected_format == "ChatML":
        text = f"<|im_start|>User\nFill in the blank (represented by _) in the following sentence with the best choice:\n\n{sentence}\n\nOptions:\nA: {option1}\nB: {option2}\n\n<|im_end|><|im_start|>Assistant\n"
    elif selected_format == "Llama3":
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nFill in the blank (represented by _) in the following sentence with the best choice:\n\n{sentence}\n\nOptions:\n- A: {option1}\n- B: {option2}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    return text

def format_musr_question(context, question, answer):
    answer_label = "A" if answer.lower() == "true" else "B"
    if selected_format == "none":
        text = f"{context}\n\nQuestion: This statement - '{question}' is:\n\nChoices:\nA: True\nB: False\n\nAnswer: "
    elif selected_format == "Alpaca":
        text = f"### Instruction:\n{context}\n\n### Input:\nQuestion:\nThis statement - '{question}' is:\n\nA: True\nB: False\n\n### Response: "
    elif selected_format == "ChatML":
        text = f"<|im_start|>User\n{context}\n\nQuestion: This statement - '{question}' is:\n\nOptions:\nA: True\nB: False\n\n<|im_end|><|im_start|>Assistant\n"
    elif selected_format == "Llama3":
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\nQuestion:\nThis statement - '{question}' is:\n\nOptions:\n- A: True\n- B: False\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    return text

def get_dataset(ds_name, category_, split_):
    print(f" -- Loading dataset: {ds_name}/{category_}...")
    return load_dataset(ds_name, category_, split=split_, cache_dir=cache_dir).shuffle(seed=seed_key)

def load_mmlu(categories, questions_per_category):
    prompts_dict = {}
    for category_ in tqdm(categories, desc="Loading datasets"):
        dataset = get_dataset(qa_set, category_, qa_split)
        rows = []
        for example in dataset:
            if len(rows) >= questions_per_category :
                break
            rows.append(example)

        prompts_ = [ format_mmlu_question(rows[j_ ]["question"], rows[j_ ]["choices"], rows[j_ ]["answer"]) for j_ in range(questions_per_category)]
        labels_ = [rows[j_ ]["answer"] for j_ in range(questions_per_category)]

        prompts_dict[category_] = {"prompts": prompts_, "labels": labels_}

    return prompts_dict

def load_winogrande(questions_per_category):
    print(" -- Loading Winogrande dataset...")
    
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir, trust_remote_code=True).shuffle(seed=seed_key).select(range(questions_per_category))
    rows = []
    for example in dataset:
        rows.append(example)
        

    prompts_ = [format_winogrande_question(rows[j_ ]["sentence"], rows[j_ ]["option1"], rows[j_]["option2"], rows[j_ ]["answer"]) for j_ in range(questions_per_category)]
    labels_ = [int(rows[j_ ]["answer"]) - 1 for j_ in range(questions_per_category)]

    return {"prompts": prompts_, "labels": labels_}

def download_musr():
    print(" -- Downloading MuSR dataset...")
    url = "https://github.com/Strong-AI-Lab/Multi-Step-Deductive-Reasoning-Over-Natural-Language/archive/refs/heads/main.zip"
    local_zip_path = os.path.join(cache_dir, "MuSR.zip")
    extract_path = os.path.join(cache_dir, "MuSR")

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        urlretrieve(url, local_zip_path)

        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)


def load_musr(questions_per_category):
    download_musr()
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

    # Use the seed key to shuffle
    random.seed(seed_key)
    random.shuffle(combined_questions)

    prompts_ = []
    labels_ = []
    for q in combined_questions[:questions_per_category]:
        prompt = format_musr_question(q["context"], q["question"], q["answer"])
        prompts_.append(prompt)
        labels_.append(0 if q["answer"].lower() == "true" else 1)  # 0 for True, 1 for False

    return {"prompts": prompts_, "labels": labels_}

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

def evaluate_model(model, tokenizer, prompts, labels, answer_logits):
    score = 0.0
    model_outputs = []
    for prompt, label in tqdm(zip(prompts, labels), total=len(prompts), desc="Evaluating model"):
        prompt_ids = tokenizer.encode(prompt)[:, :-1]
        logits = model.forward(prompt_ids, last_id_only=True).float()
        logits_ans = logits[:, :, answer_logits]
        prob_ans = torch.softmax(logits_ans, dim=-1)
        predicted_label = torch.argmax(prob_ans).item()
        score += prob_ans[0, 0, label]
        
        # Decode the predicted answer
        predicted_answer = chr(ord('A') + predicted_label)
        
        model_outputs.append({
            'prompt': prompt,
            'correct_label': chr(ord('A') + label),
            'predicted_label': predicted_answer,
            'is_correct': predicted_label == label
        })
    
    return score / len(prompts), model_outputs

mmlu_hash = generate_hash(mmlu_categories, mmlu_questions_per_category_count, seed_key,selected_format)
mmlu_filename = os.path.join(cache_dir, f"mmlu_prompts_{mmlu_hash}.json")
mmlu_prompts = cache_prompts(mmlu_filename, load_mmlu, mmlu_categories, mmlu_questions_per_category_count)

winogrande_hash = generate_hash("winogrande_xl", winogrande_questions_count, seed_key,selected_format)
winogrande_filename = os.path.join(cache_dir, f"winogrande_prompts_{winogrande_hash}.json")
winogrande_prompts = cache_prompts(winogrande_filename, load_winogrande, winogrande_questions_count)

musr_hash = generate_hash(musr_questions_count, seed_key,selected_format)
musr_filename = os.path.join(cache_dir, f"musr_prompts_{musr_hash}.json")
musr_prompts = cache_prompts(musr_filename, load_musr, musr_questions_count)

results = ";".join([""] + mmlu_categories + ["Winogrande", "MuSR"]) + "\n"

for variant in variants:

    model = None
    cache = None
    tokenizer = None
    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model, cache, tokenizer = get_model(model_base, variant, gpu_split, 1)
    cat_results = []
    all_outputs = {}

    answer_logits = [tokenizer.tokenizer.encode(f"{ch}")[-1] for ch in "ABCD"]
    for category in mmlu_categories:
        print(f" -- Testing MMLU: {category}...")
        score, outputs = evaluate_model(model, tokenizer, mmlu_prompts[category]["prompts"], mmlu_prompts[category]["labels"], answer_logits)
        print(f" -- Score: {score:.4f}")
        cat_results.append(f"{score:.4f}")
        all_outputs[f"MMLU_{category}"] = outputs

    print(" -- Testing: Winogrande...")
    answer_logits = [tokenizer.tokenizer.encode(f"{ch}")[-1] for ch in "AB"]
    score, outputs = evaluate_model(model, tokenizer, winogrande_prompts["prompts"], winogrande_prompts["labels"], answer_logits)
    print(f" -- Score: {score:.4f}")
    cat_results.append(f"{score:.4f}")
    all_outputs["Winogrande"] = outputs

    print(" -- Testing: MuSR...")
    score, outputs = evaluate_model(model, tokenizer, musr_prompts["prompts"], musr_prompts["labels"], [tokenizer.tokenizer.encode(f"{ch}")[-1] for ch in "AB"])
    print(f" -- Score: {score:.4f}")
    cat_results.append(f"{score:.4f}")
    all_outputs["MuSR"] = outputs

    results += ";".join([variant] + cat_results) + "\n"
    print(results)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

results_dir_path = os.path.join(script_dir, "results_dir")
if not os.path.exists(os.path.join(script_dir, "results_dir")):
    os.makedirs(results_dir_path, exist_ok=True)

output_file = os.path.join(results_dir_path, f"detailed_outputs_{variant}_{qa_split}_{current_datetime}.json")
with open(output_file, "w") as f:
    json.dump(all_outputs, f, indent=4)

results_path = os.path.join(results_dir_path, f"results_model_comparison_{qa_split}_{current_datetime}.csv")
with open(results_path, "w") as f:
    f.write(results)
