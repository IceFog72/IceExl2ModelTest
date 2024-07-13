import os

# Configuration
MODEL_BASE = "E:\\models"
VARIANTS = ["IceSakeV12RP-7b-4.2bpw","Kunoichi-DPO-v2-7B-exl2-4_25"]
MODEL_PARAMS = {
    #"IceSakeV12RP-7b-4.2bpw": {"max_seq_len": 2048, "rotary_embedding_base": 40000.0},
    "IceSakeV12RP-7b-4.2bpw": {"max_seq_len": 2048, "rotary_embedding_base": 40000.0},
    "Kunoichi-DPO-v2-7B-exl2-4_25": {"max_seq_len": 2048, "rotary_embedding_base": 40000.0},
    #"dolphin-2.9.3-mistral-7B-32k-4.2bpw-exl2": {"max_seq_len": 2048, "rotary_embedding_base": 100000.0}
}

PROMPT_FORMAT = "none"
GPU_SPLIT = None  # auto
QA_SET = "cais/mmlu"
QA_SPLIT = "test"
MMLU_CATEGORIES = ["abstract_algebra","formal_logic","logical_fallacies", "philosophy"]
#MMLU_CATEGORIES = ["abstract_algebra","formal_logic", "high_school_biology", "high_school_mathematics", "high_school_us_history","human_sexuality","logical_fallacies", "philosophy"]

# MMLU_CATEGORIES = [
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


MMLU_PRO_CATEGORIES = ["Biology", "Business", "Chemistry", "ComputerScience", "Economics", "Engineering", "Health", "History", "Law", "Math", "Philosophy", "Physics", "Psychology", "Other"]



MMLU_PRO_QUESTIONS_PER_CATEGORY = 100  # max?
MMLU_QUESTIONS_PER_CATEGORY = 100 # 100 max
WINOGRANDE_QUESTIONS_COUNT = 100 # max?
MUSR_QUESTIONS_COUNT = 100 # 300 max
HELLASWAG_QUESTIONS_COUNT = 100  # Adjust as needed
SEED_KEY = 46756 # shuffle seed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache_dir")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_dir")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)