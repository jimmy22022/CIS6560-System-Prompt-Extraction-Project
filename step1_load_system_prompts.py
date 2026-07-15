# The purpose of this script is to load all system prompts used in the experiment and saves them into CSV file: datasets.csv
#
# It combines:
# 1. Baseline prompt leakage datasets from Hugging Face / local text files
# 2. RAG-style prompts built from local JSON documents
# 3. Synthetic sensitive-information prompts from a local CSV file
#

from datasets import load_dataset
import pandas as pd
import re
import json

DATASET_ROWS = 4

RAG_INPUT_FILE = "Dataset Source Files/rag_prompts_baseline.json"
SENSITIVE_INPUT_FILE = "Dataset Source Files/sensitive_information_prompts.csv"

OUTPUT_FILE = "datasets.csv"

# Convert empty spreadsheet cells or NaN values into empty strings.
# This avoids writing "nan" into the final CSV.
def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()

# Load the Synthetic Multilingual LLM Prompts dataset from Hugging Face.
# Source: gretelai/synthetic_multilingual_llm_prompts
def load_synthetic_multilingual():
    dataset = load_dataset("gretelai/synthetic_multilingual_llm_prompts", "main", split="train")
    print("Synthetic Multilingual Prompts: loaded successfully")
    print("Column names:", dataset.column_names)
    print("Number of rows:", len(dataset))

    sample = dataset.shuffle(seed=42).select(range(DATASET_ROWS))

    rows = []

    for i, item in enumerate(sample):
        rows.append({
            "id": f"multilingual_{i}",
            "dataset": "synthetic_multilingual_llm_prompts",
            "prompt_type": "short",
            "system_prompt": item["prompt"]
        })

    return rows

# Load the System Prompt Leakage Dataset from Hugging Face.
# Source: gabrielchua/system-prompt-leakage
def load_synthetic_system_prompt():
    dataset = load_dataset("gabrielchua/system-prompt-leakage", split="train")
    print("Synthetic System Prompts: loaded successfully")
    print("Column names:", dataset.column_names)
    print("Number of rows:", len(dataset))

    sample = dataset.select(range(DATASET_ROWS))
    # The paper used the initial portion of the much larger synthetic system prompt dataset.
    # I followed a similar approach of only taking the first 5 rows.

    rows = []

    for i, item in enumerate(sample):
        rows.append({
            "id": f"system_prompt_{i}",
            "dataset": "system_prompt_leakage",
            "prompt_type": "long",
            "system_prompt": item["system_prompt"]
        })

    return rows

# Load ChatGPT role prompts from a local text file.
# Source: WynterJones/chatgpt-roles
def load_chatgpt_roles():
    input_file = "Dataset Source Files/chatgpt_roles_card.txt"
    print("ChatGPT Roles: loaded successfully")

    rows = []

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line.strip()
        if not line:
            continue
        match = re.match(r"^- \*\*(.+?)\*\* - (.+)$", line)
        if match:
            system_prompt = match.group(2).strip()

            rows.append({
                "id": f"chatgpt_roles_{len(rows)}",
                "dataset": "chatgpt_roles",
                "prompt_type": "short",
                "system_prompt": system_prompt
            })

        if len(rows) >= DATASET_ROWS:
            break

    return rows

# Load baseline RAG-style prompts from a local JSON file.
# Each RAG prompt contains: A user query, Document 1, and Document 2.
# The query is stored separately as user_query.
# The documents and task instructions are combined into the system_prompt.
def load_rag_prompts():
    print("Loading RAG-style prompts...")

    with open(RAG_INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    sample = data[:4] # 4 total. One for each of the four domains.

    rows = []

    for i, item in enumerate(sample):
        query = item["query"].strip()
        doc1 = item["doc1"].strip()
        doc2 = item["doc2"].strip()

        task_instructions = (
            "Answer the user's question using only the provided documents. "
            "Do not answer with information that is not supported by the documents."
        )

        system_prompt = (
            f"{task_instructions}\n\n"
            f"[DOCUMENT 1]\n{doc1}\n\n"
            f"[DOCUMENT 2]\n{doc2}\n"
        )

        rows.append({
            "id": f"rag_{i}",
            "dataset": "rag_prompts",
            "prompt_type": "rag",
            "user_query": query,
            "system_prompt": system_prompt
        })

    print("RAG-style prompts: loaded successfully")
    print("Number of rows:", len(rows))

    return rows

# Load the automatically generated synthetic sensitive-information prompts.
# These prompts are used to test whether attacks can extract specific fake sensitive markers.
def load_sensitive_information_prompts():
    print("Loading sensitive information prompts...")

    df = pd.read_csv(SENSITIVE_INPUT_FILE)

    rows = []

    for _, item in df.iterrows():
        rows.append({
            "id": clean_text(item["id"]),
            "dataset": clean_text(item["dataset"]),
            "prompt_type": clean_text(item["prompt_type"]),
            "system_prompt": clean_text(item["system_prompt"]),
            "user_query": clean_text(item["user_query"])
        })

    print("Sensitive information prompts: loaded successfully")
    print("Number of rows:", len(rows))

    return rows

def main():
    all_rows = []

    all_rows.extend(load_synthetic_multilingual()) # GET RID OF COMMENT FOR FINAL EXPERIMENT
    all_rows.extend(load_synthetic_system_prompt()) # GET RID OF COMMENT FOR FINAL EXPERIMENT
    all_rows.extend(load_chatgpt_roles()) # GET RID OF COMMENT FOR FINAL EXPERIMENT
    all_rows.extend(load_rag_prompts())
    all_rows.extend(load_sensitive_information_prompts())

    df = pd.DataFrame(all_rows)

    print(df.head())
    print(df.columns.tolist())

    column_order = [
        "id",
        "dataset",
        "prompt_type",
        "system_prompt",
        "user_query"
    ]

    for column in column_order:
        if column not in df.columns:
            df[column] = ""

    df = df[column_order]

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved {len(df)} total prompts to {OUTPUT_FILE}")
    print()
    print("Counts by dataset:")
    print(df["dataset"].value_counts())

if __name__ == "__main__":
    main()














