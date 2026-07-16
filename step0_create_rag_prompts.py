# The purpose of this script is to load the 4 baseline RAG prompts from the json files.

import json

NUM_OF_QUERIES_PER_DOMAIN = 1

INPUT_FILES = {
    "finance": "Dataset Source Files/finance_trunc_200.json"
    #"legal": "Dataset Source Files/legal_trunc_200.json",
    #"medical": "Dataset Source Files/medical_trunc_200.json",
    #"news": "Dataset Source Files/news_trunc_200.json"
}

OUTPUT_FILE = "Dataset Source Files/rag_prompts_baseline.json"

def main():
    combined_rows = []

    for domain, filename in INPUT_FILES.items():
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        for query in data[:NUM_OF_QUERIES_PER_DOMAIN]:
            combined_rows.append({
                "domain": domain,
                "query": query["query"],
                "doc1": query["doc1"],
                "doc2": query["doc2"]
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            combined_rows,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved {len(combined_rows)} RAG cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()