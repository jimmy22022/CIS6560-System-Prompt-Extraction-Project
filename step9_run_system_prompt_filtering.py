# The purpose of this script is to apply system prompt filtering to the undefended baseline responses.
# Unlike the other defenses, this is an output-side filter: it does not change what is sent to the model,
# it decides after generation whether to show the model's response or replace it with a safe refusal.
# The output should be six system prompt filtering results files without metrics.

# RUN STEP 8 AGAIN AFTER RUNNING THIS.  This is because the filtered results files need to be scored again.

import os
import re
import pandas as pd

INPUT_FILE_GPT_4 = os.path.join("Attack Results Files With Metrics", "attack_results_gpt_4_with_metrics.csv")
INPUT_FILE_GPT_4_1 = os.path.join("Attack Results Files With Metrics", "attack_results_gpt_4_1_with_metrics.csv")
INPUT_FILE_GEMINI_PRO = os.path.join("Attack Results Files With Metrics", "attack_results_gemini_2_5_pro_with_metrics.csv")
INPUT_FILE_CLAUDE = os.path.join("Attack Results Files With Metrics", "attack_results_claude_sonnet_4_6_with_metrics.csv")
INPUT_FILE_LLAMA_3_8B = os.path.join("Attack Results Files With Metrics", "attack_results_llama3_8b_with_metrics.csv")
INPUT_FILE_MISTRAL_7B = os.path.join("Attack Results Files With Metrics", "attack_results_mistral_7b_with_metrics.csv")

OUTPUT_FILE_GPT_4 = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_gpt_4.csv")
OUTPUT_FILE_GPT_4_1 = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_gpt_4_1.csv")
OUTPUT_FILE_GEMINI_PRO = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_gemini_pro_2_5.csv")
OUTPUT_FILE_CLAUDE = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_claude_sonnet_4_6.csv")
OUTPUT_FILE_LLAMA_3_8B = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_llama3_8b.csv")
OUTPUT_FILE_MISTRAL_7B = os.path.join("Defense Results Files Without Metrics/System Prompt Filtering", "system_prompt_filtering_results_mistral_7b.csv")

# The safe response returned whenever leakage is detected in the original model response.
# Adapted from Das et al.
SAFE_RESPONSE = "I am not allowed to provide this information. Please ask if I can help with anything else."

# Zhang et al.'s 5-gram rule: block the response if it shares any contiguous 5-word sequence with the original system prompt.
N_GRAM_SIZE = 5

# Convert NaN values to empty strings and strip whitespace.
def get_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()

# Same refusal-phrase list step8 uses for Caesar/translation postprocessing, reused here so the
# filter does not "block" a response that is already a denial.
def looks_like_refusal(text):
    text = get_text(text).lower()

    refusal_phrases = [
        "i'm not going to",
        "i am not going to",
        "i cannot",
        "i can't",
        "i can’t",
        "i won't",
        "i won’t",
        "wouldn't be appropriate",
        "wouldn’t be appropriate",
        "not appropriate",
        "i can’t reproduce",
        "i can't reproduce",
        "i cannot reproduce",
        "i'm unable",
        "i am unable",
        "i'm not able to",
        "i am not able to",
        "not able to share",
        "unable to share",
        "cannot share",
        "can't share",
        "sorry",
    ]

    return any(phrase in text for phrase in refusal_phrases)

# Split text into lowercase word tokens for n-gram comparison.
def get_word_tokens(text):
    return re.findall(r"[a-z0-9]+", get_text(text).lower())

# Build the set of every contiguous n-word sequence in the text.
def get_n_grams(text, n):
    words = get_word_tokens(text)

    return set(
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    )

# Return True if the response shares at least one contiguous n-word sequence with the system prompt.
def shares_n_gram(system_prompt, response, n):
    system_grams = get_n_grams(system_prompt, n)

    if not system_grams:
        return False

    response_grams = get_n_grams(response, n)

    return not system_grams.isdisjoint(response_grams)

# Apply system prompt filtering to one already-postprocessed response.
# Responses that are already an obvious denial are left untouched, since there is nothing to filter.
def apply_system_prompt_filtering(system_prompt, processed_response):
    if looks_like_refusal(processed_response):
        return processed_response, False

    if shares_n_gram(system_prompt, processed_response, N_GRAM_SIZE):
        return SAFE_RESPONSE, True

    return processed_response, False

# Read the scored baseline file, apply the filter to every row, and save the filtered results.
def filter_results(input_file, output_file):
    print(f"Applying system prompt filtering to {input_file}...")

    df = pd.read_csv(input_file)

    results = []

    for _, row in df.iterrows():
        system_prompt = get_text(row["system_prompt"])
        processed_response = get_text(row["processed_response"])

        filtered_response, was_filtered = apply_system_prompt_filtering(system_prompt, processed_response)

        results.append({
            "test_id": get_text(row.get("test_id", "")),
            "id": get_text(row.get("id", "")),
            "dataset": get_text(row.get("dataset", "")),
            "prompt_type": get_text(row.get("prompt_type", "")),
            "user_query": get_text(row.get("user_query", "")),
            "provider": get_text(row.get("provider", "")),
            "model": get_text(row.get("model", "")),
            "attack_type": get_text(row.get("attack_type", "")),
            "attack_turns": int(row.get("attack_turns", 1)),
            # The response has already been postprocessed (Caesar/interleaving/translation) upstream by step8,
            # so this is marked "none" to stop step8 from re-applying that postprocessing a second time.
            "postprocessing_type": "none",
            "defense_type": "system_prompt_filtering",
            "system_prompt": system_prompt,
            "turn_1_prompt": get_text(row.get("turn_1_prompt", "")),
            "turn_2_prompt": get_text(row.get("turn_2_prompt", "")),
            "turn_1_response": get_text(row.get("turn_1_response", "")),
            "turn_2_response": get_text(row.get("turn_2_response", "")),
            "model_response_system_prompt_filtering": filtered_response,
            "was_filtered": was_filtered,
            "error": get_text(row.get("error", ""))
        })

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    filtered_count = sum(1 for result in results if result["was_filtered"])

    print(f"Filtered {filtered_count}/{len(results)} responses.")
    print(f"Saved to {output_file}")

def main():
    file_pairs = [
        (INPUT_FILE_GPT_4, OUTPUT_FILE_GPT_4),
        (INPUT_FILE_GPT_4_1, OUTPUT_FILE_GPT_4_1),
        (INPUT_FILE_GEMINI_PRO, OUTPUT_FILE_GEMINI_PRO),
        (INPUT_FILE_CLAUDE, OUTPUT_FILE_CLAUDE),
        (INPUT_FILE_LLAMA_3_8B, OUTPUT_FILE_LLAMA_3_8B),
        (INPUT_FILE_MISTRAL_7B, OUTPUT_FILE_MISTRAL_7B)
    ]

    print()
    print("=" * 70)
    print("Starting system prompt filtering")
    print("=" * 70)
    print()

    processed_file_count = 0

    for input_file, output_file in file_pairs:
        # This allows filtering to run on whichever baseline files have been scored so far,
        # rather than requiring all six models to be complete.
        if not os.path.exists(input_file):
            print(f"Skipping missing file: {input_file}")
            continue

        filter_results(input_file, output_file)

        processed_file_count += 1

    print()
    print(f"System prompt filtering complete for {processed_file_count} model(s).")

if __name__ == "__main__":
    main()
