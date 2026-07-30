# The purpose of this script is to run attack_cases.csv against the named models but with XML Tagging applied.
# The output should be six XML Tagging results files without metrics.

import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic
import requests

MODEL_NAME_GPT_4 = "gpt-4"
MODEL_NAME_GPT_4_1 = "gpt-4.1"
MODEL_NAME_GEMINI_PRO = "gemini-2.5-pro"
MODEL_NAME_CLAUDE = "claude-sonnet-4-6"
MODEL_NAME_LLAMA_3_8B = "llama3:8b"
MODEL_NAME_MISTRAL_7B = "mistral:latest"

INPUT_FILE = "attack_cases.csv"

OUTPUT_FILE_GPT_4 = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_gpt_4.csv")
OUTPUT_FILE_GPT_4_1 = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_gpt_4_1.csv")
OUTPUT_FILE_GEMINI_PRO = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_gemini_pro_2_5.csv")
OUTPUT_FILE_CLAUDE = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_claude_sonnet_4_6.csv")
OUTPUT_FILE_LLAMA_3_8B = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_llama3_8b.csv")
OUTPUT_FILE_MISTRAL_7B = os.path.join("Defense Results Files Without Metrics/XML Tagging", "xml_tagging_defense_results_mistral_7b.csv")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 1
MAX_OUTPUT_TOKENS = 5000

# Convert NaN values to empty strings and strip whitespace.
def get_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()

# In XML tagging, different sections of the system prompt are surrounded using XML tags, creating boundary awareness for the LLM.
def xml_tagging(text, tag_name):
    text = get_text(text)

    return (
        f"===== {tag_name} =====\n"
        f"{text}\n"
        f"===== /{tag_name} ====="
    )

# Convert the attack_turns column into an integer.
# Single-turn attacks = 1 and Multi-turn attacks = 2
def get_attack_turns(value):
    return int(value)

# Claude responses may contain multiple content blocks.
    # This helper extracts only the text blocks and joins them together.
def extract_claude_text(response):
    output = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            output.append(block.text)

    return "".join(output)

# Experiment behavior:
#
# For every attack case,
# 1. Send turn_1_prompt to the model, which is the attack prompt.
# 2. If attack_turns = 1, save the first response as the final model response.
# 3. If attack_turns = 2, send turn_2_prompt and save the second response as the final model response.
#
# The final response is stored in model_response_xml_tagging_defense.

# Send a single-turn attack to an OpenAI model.
# The hidden system prompt is passed through the instructions field.
def call_openai_single_turn(client, model_name, system_prompt, turn_1_prompt):
    response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=turn_1_prompt
    )

    return response.output_text

# Send a multi-turn attack to an OpenAI model. turn_1_prompt and turn_2_prompt are sent in the same conversation context.
def call_openai_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt):
    first_response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=turn_1_prompt
    )

    turn_1_response = first_response.output_text

    second_response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": turn_1_prompt
            },
            {
                "role": "assistant",
                "content": turn_1_response
            },
            {
                "role": "user",
                "content": turn_2_prompt
            }
        ]
    )

    turn_2_response = second_response.output_text

    return turn_1_response, turn_2_response

# Send a single-turn attack to a Gemini model.
# Gemini uses system_instruction inside the generation config.
def call_gemini_single_turn(client, model_name, system_prompt, turn_1_prompt):
    response = client.models.generate_content(
        model=model_name,
        contents=turn_1_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )

    return response.text or ""

# Send a multi-turn attack to Gemini. turn_1_prompt and turn_2_prompt are sent in the same conversation context.
def call_gemini_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt):
    first_response = client.models.generate_content(
        model=model_name,
        contents=turn_1_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )

    turn_1_response = first_response.text or ""

    second_response = client.models.generate_content(
        model=model_name,
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": turn_1_prompt
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": turn_1_response
                    }
                ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": turn_2_prompt
                    }
                ]
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )

    turn_2_response = second_response.text or ""

    return turn_1_response, turn_2_response

# Send a single-turn attack to Claude.
# Claude requires max_tokens to be provided.
def call_claude_single_turn(client, model_name, system_prompt, turn_1_prompt):
    response = client.messages.create(
        model=model_name,
        max_tokens = MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": turn_1_prompt
            }
        ]
    )

    return extract_claude_text(response)

# Send a multi-turn attack to Claude. turn_1_prompt and turn_2_prompt are sent in the same conversation context.
def call_claude_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt):
    first_response = client.messages.create(
        model=model_name,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": turn_1_prompt
            }
        ]
    )

    turn_1_response = extract_claude_text(first_response)

    second_response = client.messages.create(
        model=model_name,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": turn_1_prompt
            },
            {
                "role": "assistant",
                "content": turn_1_response
            },
            {
                "role": "user",
                "content": turn_2_prompt
            }
        ]
    )

    turn_2_response = extract_claude_text(second_response)

    return turn_1_response, turn_2_response

# Send a single-turn attack to a local Ollama model.
def call_ollama_single_turn(model_name, system_prompt, turn_1_prompt):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": turn_1_prompt
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 1200,
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]

# Send a multi-turn attack to a local Ollama model. turn_1_prompt and turn_2_prompt are sent in the same conversation context.
def call_ollama_multi_turn(model_name, system_prompt, turn_1_prompt, turn_2_prompt):
    first_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": turn_1_prompt
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 1200,
            "temperature": 0
        }
    }

    first_response = requests.post(
        OLLAMA_URL,
        json=first_payload,
        timeout=600
    )

    first_response.raise_for_status()

    turn_1_response = first_response.json()["message"]["content"]

    second_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": turn_1_prompt
            },
            {
                "role": "assistant",
                "content": turn_1_response
            },
            {
                "role": "user",
                "content": turn_2_prompt
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 1200,
            "temperature": 0
        }
    }

    second_response = requests.post(
        OLLAMA_URL,
        json=second_payload,
        timeout=600
    )

    second_response.raise_for_status()

    turn_2_response = second_response.json()["message"]["content"]

    return turn_1_response, turn_2_response

# Provider routing function for single-turn attacks.
# This is to make calling each of the single-turn attack functions easier.
def call_model_single_turn(provider, client, model_name, system_prompt, turn_1_prompt):
    if provider == "openai":
        return call_openai_single_turn(client, model_name, system_prompt, turn_1_prompt)
    if provider == "gemini":
        return call_gemini_single_turn(client, model_name, system_prompt, turn_1_prompt)
    if provider == "claude":
        return call_claude_single_turn(client, model_name, system_prompt, turn_1_prompt)
    if provider == "ollama":
        return call_ollama_single_turn(model_name, system_prompt, turn_1_prompt)

    raise ValueError(f"Unsupported Provider: {provider}")

# Provider routing function for multi-turn attacks.
# This is to make calling each of the multi-turn attack functions easier.
def call_model_multi_turn(provider, client, model_name, system_prompt, turn_1_prompt, turn_2_prompt):
    if provider == "openai":
        return call_openai_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt)
    if provider == "gemini":
        return call_gemini_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt)
    if provider == "claude":
        return call_claude_multi_turn(client, model_name, system_prompt, turn_1_prompt, turn_2_prompt)
    if provider == "ollama":
        return call_ollama_multi_turn(model_name, system_prompt, turn_1_prompt, turn_2_prompt)

    raise ValueError(f"Unsupported Provider: {provider}")

# Run every attack case against one model and save the results.
    #
    # Results are saved after every row so partial progress is preserved
    # if the script stops or an API/local model fails.
def run_experiment(provider, client, attack_df, model_name, output_file):
    results = []

    print()
    print("=" * 70)
    print(f"Starting XML tagging defense experiment with model: {model_name}")
    print(f"Loaded {len(attack_df)} attack cases")
    print("=" * 70)
    print()

    for position, (_, row) in enumerate(attack_df.iterrows(), start=1):
        prompt_id = get_text(row["id"])
        attack_type = get_text(row["attack_type"])
        test_id = f"{prompt_id}_{attack_type}"

        print(
            f"[{model_name}] Running "
            f"{position}/{len(attack_df)}: {test_id}"
        )

        system_prompt = get_text(row["system_prompt"])
        attack_type = get_text(row["attack_type"])
        user_query = get_text(row.get("user_query", ""))
        turn_1_prompt = get_text(row["turn_1_prompt"])
        turn_2_prompt = get_text(row.get("turn_2_prompt", ""))
        attack_turns = get_attack_turns(row.get("attack_turns", 1))

        system_prompt_with_xml_tagging_defense = xml_tagging(system_prompt, "SYSTEM INSTRUCTIONS")
        turn_1_prompt_with_xml_tagging_defense = xml_tagging(turn_1_prompt, "USER INPUT")
        turn_2_prompt_with_xml_tagging_defense = xml_tagging(turn_2_prompt, "USER INPUT")

        model_response = ""
        turn_1_response = ""
        turn_2_response = ""
        error_message = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attack_turns == 1:
                    turn_1_response = call_model_single_turn(provider, client, model_name, system_prompt_with_xml_tagging_defense, turn_1_prompt_with_xml_tagging_defense)
                    model_response = turn_1_response
                else:
                    turn_1_response, turn_2_response = call_model_multi_turn(provider, client, model_name, system_prompt_with_xml_tagging_defense, turn_1_prompt_with_xml_tagging_defense, turn_2_prompt_with_xml_tagging_defense)
                    model_response = turn_2_response

                error_message = ""
                break

            except Exception as error:
                error_message = str(error)

                print(
                    f"Attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{error_message}"
                )

                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt

                    print(
                        f"Waiting {wait_time} seconds "
                        f"before retrying..."
                    )

                    time.sleep(wait_time)

        results.append({
            "test_id": test_id,
            "id": prompt_id,
            "dataset": get_text(row.get("dataset", "")),
            "prompt_type": get_text(row.get("prompt_type", "")),
            "user_query": user_query if attack_type == "multi_turn_attack" else get_text(row.get("user_query", "")),
            "provider": provider,
            "model": model_name,
            "attack_type": attack_type,
            "attack_turns": attack_turns,
            "postprocessing_type": get_text(row.get("postprocessing_type", "none")),
            "defense_type": "xml_tagging_defense",
            "system_prompt": system_prompt,
            "system_prompt_with_xml_tagging_defense": system_prompt_with_xml_tagging_defense,
            "turn_1_prompt": turn_1_prompt,
            "turn_2_prompt": turn_2_prompt,
            "turn_1_prompt_with_xml_tagging_defense": turn_1_prompt_with_xml_tagging_defense,
            "turn_2_prompt_with_xml_tagging_defense": turn_2_prompt_with_xml_tagging_defense,
            "turn_1_response": turn_1_response,
            "turn_2_response": turn_2_response,
            "model_response_xml_tagging_defense": model_response,
            "error": error_message
        })

        pd.DataFrame(results).to_csv(
            output_file,
            index=False
        )

        print(f"Saved {len(results)} results to {output_file}")

        time.sleep(DELAY_BETWEEN_CALLS)

    successful_runs = sum(
        1 for result in results
        if result["error"] == ""
    )

    failed_runs = len(results) - successful_runs

    print()
    print(f"{model_name} experiment complete.")
    print(f"Total cases: {len(results)}")
    print(f"Successful cases: {successful_runs}")
    print(f"Failed cases: {failed_runs}")
    print(f"Results saved to: {output_file}")

def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    attack_df = pd.read_csv(INPUT_FILE)

    print("Attack cases loaded successfully.")
    print(f"Rows: {len(attack_df)}")
    print()
    print("Attack counts:")
    print(attack_df["attack_type"].value_counts())
    print()

    openai_client = OpenAI(api_key=openai_api_key)
    gemini_client = genai.Client(api_key=gemini_api_key)
    claude_client = Anthropic(api_key=anthropic_api_key)

    #run_experiment("openai", openai_client, attack_df, MODEL_NAME_GPT_4, OUTPUT_FILE_GPT_4)
    #run_experiment("openai", openai_client, attack_df, MODEL_NAME_GPT_4_1, OUTPUT_FILE_GPT_4_1)
    run_experiment("gemini", gemini_client, attack_df, MODEL_NAME_GEMINI_PRO, OUTPUT_FILE_GEMINI_PRO)
    #run_experiment("claude", claude_client, attack_df, MODEL_NAME_CLAUDE, OUTPUT_FILE_CLAUDE)
    #run_experiment("ollama", None, attack_df, MODEL_NAME_LLAMA_3_8B, OUTPUT_FILE_LLAMA_3_8B)
    #run_experiment("ollama", None, attack_df, MODEL_NAME_MISTRAL_7B, OUTPUT_FILE_MISTRAL_7B)

    print()
    print("All XML tagging defense experiments complete.")

if __name__ == "__main__":
    main()