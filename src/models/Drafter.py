import json
import os
import time
from http import HTTPStatus
import dashscope
from dashscope import Generation
from tqdm import tqdm

# ================= Configuration Setup =================

# 1. API Key Configuration (Securely load from environment variables)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_ANONYMOUS_API_KEY")

# 2. Model Selection
BASELINE_MODEL = "qwen-turbo"

# 3. Input/Output Paths (Anonymized and converted to relative paths)
INPUT_JSON_PATH = "./data/chemistry.json"
OUTPUT_JSON_PATH = "./data/output-zero-shot-chemistry.json"

# =========================================================
TARGET_LANGUAGE = "English"


def call_qwen_draft_claims(disclosure):
    """
    Call Qwen model for zero-shot patent claims generation
    """
    prompt = f"""
    You are a senior patent attorney proficient in patent law. 
    Please read the following [Technical Disclosure] and draft a complete set of patent claims that meet the standards for allowance. 

    ### ⚠️ CRITICAL LANGUAGE RULE ⚠️
    You MUST output the final claims strictly in {TARGET_LANGUAGE}. 
    Ensure you use the standard, formal patent drafting terminology specific to {TARGET_LANGUAGE}.

    ### Requirements: 
    1. Independent claims must contain essential technical features to ensure novelty.
    2. Dependent claims must provide progressive limitations. 
    3. The drafting must reflect rigorous logic and standardized patent terminology.
    4. Output ONLY the claims, starting from "1. A..." without any conversational filler or explanations.

    ### Input Data
    [Technical Disclosure]:
    {disclosure}
    """

    messages = [
        {'role': 'system',
         'content': 'You are a highly skilled Patent Attorney. You output strictly formatted patent claims based on technical disclosures.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(
            model=BASELINE_MODEL,
            messages=messages,
            result_format='message',
            temperature=0.5
        )

        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]['message']['content']
            return content.replace("```text", "").replace("```", "").strip()
        else:
            print(f"API Request Failed: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"API Error: {e}")
        return None


def main():
    # 1. Load Dataset
    print(f"Loading dataset from: {INPUT_JSON_PATH}")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Starting Inference (Zero-Shot), total samples: {len(data)}...")

    # 2. Iterate and Process
    for i, item in enumerate(tqdm(data, desc="Processing")):

        disclosure = item.get('draft_description')

        if not disclosure:
            print(f"Warning: Index {i + 1} missing 'draft_description', skipping.")
            continue

        # Checkpoint resume protection
        if item.get('zero_shot_claim'):
            continue

        try:
            draft = call_qwen_draft_claims(disclosure)

            if draft:
                item['zero_shot_claim'] = draft

            time.sleep(0.5)  # Rate limit protection

        except Exception as e:
            print(f"Error processing index {i + 1}: {e}")
            item['zero_shot_error'] = str(e)

        # Periodic checkpoint saving
        if (i + 1) % 5 == 0:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    # 3. Final Save
    print(f"Saving final results to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()