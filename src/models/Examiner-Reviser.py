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
TARGET_LANGUAGE = "English"

CRITIC_MODEL = "qwen-turbo"
REVISER_MODEL = "qwen-turbo"

# 3. Input/Output Paths (Anonymized and aligned with Drafter.py output)
INPUT_JSON_PATH = "./data/output-zero-shot-chemistry.json"
OUTPUT_JSON_PATH = "./data/output-two-agents-chemistry.json"


# =========================================================

def call_qwen_examiner(disclosure, initial_draft):
    """
    Agent 1: AI Examiner (Defensive Tree Strategist)
    """
    prompt = f"""
    ### Task Objective
    You are a Master Patent Strategist. Compare the [Initial Draft Claims] with the original [Technical Disclosure].
    Your mission is to build a "Defensive Claim Tree" by extracting valuable technical details from the disclosure and adding them as fallback positions.

    ### ⚠️ STRATEGIC RULES (MUST FOLLOW) ⚠️
    1. **DO NOT TOUCH CLAIM 1**: The independent claim must remain broad. DO NOT add any new limitations to Claim 1.
    2. **Hierarchical Expansion**: Dig deep into the [Technical Disclosure]. Find specific embodiments (e.g., specific temperatures, materials, exact mechanisms) that are missing from the current draft.
    3. **Actionable Addition**: Instruct the attorney to add these missing specific features ONLY as NEW DEPENDENT CLAIMS at the end of the claim set. Do NOT output NONE unless the draft has exhausted every single detail from the disclosure.

    ### Input Data
    [Technical Disclosure]:
    {disclosure}

    [Initial Draft Claims]:
    {initial_draft}

    ### Output Requirements
    **[Defect Analysis]**
    (List the specific valuable features from the disclosure that are currently missing in the dependent claims.)

    **[Revision Directives]**
    (Provide exact commands to append new dependent claims. Write in {TARGET_LANGUAGE}.)
    """

    messages = [
        {'role': 'system', 'content': 'You are a Master Patent Strategist focused on building robust defensive claim trees via dependent claims.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(model=CRITIC_MODEL, messages=messages, result_format='message', temperature=0.3)
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content']
    except Exception as e:
        print(f"Examiner API Error: {e}")
    return None


def call_qwen_reviser(initial_draft, critique):
    """
    Agent 2: Lead Attorney (Reviser)
    """
    prompt = f"""
    You are a meticulous Senior Patent Attorney. 
    A Patent Examiner has reviewed your [Initial Draft Claims] and provided [Revision Directives].
    Your task is to refine the claims by strictly incorporating the Examiner's feedback.

    ### ⚠️ CRITICAL REWRITE RULES (CONSERVATIVE ANCHOR) ⚠️
    1. **Conservative Refinement**: Your rewrite must be HIGHLY CONSERVATIVE. You MUST retain 95% of the original text structure. Simply weave the new requested features into the existing claims, or append new dependent claims.
    2. **No Deletion**: DO NOT delete existing valid claims, steps, or features that were not explicitly criticized by the Examiner.
    3. **Global Consistency**: Ensure the "antecedent basis" (e.g., 'a/an' vs 'the/said') is perfectly maintained after your edits.
    4. **Auto-Numbering**: Ensure all claims are sequentially numbered.
    5. **No Filler**: Output ONLY the full text of the revised claims, starting from "1. A...".

    ### Input Data
    [Initial Draft Claims]:
    {initial_draft}

    [Revision Directives]:
    {critique}
    """

    messages = [
        {'role': 'system',
         'content': 'You are a meticulous Attorney. You weave changes into drafts while preserving original integrity.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(model=REVISER_MODEL, messages=messages, result_format='message', temperature=0.2)
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]['message']['content']
            return content.replace("```text", "").replace("```", "").strip()
    except Exception as e:
        print(f"Reviser API Error: {e}")
    return None


def main():
    print(f"Loading dataset from: {INPUT_JSON_PATH}")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Starting Multi-Agent Refinement, total samples: {len(data)}...")

    for i, item in enumerate(tqdm(data, desc="Optimizing")):

        disclosure = item.get('draft_description')
        initial_draft = item.get('zero_shot_claim')

        if not initial_draft or not disclosure:
            continue

        if item.get('multi_agent_conservative_claim'):
            continue

        try:
            # === Step 1: Critic Phase ===
            critique = call_qwen_examiner(disclosure, initial_draft)
            if not critique:
                continue
            item['multi_agent_critique'] = critique

            time.sleep(0.5)

            # === Step 1.5: Short-circuit Mechanism ===
            if "NONE" in critique.upper():
                print(f"  [Short-circuit Triggered] Index {i + 1} initial draft is optimal. 100% preserved.")
                item['multi_agent_conservative_claim'] = initial_draft
            else:
                # === Step 2: Reviser Phase ===
                final_claim = call_qwen_reviser(initial_draft, critique)
                if final_claim:
                    item['multi_agent_conservative_claim'] = final_claim

            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing index {i + 1}: {e}")
            item['multi_agent_error'] = str(e)

        # Periodic checkpoint saving
        if (i + 1) % 5 == 0:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    # Final Save
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Multi-agent refinement pipeline completed successfully.")


if __name__ == "__main__":
    main()