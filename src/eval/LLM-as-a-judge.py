import json
import os
import time
from http import HTTPStatus
import dashscope
from dashscope import Generation
from tqdm import tqdm

# ================= Language & Model Domain Setup =================
# Global flag to switch between "en" (English) and "zh" (Chinese)
LANGUAGE = "en"

# ================= Configuration Setup =================
# Securely load API Key from environment variables
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_ANONYMOUS_API_KEY")

# Powerful judge model configuration
JUDGE_MODEL = "qwen-max"

# Dynamic mapping according to domain names established previously
DOMAIN = "chemistry" if LANGUAGE == "en" else "lithography"

# Dynamic model source configurations (Pure 6-way Baseline Alignment)
INPUT_SOURCES = {
    "flan_t5": (
        f"./data/output-flan-t5-{DOMAIN}.json",
        "flant5_zeroshot_claim"
    ),
    "FT-llama3": (
        f"./data/output-FT-{DOMAIN}.json",
        "generated_claims"
    ),
    "qwen-turbo": (
        f"./data/output-zero-shot-{DOMAIN}.json",
        "zero_shot_claim"
    ),
    "qwen-max": (
        f"./data/output-two-agents-{DOMAIN}-max.json",
        "zero_shot_claim"
    ),
    "qwen-turbo&self-refine": (
        f"./data/output-selfrefine-{DOMAIN}-turbo.json",
        "api_baseline2_selfrefine_claim"
    ),
    "Ours(Claim-DER)": (
        f"./data/output-two-agents-{DOMAIN}.json",
        "multi_agent_conservative_claim"
    )
}

# Unified file path for cache and results saving
MERGED_OUTPUT_PATH = f"./data/{LANGUAGE}-llm-eval-aligned-judged.json"

DIMENSIONS = ["Feature_Completeness", "Conceptual_Clarity", "Terminology_Consistency", "Logical_Linkage",
              "Overall_Quality"]


# ================= Core Functions =================

def build_merged_dataset():
    """Merge multi-source outputs via inner join on 'number' field"""
    print(f"⏳ [Phase 1] Merging multi-source records for Lang=[{LANGUAGE.upper()}] via Inner Join...")
    temp_dict = {}
    model_names = list(INPUT_SOURCES.keys())

    for m_name, (file_path, claim_key) in INPUT_SOURCES.items():
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File missing for [{m_name}] at '{file_path}'. Excluded.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    num = item.get("publication_number") or item.get("number")
                    if not num:
                        continue
                    draft = item.get(claim_key)
                    if not draft:
                        continue

                    if num not in temp_dict:
                        temp_dict[num] = {
                            "number": num,
                            "published_claim": item.get("published_claim", ""),
                            "drafts": {}
                        }
                    temp_dict[num]["drafts"][m_name] = draft
        except Exception as e:
            print(f"❌ Error reading {m_name}: {e}")
            return [], []

    actual_active_models = [m for m in model_names if os.path.exists(INPUT_SOURCES[m][0])]
    perfect_matches = [record for num, record in temp_dict.items() if
                       len(record["drafts"]) == len(actual_active_models)]
    print(f"✅ Data alignment completed. Found {len(perfect_matches)} records matched across all active models.\n")
    return perfect_matches, actual_active_models


def parse_score(val):
    try:
        if isinstance(val, str):
            return float(val.split('/')[0].strip())
        return float(val)
    except:
        return 0.0


def call_qwen_judge(authorized_claims, drafts_mapping, letters):
    """Call Qwen-Max to perform dynamic N-way blind peer review evaluation"""

    drafts_input_prompt = ""
    json_template_scores = ", ".join([f'"{dim}": <int>' for dim in DIMENSIONS])
    json_template_drafts = ", ".join(
        [f'"{let}": {{"Scores": {{{json_template_scores}}}, "Reasoning": "<brief analysis>"}}' for let in letters])
    winner_options = " or ".join(letters) + " or Tie"

    for let in letters:
        drafts_input_prompt += f"[{let}]: {drafts_mapping[let]}\n\n"

    prompt = f"""
    You are a Senior Patent Strategist acting as an impartial AI Judge. 
    I will provide you with the [Authorized Claims] (as a baseline reference for the core invention) and {len(letters)} drafted versions.

    ### ⚠️ CRITICAL JUDGING PHILOSOPHY ⚠️
    A better patent draft builds a stronger "Defensive Claim Tree". 
    Do NOT penalize a draft simply because it includes MORE dependent claims than the [Authorized Claims]. Reward drafts that successfully extract additional valid, specific embodiments to create fallback dependent claims.

    ### ⚖️ Evaluation Metrics (1-10 for each):
    1. **Feature_Completeness**: Does it capture the core invention AND expand with rich, valid dependent features?
    2. **Conceptual_Clarity**: Are the technical boundaries defined clearly?
    3. **Terminology_Consistency**: Is the antecedent basis strict and terminology consistent?
    4. **Logical_Linkage**: Is the hierarchical structure logically robust?
    5. **Overall_Quality**: Which draft provides the most robust legal defense wall?

    ### Input Data
    [Authorized Claims (Reference)]:
    {authorized_claims}

    {drafts_input_prompt}

    ### Output Format Requirements
    Output ONLY a valid JSON object. No markdown code blocks. 
    Format exactly like this template:
    {{
        {json_template_drafts},
        "Winner": "<{winner_options}>" 
    }}
    """

    messages = [
        {'role': 'system',
         'content': 'You are an objective AI evaluator. Output strictly in valid raw JSON format without markdown ticks.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = Generation.call(model=JUDGE_MODEL, messages=messages, result_format='message', temperature=0.1)
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]['message']['content'].replace("```json", "").replace("```", "").strip()
            return json.loads(content)
    except Exception as e:
        print(f"⚠️ Judge API Execution/Parsing Error: {e}")
    return None


def print_statistics(data, model_names):
    """Aggregate and print the scholastic N-way leaderboard matrix"""
    scores = {name: {dim: 0.0 for dim in DIMENSIONS} for name in model_names}
    wins = {name: 0 for name in model_names}
    valid_count = 0
    ties = 0

    for item in data:
        eval_data = item.get("llm_judge_evaluation")
        mapping = item.get("llm_judge_mapping")
        if not eval_data or not mapping:
            continue

        try:
            for letter, name in mapping.items():
                s = eval_data.get(letter, {}).get("Scores", {})
                for dim in DIMENSIONS:
                    scores[name][dim] += parse_score(s.get(dim, 0))

            winner = str(eval_data.get("Winner", ""))
            matched = False
            for letter, name in mapping.items():
                if letter in winner:
                    wins[name] += 1
                    matched = True
                    break
            if not matched:
                ties += 1
            valid_count += 1
        except:
            continue

    if valid_count == 0:
        print("\n❌ No valid judging score summaries generated yet.")
        return

    table_width = 25 + 18 * len(model_names)
    print("\n" + "=" * table_width)
    print(f"�� Main Metric: N-way LLM-as-a-Judge Table [Lang: {LANGUAGE.upper()} | Samples: {valid_count}]")
    print("=" * table_width)

    header = f"{'Evaluation Dimensions':<24}"
    for name in model_names:
        header += f" | {name:<15}"
    print(header)
    print("-" * table_width)

    for dim in DIMENSIONS:
        row = f"{dim:<24}"
        for name in model_names:
            avg = scores[name][dim] / valid_count
            row += f" | {avg:<15.2f}"
        print(row)

    print("-" * table_width)
    print("�� Win Rate Standings:")
    for name in model_names:
        rate = (wins[name] / valid_count) * 100
        print(f"   [{name:<18}] : {rate:>5.1f}% ({wins[name]} rounds)")
    print(f"   [Tie/Draw            ] : {(ties / valid_count) * 100:>5.1f}% ({ties} rounds)")
    print("=" * table_width + "\n")


# ================= Main Execution =================

def main():
    if os.path.exists(MERGED_OUTPUT_PATH):
        print(f"⏳ Aligned evaluation cache detected. Commencing checkpoint resume: {MERGED_OUTPUT_PATH}")
        with open(MERGED_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            unified_data = json.load(f)
        model_names = list(INPUT_SOURCES.keys())
        model_names = [m for m in model_names if os.path.exists(INPUT_SOURCES[m][0])]
    else:
        unified_data, model_names = build_merged_dataset()
        if not unified_data:
            return

    print(f"⚖️ [Phase 2] Launching Dynamic N-way Peer Judging for {len(unified_data)} records...")
    need_save = False

    for i in tqdm(range(len(unified_data)), desc="Judging"):
        item = unified_data[i]

        if item.get("llm_judge_evaluation"):
            continue

        authorized_claims = item.get('published_claim')
        if not authorized_claims:
            continue

        letters = [f"Draft_{chr(65 + idx)}" for idx in range(len(model_names))]
        drafts_mapping = {}
        for idx, m_name in enumerate(model_names):
            drafts_mapping[letters[idx]] = item["drafts"][m_name]

        try:
            eval_result = call_qwen_judge(authorized_claims, drafts_mapping, letters)
            if eval_result:
                item['llm_judge_evaluation'] = eval_result
                item['llm_judge_mapping'] = {letters[idx]: m_name for idx, m_name in enumerate(model_names)}
                need_save = True
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error compiling score card at index {i + 1}: {e}")

        if need_save and (i + 1) % 5 == 0:
            with open(MERGED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, ensure_ascii=False, indent=4)
            need_save = False

    if need_save:
        with open(MERGED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=4)

    print_statistics(unified_data, model_names)


if __name__ == "__main__":
    main()