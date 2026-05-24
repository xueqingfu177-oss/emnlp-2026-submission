import json
import os
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

# ================= Language & Model Domain Setup =================
# Global flag to switch between "en" (English) and "zh" (Chinese)
LANGUAGE = "en"

if LANGUAGE == "zh":
    import jieba

# ================= Configuration Setup =================
# Automatically aligned with previous pipeline output paths
INPUT_JSON_PATH = f"./data/{LANGUAGE}-eval-positional-recall.json"

# Academic display naming mapping for paper tables (Pure 6-way Baselines)
DISPLAY_NAMES = {
    "flan_t5": "Flan-T5",
    "FT-llama3": "FT-Llama3",
    "qwen-turbo": "Qwen-Turbo",
    "qwen-max": "Qwen-Max",
    "qwen-turbo&self-refine": "Self-Refine",
    "qwen-turbo&M": "Claim-DER (Ours)"
}


# ================= Core Tokenization Wrapper =================

def preprocess_text(text):
    """Normalize token boundaries based on language rules to prevent ROUGE breakdown"""
    if not text:
        return ""
    if LANGUAGE == "en":
        return text
    elif LANGUAGE == "zh":
        # Crucial fix for Chinese: segment tokens and re-join with whitespace characters
        return " ".join(jieba.cut(text))


# ================= Main Execution Pipeline =================

def main():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ Error: Cannot find matched evaluation cache file at {INPUT_JSON_PATH}")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    if not data_list:
        print("❌ Error: Cached dataset file is empty.")
        return

    # Initialize ROUGE scorer (Use stemming for English structural syntax matches)
    use_stemmer_flag = True if LANGUAGE == "en" else False
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer_flag)

    # Dynamic target base model list detection from cached data records
    raw_models = list(data_list[0].get("drafts", {}).keys())

    # Sort models nicely ensuring Ours and big baselines stand at prominent columns
    sorted_models = [m for m in DISPLAY_NAMES.keys() if m in raw_models]
    for m in raw_models:
        if m not in sorted_models:
            sorted_models.append(m)

    # Container initializing pure recall score arrays
    recall_results = {name: {"r1_recall": [], "r2_recall": [], "rl_recall": []} for name in sorted_models}

    print(f"⚙️ Computing Full ROUGE Recall Matrix for {len(data_list)} samples [Language: {LANGUAGE.upper()}]...")

    for item in tqdm(data_list, desc="Processing"):
        gold = item.get('published_claim', '')
        if not gold:
            continue

        gold_norm = preprocess_text(gold)
        drafts = item.get('drafts', {})

        for m_name in sorted_models:
            draft = drafts.get(m_name, '')
            if not draft:
                continue

            draft_norm = preprocess_text(draft)

            try:
                scores = scorer.score(gold_norm, draft_norm)
                # Strict core filtering: isolate .recall metrics, discarding precision/F1 noises
                recall_results[m_name]["r1_recall"].append(scores['rouge1'].recall)
                recall_results[m_name]["r2_recall"].append(scores['rouge2'].recall)
                recall_results[m_name]["rl_recall"].append(scores['rougeL'].recall)
            except Exception:
                pass

    # ================= Scholarly Leaderboard Printing =================
    table_width = 24 + 25 * len(sorted_models)
    print("\n" + "=" * table_width)
    print(f"�� Matrix: ROUGE Recall Leaderboard [Lang: {LANGUAGE.upper()}]")
    print("�� Academic Rule: This table evaluates what percentage of the ground truth tokens are successfully hit.")
    print("=" * table_width)

    # Print Table Headers
    header = f"{'ROUGE Recall Metrics':<24}"
    for name in sorted_models:
        header += f" | {DISPLAY_NAMES.get(name, name):<22}"
    print(header)
    print("-" * table_width)

    # 1. Print ROUGE-1 Recall (Unigram coverage)
    row_r1 = f"{'ROUGE-1 Recall':<24}"
    for name in sorted_models:
        val = np.mean(recall_results[name]["r1_recall"]) * 100 if recall_results[name]["r1_recall"] else 0.0
        row_r1 += f" | {val:>19.2f}%"
    print(row_r1)

    # 2. Print ROUGE-2 Recall (Bigram technical phrasing match)
    row_r2 = f"{'ROUGE-2 Recall':<24}"
    for name in sorted_models:
        val = np.mean(recall_results[name]["r2_recall"]) * 100 if recall_results[name]["r2_recall"] else 0.0
        row_r2 += f" | {val:>19.2f}%"
    print(row_r2)

    # 3. Print ROUGE-L Recall (Longest Common Subsequence / Syntactic structural framework)
    row_rl = f"{'ROUGE-L Recall':<24}"
    for name in sorted_models:
        val = np.mean(recall_results[name]["rl_recall"]) * 100 if recall_results[name]["rl_recall"] else 0.0
        row_rl += f" | {val:>19.2f}%"
    print(row_rl)

    print("=" * table_width + "\n")


if __name__ == "__main__":
    main()