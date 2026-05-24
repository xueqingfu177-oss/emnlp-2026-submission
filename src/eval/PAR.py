import json
import os
import re
from collections import Counter
import numpy as np
from tqdm import tqdm

# ================= Language-Specific Imports & Setup =================
# Global flag to switch between "en" (English) and "zh" (Chinese)
LANGUAGE = "en"

if LANGUAGE == "en":
    import nltk
    from nltk.corpus import stopwords

    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("⏳ Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('stopwords')
        print("✅ NLTK setup completed.\n")
elif LANGUAGE == "zh":
    import jieba
    import jieba.analyse

# ================= Configuration Setup =================
# Anonymized paths aligned with repository standards
MERGED_OUTPUT_PATH = f"./data/{LANGUAGE}-eval-positional-recall.json"
TOP_K_FEATURES = 20

# Automatically map dataset domain based on language selection
DOMAIN = "chemistry" if LANGUAGE == "en" else "lithography"

# Unified dictionary template matching your new dataset names (Pure 6-way Baselines)
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


# ================= Core Functions =================

def extract_keywords(text, top_k):
    """Extract language-specific core keywords from text"""
    if not text:
        return []

    if LANGUAGE == "en":
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        stop_words = set(stopwords.words('english'))
        valid_tags = ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
        valid_words = [word for word, tag in tagged_tokens if
                       word.isalpha() and word not in stop_words and tag in valid_tags]
        word_counts = Counter(valid_words)
        return [word for word, count in word_counts.most_common(top_k)]

    elif LANGUAGE == "zh":
        allowed_pos = ('n', 'vn', 'v', 'nz', 'eng')
        return jieba.analyse.extract_tags(text, topK=top_k, allowPOS=allowed_pos)


def map_features_to_positions(features, disclosure_text):
    """Map keywords back to the original technical disclosure and categorize into 3 zones"""
    head_feats, mid_feats, tail_feats = [], [], []
    if not disclosure_text:
        return head_feats, mid_feats, tail_feats

    disclosure_lower = disclosure_text.lower()
    total_len = len(disclosure_lower)

    for kw in features:
        kw_lower = kw.lower()
        if LANGUAGE == "en":
            match = re.search(r'\b' + re.escape(kw_lower) + r'\b', disclosure_lower)
            match_idx = match.start() if match else -1
        else:
            match_idx = disclosure_lower.find(kw_lower)

        if match_idx != -1:
            rel_pos = match_idx / total_len
            if rel_pos < 0.333:
                head_feats.append(kw)
            elif rel_pos < 0.666:
                mid_feats.append(kw)
            else:
                tail_feats.append(kw)
    return head_feats, mid_feats, tail_feats


def build_merged_dataset():
    """Merge multi-source model outputs via inner join on 'number' field"""
    print(f"⏳ [Phase 1] Merging multi-source datasets for Lang=[{LANGUAGE}] (Domain=[{DOMAIN}]) via Inner Join...")
    temp_dict = {}
    model_names = list(INPUT_SOURCES.keys())

    for m_name, (file_path, claim_key) in INPUT_SOURCES.items():
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: Optional model file missing for [{m_name}] at '{file_path}'. Excluded from evaluation.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    num = item.get("number") or item.get("publication_number")
                    gold = item.get("published_claim")
                    draft = item.get(claim_key)
                    disclosure = item.get("draft_description")

                    if not num or not gold or not draft or not disclosure:
                        continue

                    if num not in temp_dict:
                        temp_dict[num] = {
                            "number": num,
                            "published_claim": gold,
                            "draft_description": disclosure,
                            "drafts": {}
                        }
                    temp_dict[num]["drafts"][m_name] = draft
        except Exception as e:
            print(f"❌ Error loading file for {m_name}: {e}")
            return [], []

    actual_active_models = [m for m in model_names if os.path.exists(INPUT_SOURCES[m][0])]
    perfect_matches = [record for num, record in temp_dict.items() if
                       len(record["drafts"]) == len(actual_active_models)]
    print(f"✅ Data alignment completed. Found {len(perfect_matches)} matched records across all active baselines.")

    with open(MERGED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(perfect_matches, f, ensure_ascii=False, indent=4)
    return perfect_matches, actual_active_models


# ================= Main Execution =================

def main():
    if os.path.exists(MERGED_OUTPUT_PATH):
        try:
            os.remove(MERGED_OUTPUT_PATH)
        except Exception:
            pass

    aligned_data, model_names = build_merged_dataset()
    if not aligned_data:
        return

    valid_count = len(aligned_data)
    recalls_dict = {name: {'head': [], 'mid': [], 'tail': []} for name in model_names}

    print(f"⚙️ [Phase 2] Computing Position-Aware Recall (PAR) for {valid_count} samples...")

    for item in aligned_data:
        gold = item.get('published_claim', '')
        disclosure = item.get('draft_description', '')

        gold_keywords = extract_keywords(gold, TOP_K_FEATURES)
        head_kw, mid_kw, tail_kw = map_features_to_positions(gold_keywords, disclosure)

        for m_name in model_names:
            draft_lower = item['drafts'].get(m_name, '').lower()

            def calc_recall(kw_list):
                if not kw_list:
                    return None
                if LANGUAGE == "en":
                    hits = sum(1 for kw in kw_list if re.search(r'\b' + re.escape(kw.lower()) + r'\b', draft_lower))
                else:
                    hits = sum(1 for kw in kw_list if kw.lower() in draft_lower)
                return hits / len(kw_list)

            r_head = calc_recall(head_kw)
            r_mid = calc_recall(mid_kw)
            r_tail = calc_recall(tail_kw)

            if r_head is not None: recalls_dict[m_name]['head'].append(r_head)
            if r_mid is not None: recalls_dict[m_name]['mid'].append(r_mid)
            if r_tail is not None: recalls_dict[m_name]['tail'].append(r_tail)

    # ================= Scholarly Display Mapping =================
    display_names = {
        "flan_t5": "Flan-T5",
        "FT-llama3": "FT-Llama3",
        "qwen-turbo": "Qwen-Turbo",
        "qwen-max": "Qwen-Max",
        "qwen-turbo&self-refine": "Self-Refine",
        "Ours(Claim-DER)": "Claim-DER (Ours)"
    }

    # Sort models nicely ensuring Ours and big baselines stand at prominent columns
    sorted_models = [m for m in display_names.keys() if m in model_names]
    for m in model_names:
        if m not in sorted_models: sorted_models.append(m)

    # ================= Scholarly Leaderboard Printing =================
    table_width = 16 + 25 * len(sorted_models)
    print("\n" + "=" * table_width)
    print(f"�� Main Metric: Position-Aware Recall (PAR) Table [Language: {LANGUAGE.upper()} | Domain: {DOMAIN.upper()}]")
    print("=" * table_width)

    header = f"{'PAR Zones':<14}"
    for name in sorted_models:
        header += f" | {display_names.get(name, name):<18}"
    print(header)
    print("-" * table_width)

    for zone in ['head', 'mid', 'tail']:
        row = f"{zone.capitalize()} Recall"
        for name in sorted_models:
            arr = recalls_dict[name][zone]
            avg_recall = np.mean(arr) * 100 if arr else 0.0
            row += f" | {avg_recall:>15.2f}%"
        print(row)
    print("=" * table_width)


if __name__ == "__main__":
    main()