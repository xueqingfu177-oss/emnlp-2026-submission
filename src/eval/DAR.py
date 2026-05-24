import json
import os
import re
import numpy as np
from rouge import Rouge
from tqdm import tqdm

# ================= Language-Specific Imports & Setup =================
LANGUAGE = "en"

print(f"⏳ Initializing environment and NLP models for Language=[{LANGUAGE.upper()}]...")

if LANGUAGE == "en":
    import nltk
    import spacy
    from nltk.corpus import stopwords

    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('stopwords', quiet=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
elif LANGUAGE == "zh":
    import spacy
    import jieba

    try:
        nlp = spacy.load("zh_core_web_sm")
    except Exception:
        os.system("python -m spacy download zh_core_web_sm")
        nlp = spacy.load("zh_core_web_sm")

# ================= Configuration Setup =================
INPUT_JSON_PATH = f"./data/{LANGUAGE}-eval-positional-recall.json"


# ================= Core Analytical Tools =================

def extract_keywords(text):
    """Extract core technical feature tokens for omission rate estimation"""
    if not text:
        return set()

    if LANGUAGE == "en":
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        stop_words = set(stopwords.words('english'))
        valid_tags = ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
        words = [w for w, t in tagged if w.isalpha() and w not in stop_words and t in valid_tags]
        return set(words)

    elif LANGUAGE == "zh":
        doc = nlp(text)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'is', 'and', 'of', 'said', 'this', 'that'}
        valid_tags = ('NN', 'NR', 'VV')
        keywords = set()
        for token in doc:
            if not token.is_punct and token.text not in stop_words and token.pos_ in valid_tags and len(token.text) > 1:
                keywords.add(token.text)
        return keywords


def parse_claims(text):
    """Parse unified patent claim trees across English and Chinese regex variations"""
    claims = {}
    if not text:
        return claims

    if LANGUAGE == "en":
        pattern = re.compile(r'\b(\d+)\.\s+(.*?)(?=\b\d+\.\s+|$)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            cid = int(match[0])
            content = match[1].strip()
            parent_match = re.search(r'claim\s+(\d+)', content, re.IGNORECASE)
            parent_id = int(parent_match.group(1)) if parent_match else 0
            claims[cid] = {"text": content, "parent": parent_id}
    else:
        pattern = re.compile(r'(?:^|\n)\s*(\d+)[\.．、]\s*(.*?)(?=(?:^|\n)\s*\d+[\.．、]|$)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            cid = int(match[0])
            content = match[1].strip()
            parent_match = re.search(r'claim\s*[\(（]?\s*(\d+)\s*[\)）]?', content, re.IGNORECASE)
            parent_id = int(parent_match.group(1)) if parent_match else 0
            claims[cid] = {"text": content, "parent": parent_id}

    return claims


def extract_explicit_entities(text):
    """Extract introduced and referenced base entities for Antecedent Basis check"""
    introduced, referenced = set(), set()
    if not text:
        return introduced, referenced

    if LANGUAGE == "en":
        doc = nlp(text.lower())
        for chunk in doc.noun_chunks:
            words = chunk.text.split()
            if not words: continue
            core_noun = words[-1] if words[-1].isalpha() else ""
            if not core_noun: continue
            first_word = words[0]
            if first_word in ['a', 'an']:
                introduced.add(core_noun)
            elif first_word in ['the', 'said']:
                referenced.add(core_noun)

    elif LANGUAGE == "zh":
        doc = nlp(text)
        ref_kws = {'said', 'the', 'this', 'that'}
        valid_noun_tags = {'NN', 'NR', 'NOUN', 'PROPN'}
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.text in ref_kws:
                j = i + 1
                entity_parts = []
                while j < len(doc) and doc[j].pos_ in {'NN', 'NR', 'ADJ', 'NOUN', 'PROPN'}:
                    entity_parts.append(doc[j].text)
                    j += 1
                if entity_parts:
                    referenced.add("".join(entity_parts))
                i = j
                continue
            if token.pos_ in valid_noun_tags and len(token.text) > 1:
                introduced.add(token.text)
            i += 1

    return introduced, referenced


def get_node_health_status(text):
    """Verify entity tracking integrity across nodes in a claim set"""
    health_status = {}
    claims = parse_claims(text)
    if not claims:
        return health_status

    history_entities = {0: set()}

    for cid in sorted(claims.keys()):
        info = claims[cid]
        parent_id = info["parent"]
        inherited = set(history_entities.get(parent_id, set()))
        intro, ref = extract_explicit_entities(info["text"])

        current_valid_pool = inherited.union(intro)

        is_healthy = True
        for r in ref:
            if r not in current_valid_pool:
                is_healthy = False
                break

        health_status[cid] = is_healthy
        history_entities[cid] = current_valid_pool

    return health_status


# ================= Evaluation Core Pipeline =================

def run_evaluation(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: Cannot find evaluated cache file at {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    rouge = Rouge()
    sample_drafts = data_list[0].get("drafts", {})

    # Configuration-driven dynamic mapping from input datasets
    models = list(sample_drafts.keys())

    stats = {m: {
        "claim1_sim": [],
        "omission": [],
        "gold_healthy_nodes": 0,
        "regressed_nodes": 0
    } for m in models}

    print(f"\n�� Running Comprehensive Evaluation Pipeline... (Total Samples: {len(data_list)})")

    for item in tqdm(data_list, desc="Evaluating"):
        gold_text = item.get("published_claim", "")
        g_claims = parse_claims(gold_text)
        g_claim1 = g_claims.get(1, {}).get("text", "") if gold_text else ""
        g_features = extract_keywords(gold_text) if gold_text else set()
        gold_health = get_node_health_status(gold_text) if gold_text else {}

        drafts = item.get("drafts", {})

        for model in models:
            gen_text = drafts.get(model, "")
            if not gen_text:
                continue

            gen_claims = parse_claims(gen_text)
            gen_claim1 = gen_claims.get(1, {}).get("text", "")
            gen_features = extract_keywords(gen_text)
            gen_health = get_node_health_status(gen_text)

            # Metric A: Claim 1 Fidelity via ROUGE-L
            if g_claim1 and gen_claim1:
                try:
                    if LANGUAGE == "en":
                        score = rouge.get_scores(gen_claim1, g_claim1)[0]['rouge-l']['f']
                    else:
                        gen_claim1_seg = " ".join(jieba.cut(gen_claim1))
                        g_claim1_seg = " ".join(jieba.cut(g_claim1))
                        score = rouge.get_scores(gen_claim1_seg, g_claim1_seg)[0]['rouge-l']['f']
                    stats[model]["claim1_sim"].append(score)
                except Exception:
                    pass

            # Metric B: Technical Feature Omission Rate
            if g_features:
                missing = g_features - gen_features
                stats[model]["omission"].append(len(missing) / len(g_features))

            # Metric C: Dependency-Aware Regression (DAR) vs Gold Standard Nodes
            for cid, is_gold_healthy in gold_health.items():
                if is_gold_healthy:
                    stats[model]["gold_healthy_nodes"] += 1
                    if cid not in gen_health or not gen_health[cid]:
                        stats[model]["regressed_nodes"] += 1

    # ================= Display Name Standardization Mapping =================
    display_names = {
        "flan_t5": "Flan-T5",
        "FT-llama3": "FT-Llama3",
        "qwen-turbo": "Qwen-Turbo",
        "qwen-max": "Qwen-Max",
        "qwen-turbo&self-refine": "Self-Refine",
        "qwen-turbo&M": "Claim-DER (Ours)"
    }

    # Sort models nicely ensuring Ours and big baselines stand at prominent columns
    sorted_models = [m for m in display_names.keys() if m in models]
    for m in models:
        if m not in sorted_models: sorted_models.append(m)

    col_claim1 = "Claim 1 Sim (↑)"
    col_omission = "Omission Rate (↓)"
    col_regression = "DAR (↓)"

    print("\n" + "=" * 125)
    print(f"{'Evaluated Models':<25} | {col_claim1:<16} | {col_omission:<16} | {'Gold Healthy':<12} | {'Broken Nodes':<12} | {col_regression}")
    print("-" * 125)

    for m in sorted_models:
        c1 = (sum(stats[m]["claim1_sim"]) / len(stats[m]["claim1_sim"])) * 100 if stats[m]["claim1_sim"] else 0
        om = (sum(stats[m]["omission"]) / len(stats[m]["omission"])) * 100 if stats[m]["omission"] else 0

        total_healthy = stats[m]["gold_healthy_nodes"]
        regressed = stats[m]["regressed_nodes"]
        reg_rate = (regressed / total_healthy * 100) if total_healthy > 0 else 0

        name_str = display_names.get(m, m)
        print(f"{name_str:<25} | {c1:>14.2f}% | {om:>14.2f}% | {total_healthy:>12} | {regressed:>12} | {reg_rate:>13.2f}%")

    print("=" * 125)


if __name__ == "__main__":
    run_evaluation(INPUT_JSON_PATH)