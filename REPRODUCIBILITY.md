# Claim-DER reproducibility materials

This release accompanies the revised Claim-DER experiments on 1,000 English and 1,000 Chinese patents. It includes frozen input texts, sample manifests, saved per-patent scores, table reconstruction, metric functions, generation prompts, baseline and controlled-ablation runners, and judge protocols.

**Archive status:** all 24 original output exports are included byte-identically under `outputs/`: 14 main-method exports and 10 controlled-ablation exports, including draft-only outputs. All 22 hashes recorded during earlier evaluation match. The two draft-only hashes were first captured during recovery and additionally checked against per-case draft checkpoints and saved metric lengths.

`provenance/ablation_stages.jsonl` contains compact response records for the 2,794 corrected-ablation requests. Every stored request was checked against the frozen source text, shared draft, feedback, exact prompt, temperature, token cap and seed. Request hashes, final-output bindings and recorded reused-stage hashes were verified. Full corrected-ablation request/checkpoint files are retained in the private recovery archive. Main baseline intermediate checkpoints and raw judge response trees remain on the experiment server; this package does not claim they are included.

`protocols/expected_output_hashes.json`, `outputs/IMPORT_MANIFEST.json`, and `provenance/recovery_checks.json` record these checks. The earlier candidate's portable Self-Refine loader lacked archived initial drafts. This release corrects it to reuse the 999 English and 1,000 Chinese saved Qwen-Turbo drafts; the original direct-fill branch applies only to the one missing English initial draft. No published result or score was changed.

## Reconstruct saved table values (offline)

Python's standard library is sufficient:

```sh
python verify_release.py
python verify_outputs.py
python rebuild_tables.py --output rebuilt
```

The third command rebuilds seven CSV views: the two main result tables, supplementary ROUGE, BERTScore and structural checks, Qwen3-Max scores, and controlled ablation scores. It checks 974 values and denominators against the saved summaries. It does not call an API. Bootstrap intervals, metric–judge correlations, cross-judge matched results, and order diagnostics are included as archived CSVs; this command does not recompute their resampling procedures. CSV scores retain full precision; round only for display.

The English/Chinese common-output cohorts are 990/992 patents. Qwen-Max main judgments cover 953/950, and Qwen3-Max covers 989/992. Automatic controlled-ablation comparisons cover 197/199 of the 200 selected patents per language, while their paired judge comparison covers 196/197. Missing cases are retained as missing, never replaced.

## Data and sampling

`dataset/data/english_1000.json` and `chinese_1000.json` contain only `publication_number`, `draft_description` and `published_claim`. These fields are copied without changes from the frozen evaluation files; unrelated historical/model fields were omitted. Original file hashes, release file hashes and per-field hashes are supplied. Generation loaders expose the disclosure and ID only; reference claims are used exclusively by evaluation.

The source pools are the Patent-CR-derived English chemistry collection (2,931 patents) and the self-collected Patent-LG-zh Chinese lithography collection (1,419 patents). For each language independently, sampling used `sorted(random.Random(20260929).sample(sorted(ids), 1000))`. Sampling did not use outputs, scores, output availability, or length resampling. Stored disclosures retain their previous 30,000-character cap. `length_distribution.json` and `selected_samples.csv` preserve the length check. Pool-to-subset comparisons are descriptive; the two language collections also differ in domain.

## Score new or recovered texts (offline)

Use an isolated Python environment and install `requirements-metrics.txt`. Install the spaCy model versions below if using DAR, and install the NLTK resources `punkt_tab`, `averaged_perceptron_tagger_eng`, and `stopwords`:

```sh
python -m pip install -r requirements-metrics.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng stopwords
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
python -m pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
python evaluate_texts.py --language en --predictions predictions.json --output scores.json --dar
```

The prediction file is a JSON list of `{ "publication_number": "...", "text": "..." }`. Duplicate, empty or unknown records are rejected. Use the same patent-ID intersection across methods before comparison. The evaluator returns fractions; table scores are percentages. DAR aggregates **summed failing nodes divided by summed eligible nodes**, not the mean of per-patent ratios. In archived CSVs, the main DAR per-case column is already a percentage, while the ablation DAR column is a fraction; node numerator/denominator fields are the preferred aggregation source. The portable text evaluator consistently returns fractions.

- ROUGE uses `rouge-score` stemming/tokenization for English and lowercased Jieba tokens for Chinese. The main RG column is ROUGE-L recall; the old Chinese ASCII-tokenized score remains only in the archived PAR CSV for provenance.
- PAR preserves the frozen keyword extraction and first-occurrence regional mapping. Empty reference-keyword regions are undefined (`null`), and omitted from the regional macro mean. Keyword repetition cannot add recall; lexical matches do not establish factual support.
- DAR preserves the frozen parsing/entity/health functions. No eligible reference nodes means undefined (`null`). English DAR is sparse (42 contributing patents, 54 nodes in the main common cohort); it is an auxiliary diagnostic, not overall dependency or legal validity.
- Surface checks report explicit numbering and reference validity plus parseability. Unparseable outputs do not pass. These are syntactic checks, not substantive patent examination.

`examples/metric_trace.json` contains the paper's constructed examples, not corpus validation. `python smoke_metrics.py` checks these examples and important empty-denominator/branch behavior using the pinned NLP resources.

BERTScore is supplied as saved per-patent scores with its encoder revisions and protocol in `protocols/bertscore.json`. It uses 510 content-token right truncation, no IDF and no baseline rescaling. This release's portable text evaluator does not rerun BERTScore; the archived scoring script is included under `protocols/bertscore_reference_code.py`. Its original archive layout must be restored before use.

## Prepare a new baseline run (explicit paid execution)

```sh
python -m pip install -r requirements-generation.txt
python prepare_run.py baselines --output runs/baselines_new
python runs/baselines_new/run_baselines.py --config runs/baselines_new/config.json --mode audit
```

Preparation and audit are offline. They verify all 2,000 frozen input records. Set `DASHSCOPE_API_KEY` in your own environment. Only changing `--mode audit` to `--mode pilot` or `--mode run` initiates paid DashScope calls. Keep credentials outside this package.

This runner covers Qwen-Turbo Self-Refine (one cycle), patent-specific implicit CoT and adapted PlanGEN-BoN5. It preserves the recorded prompts, stage temperatures, token budgets, seed derivation and response handling. It exports only those newly generated methods. The four archived main methods (FT-Llama3, Qwen-Turbo, Qwen-Max and Claim-DER) are included under `outputs/main/` and are not regenerated by this command. Self-Refine loads its initial draft from the archived Qwen-Turbo outputs. New API runs need not reproduce historical texts or scores because model aliases and provider nondeterminism can change.

## Prepare a controlled ablation run

```sh
python prepare_run.py ablation --output runs/ablation_new
python runs/ablation_new/run_ablation.py --mode audit
python runs/ablation_new/test_protocol.py
```

After setting the environment key, an explicit `--mode pilot` run processes the six fixed pilot patents. The `--mode run` command requires successful pilot validation before processing the frozen 400 patents. The runner uses one fresh shared draft per patent, targeted/generic feedback, and four final conditions: full DER, generic feedback, full disclosure provided to the Reviser, and relaxed conservative revision. It also exports the draft-only diagnostic. A/C/D share targeted feedback and can have identical outputs when the Examiner returns exactly `NONE`.

Short-circuiting is `text.strip().upper() == "NONE"`, not substring matching. The original public code's substring behavior is superseded for this controlled ablation. The adapters preserve prompt bodies and generation logic, with portable data loading and environment-only credential handling. A **new release manifest** is written to each prepared run; it is not represented as the historical execution manifest. Original hashes remain in `protocols/ablation_original_hashes.json`. The archived corrected experiment reused valid stages from its preceding run; preparing a new run starts fresh and is not a replay of that correction history.

## Judge protocol and evidence limits

`protocols/judges/` contains Qwen-Max, Qwen3-Max and ablation judge rubrics, configurations, and archival reference scripts. Five dimensions are scored: feature completeness, conceptual clarity, terminology consistency, logical linkage and overall quality. Main outputs were anonymized and order-balanced; ablation judges deduplicated identical texts within a patent. The reference scripts depend on the original output/checkpoint layout and are not the portable launch interface. Restoring that archive is required before rerunning them.

These are **LLM judgments**, not human expert ratings. Qwen-family judgments are not fully independent of Qwen-family generation. Qwen-Max used an unpinned model alias; historical generation snapshots and the exact main Claim-DER execution binding are not fully verified. FT-Llama3 has unresolved checkpoint/training binding and 16 possible English training overlaps. No new cross-family DER experiment, independent human assessment or graph-repair guarantee is claimed. See `protocols/provenance.json` for the limits retained in the revised paper.

## Layout and attribution

`code/` holds portable generation and metric modules; `protocols/` holds exact prompts, relevant archived configurations and provenance; `scores/` holds saved score-level evidence; `dataset/` holds the frozen evaluation projection; `outputs/` holds original generated-text exports; `provenance/` holds recovery checks and compact ablation traces. Generated run directories are separate and must not be published with credentials or unscreened logs.

The English collection is derived from Patent-CR. Self-Refine and PlanGEN are adapted baseline methods; their original works are cited in the accompanying paper. The package uses NLTK, Jieba, spaCy, ROUGE and BERTScore; their upstream licenses continue to apply. This preparation step does not assign a new license to source collections or third-party materials.
