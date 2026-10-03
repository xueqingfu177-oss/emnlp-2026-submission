# Claim-DER: Patent Claim Generation

This repository accompanies the revised Claim-DER experiments. The current reproducibility release uses 1,000 English and 1,000 Chinese patents sampled independently with seed 20260929. The seven-method common-output comparison covers 990 English and 992 Chinese patents; individual metrics report their effective sample sizes.

## Current release

Download **[Claim-DER-reproducibility-20261003.tar.xz](Claim-DER-reproducibility-20261003.tar.xz)** and follow the commands below. The archive contains the complete revised materials in one directory; it is an XZ-compressed TAR file. [Detailed reproduction instructions](REPRODUCIBILITY.md) and [the archive checksum](SHA256SUMS.txt) are also available separately.

```sh
tar -xf Claim-DER-reproducibility-20261003.tar.xz
cd claim_der_reproducibility
python verify_release.py
python verify_outputs.py
python rebuild_tables.py --output rebuilt
```

These commands require Python's standard library and make no model API calls. They verify file integrity and frozen text hashes, verify saved output bindings, and reconstruct seven result-table views with 974 value/denominator checks. See `REPRODUCIBILITY.md` for metric dependencies and explicit instructions for new, paid generation runs.

Archive SHA-256:

```text
7f4f55cea5c963ac990c78f1aa2d5e535c89f81d39357eec9df14a5df375129e
```

The archive includes:

- The frozen 2,000 disclosures and reference claim sets, sample IDs, text hashes and length-distribution checks.
- All 24 original output exports: 14 main-method exports and 10 controlled-ablation/draft-only exports.
- Saved per-patent automatic scores, LLM judge scores, protocols and table reconstruction scripts.
- Portable ROUGE, PAR, DAR and numbering/reference checks; baseline and controlled-ablation runners with recorded prompts.
- Compact corrected-ablation stage records and provenance checks. Full private request/checkpoint archives and raw judge response trees are not included.

## Evaluation scope

The main metrics are ROUGE recall, position-aware keyword recall (PAR), auxiliary dependency-aware regression (DAR), and five-dimension Qwen-Max judgments. Supplementary materials include ROUGE precision/F1, BERTScore, Qwen3-Max judgments and explicit numbering/reference checks. DAR is sparse for English and is not an overall dependency-validity measure. LLM judgments are identified as model evaluations; independent human expert ratings are not claimed. Historical execution and training-provenance limits are documented in the release.

## Historical files

The existing `data/`, `src/` and `requirements.txt` are retained as the initial submission snapshot. **Use the current release above for the revised experiments.** The initial snapshot does not contain the new seven-method comparison or controlled ablation and is not the current reproduction entry point. Its legacy metric handling and Examiner short-circuit behavior are superseded by the release, as detailed in `REPRODUCIBILITY.md`.

## Attribution

The English collection is derived from Patent-CR; Patent-LG-zh is the self-collected Chinese collection. Self-Refine and PlanGEN are adapted comparison methods. Source collections and third-party dependencies retain their applicable licenses; this release does not assign new rights to those materials.
