# Who Gets Flagged? Auditing Four Off-the-Shelf Toxicity Classifiers Across Identity Groups, Thresholds, and Distribution Shift

CS74 Final Project (Track 2: ML audit), Dartmouth, Fall 2026.

This project audits four pretrained toxicity classifiers from Hugging Face and a TF-IDF + logistic regression baseline. The question is whether their errors, especially false positives, concentrate on comments that mention certain identity groups. The experiments look at overall performance, threshold choice, identity subgroups, distribution shift (HateCheck), and identity-word swaps.

## Repository structure

```
01_load_data.ipynb       download data, leakage check, build the audit splits
02_score_models.ipynb    score all texts with the 4 pretrained models (GPU)
03_baseline.ipynb        train/tune the TF-IDF + logistic regression baseline and score the same texts
04_experiments.ipynb     Experiments A–E; writes figures/ and results/
data/processed/          audit splits and HateCheck (see Data)
scores/                  model scores used by 04, plus model/baseline settings
results/                 result tables (CSV/JSON)
figures/                 figures used in the report and slides
requirements.txt
```

## Data

| Dataset | Source | License | Use |
|---|---|---|---|
| CivilComments-WILDS (Koh et al., 2021) | [`pangwei/wilds-civilcomments`](https://huggingface.co/datasets/pangwei/wilds-civilcomments) | CC0 | Main audit data, with identity-group labels |
| Civil Comments, Jigsaw/Kaggle splits | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | CC0 | Leakage check only (text matching) |
| HateCheck (Röttger et al., 2021) | [`Paul/hatecheck`](https://huggingface.co/datasets/Paul/hatecheck) | CC BY 4.0 | Distribution-shift test set |

`01_load_data.ipynb` downloads everything automatically and builds:

- **`audit_tune` / `audit_test`** (21,148 / 21,147 comments, 11.1% toxic): WILDS test comments that the pretrained models could not have seen in training. Only comments from the Kaggle 2019 *test* ID range (≥ 7,000,000) are kept, and any whose text also appears in the Kaggle training set is dropped. The 42,295 remaining comments are split 50/50, stratified by label, with random seed 24. Thresholds are tuned on `audit_tune`; all results are reported on `audit_test`.
- **`baseline_train` / `baseline_val`** (WILDS train / val): used only to train and tune the TF-IDF baseline. Not included in the repo because of size; re-created by notebook 01.
- **`hatecheck`**: 3,728 functional test cases, 68.8% hateful.

Labels: a comment is toxic if `toxicity ≥ 0.5` (share of annotators who called it toxic). It mentions an identity group if that group's label is ≥ 0.5 (8 WILDS groups: male, female, LGBTQ, Christian, Muslim, other religions, Black, white).

## Models

| Short name | Hugging Face model | Score used |
|---|---|---|
| Unbiased-RoBERTa | [`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta) | sigmoid of `toxicity` |
| Toxic-BERT | [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) | sigmoid of `toxic` |
| S-NLP RoBERTa | [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | softmax, `toxic` class |
| Cardiff-Offensive | [`cardiffnlp/twitter-roberta-base-offensive`](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive) | softmax, `offensive` class |
| TF-IDF LR | trained in `03_baseline.ipynb` | predicted probability |

## How to reproduce

The notebooks were run on Google Colab. Install the requirements with `pip install -r requirements.txt` (each notebook also installs what it needs in its first cell).

1. Put the project folder on Google Drive and set `PROJECT_DIR` in the setup cell of notebooks 02–04 to its path. Notebook 01 works in `/content` and copies `data/` and `figures/` to Drive in its last cell; change that path too.
2. Run the notebooks in order: **01 → 02 → 03 → 04**.
   - 01: CPU is fine; most of the time is downloading the data.
   - 02: needs a **GPU**. Scoring took about 5 min in total on a Colab GPU (~35–55 s per model per audit split), plus model downloads. Scores are cached per model, so an interrupted run can resume.
   - 03: a few minutes on CPU (each of the 6 grid fits takes 5–25 s).
   - 04: Experiments A–D take seconds. Experiment E re-runs the models on ~750 edited comments and needs a GPU for a fast run; set `RUN_EXP_E = False` to skip it.

**Results only, no GPU:** the model scores are included in `scores/`, so `04_experiments.ipynb` can be run directly to reproduce all tables and figures for Experiments A–D. Experiment E uses its cached scores in `results/expE_swap_scores.parquet`.

## Experiments and main results

| | Experiment | Main result |
|---|---|---|
| A | Overall performance (`audit_test`) | ROC-AUC 0.90–0.96. The TF-IDF baseline (0.92) beats Toxic-BERT and Cardiff-Offensive. |
| B | Threshold sensitivity | Best-F1 thresholds range from 0.18 to 0.68. Tuning for F1 can widen the false-positive-rate gap between groups. |
| C | Identity subgroups | Every model flags non-toxic comments that mention Black, white, or LGBTQ people more often than comments with no identity (e.g. Unbiased-RoBERTa 14.8% vs. 5.1% for Black). |
| D | HateCheck (distribution shift) | ROC-AUC drops by 0.24–0.34. Counter-speech and non-hateful profanity are flagged most of the time. |
| E | Identity swap | Swapping "white" → "black" or "straight" → "gay" raises scores; gender swaps barely change them. |

## References

- Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. *Companion Proceedings of WWW 2019*.
- Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. *ICML 2021*.
- Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert, J. (2021). HateCheck: Functional tests for hate speech detection models. *ACL 2021*.
