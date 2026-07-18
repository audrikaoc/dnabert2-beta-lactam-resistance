# DNABERT-2 Embeddings for Beta-Lactam Resistance Classification in *E. coli*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21424241.svg)](https://doi.org/10.5281/zenodo.21424241)

This repository contains the code and processed public data for a research project evaluating whether frozen, pre-trained genomic language model embeddings can distinguish beta-lactam resistance-associated *Escherichia coli* sequences from non-resistance-associated control sequences.

The project uses `zhihan1996/DNABERT-2-117M` as a zero-shot feature extractor. DNA sequences are embedded into fixed 768-dimensional vectors, then a Random Forest classifier is trained on those embeddings.

## Research Question

Can pre-trained multi-species genomic language model embeddings encode enough biological signal to differentiate beta-lactam resistant and non-resistant *E. coli* sequences without fine-tuning the language model?

## Pipeline Overview

1. Parse public FASTA records and build a labeled sequence table.
2. Clean DNA sequences to canonical `A/C/G/T` characters.
3. Label beta-lactamase-associated records as resistant (`1`) and control records as non-resistance-associated (`0`).
4. Extract frozen DNABERT-2 embeddings using mean pooling.
5. Train a Random Forest classifier on the embedding matrix.
6. Evaluate with both a naive random split and a leakage-aware near-duplicate grouped split.
7. Save metrics, confusion matrices, and ROC-AUC plots.

## Repository Contents

```text
data_loader.py                    # FASTA parsing, sequence cleaning, labeling, optional BLAST verification
embedder.py                       # DNABERT-2 embedding extraction and .npy export
classifier.py                     # Random Forest training, evaluation, plots, JSON metrics

labeled_sequences.csv             # Final labeled dataset used for embeddings/classification
labeled_sequences_combined.csv    # Intermediate/alternate processed dataset
labeled_sequences_all_controls.csv # Intermediate/alternate processed dataset
nucleotide_fasta_protein_homolog_model.fasta

embeddings.npy                    # 432 x 768 DNABERT-2 embedding matrix
test_metrics.json                 # Leakage-aware grouped-split metrics
test_metrics_random_split.json    # Original naive random-split metrics

classifier_plots/                 # Leakage-aware confusion matrix and ROC curve
classifier_plots_random_split/    # Naive random-split confusion matrix and ROC curve
```

## Data Notes

The data used in this project is publicly available genomic sequence data. The processed control group should be interpreted carefully: controls are computationally defined non-resistance-associated *E. coli* sequences, with a subset verified through BLAST/reference checks. They should not be described as experimentally confirmed susceptible isolates unless additional phenotype metadata is added.

Final dataset:

```text
432 total sequences
232 resistant sequences
200 control sequences
```

Final embedding matrix:

```text
embeddings.npy shape: (432, 768)
```

## Installation

Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DNABERT-2 is loaded from Hugging Face with `trust_remote_code=True`. The first embedding run may download model files.

## Usage

### 1. Parse FASTA Data

Example:

```bash
python data_loader.py nucleotide_fasta_protein_homolog_model.fasta --output labeled_sequences.csv
```

Optional BLAST verification can be supplied with `--blast-reference` if local BLAST+ tools and reference FASTA files are available.

### 2. Generate DNABERT-2 Embeddings

For laptop-safe CPU inference:

```bash
python embedder.py labeled_sequences.csv --output embeddings.npy --batch-size 1 --device cpu
```

This produces one 768-dimensional vector per sequence.

### 3. Train and Evaluate the Classifier

Leakage-aware grouped split, recommended:

```bash
python classifier.py --embeddings embeddings.npy --labels labeled_sequences.csv --output-dir classifier_plots --metrics-output test_metrics.json
```

Naive random split, useful only as a baseline comparison:

```bash
python classifier.py --embeddings embeddings.npy --labels labeled_sequences.csv --output-dir classifier_plots_random_split --metrics-output test_metrics_random_split.json --no-group-near-duplicates
```

## Model and Evaluation Details

Embedding model:

```text
zhihan1996/DNABERT-2-117M
```

Embedding strategy:

```text
Frozen model inference
Mean pooling over token embeddings
768-dimensional final sequence vectors
```

Classifier:

```text
RandomForestClassifier
300 trees
class_weight="balanced"
random_state=42
```

Primary split:

```text
Approximately 80% train / 20% test
Near-duplicate grouped split enabled by default
15-mer Jaccard similarity threshold: 0.85
```

The near-duplicate audit found:

```text
126 near-duplicate groups
Largest group size: 92 sequences
4,788 near-duplicate pairs
0 mixed-label groups
```

## Results

### Leakage-Aware Grouped Split

These are the recommended headline results because near-duplicate sequence families were kept entirely in train or test.

```text
Accuracy: 0.9659
ROC-AUC: 0.9749
Macro F1: 0.9657
Resistant-class precision: 1.0000
Resistant-class recall: 0.9388
Resistant-class F1: 0.9684
```

### Naive Random Split

These results are included for comparison and are likely inflated by sequence similarity leakage.

```text
Accuracy: 0.9770
ROC-AUC: 0.9995
Macro F1: 0.9769
Resistant-class F1: 0.9787
```

## Interpretation

The leakage-aware results support the hypothesis that frozen DNABERT-2 embeddings contain biologically meaningful signal associated with beta-lactam resistance in *E. coli*. Because DNABERT-2 was not fine-tuned, the downstream Random Forest classifier tests whether useful resistance-related information is already present in the pre-trained embedding space.

## Limitations

- The task is binary: resistant-associated vs control/non-resistance-associated.
- The control group is computationally defined and not fully phenotype-confirmed.
- The dataset contains only 432 sequences from one species.
- Resistance levels such as MIC values were not modeled.
- External validation on independently collected *E. coli* genomes would strengthen the findings.

## Citation Notes

If using this repository or methodology, cite DNABERT-2 and the public sequence resources used to construct the dataset. Relevant software includes PyTorch, Hugging Face Transformers, scikit-learn, Biopython, NumPy, pandas, and matplotlib.

If you use this repository, please cite:

Chakrobartty, A. (2026). *DNABERT2 beta-lactam resistance analysis* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21424241
