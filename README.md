# SpatPTM-GATE

SpatPTM-GATE is a deep-learning framework for predicting cancer-associated
post-translational modification sites. The model combines PSI-BLAST PSSM
features, SPOT-derived protein contact graphs, GATE residue embeddings, and a
multi-scale CNN-Transformer classifier.

```text
Protein sequence
  -> PSSM and contact map
  -> GATE residue embedding
  -> 43-residue PTM window
  -> CNN + Transformer
  -> cancer-associated PTM probability
```

## Requirements

```text
python==3.10.20
numpy==2.0.1
pandas==2.3.3
scikit-learn==1.7.2
torch==2.5.1
openpyxl==3.1.5
```

Feature generation additionally uses PSI-BLAST, SPOT-1D-Single,
SPOT-Contact-Single, ESM-1b, and the GATE implementation from PMiSLocMF.

## Quick start

Install the model environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Show the available commands:

```bash
python main.py --help
```

Reproduce the site-level cross-validation experiment from prepared GATE
windows:

```bash
python main.py evaluate \
  --data-dir /path/to/balanced_data_store \
  --output-dir runs/formal_site_cv \
  --seeds 42 2024 888 --folds 10 --epochs 50 --patience 5 --device cuda
```

The reported experiment gives an average AUC of 0.9145 over 30 validation
folds.

Run prediction with trained model checkpoints:

```bash
python main.py predict \
  --windows /path/to/gate_windows.npy \
  --sites /path/to/window_manifest.csv \
  --model-dir /path/to/model_checkpoints \
  --output predictions.csv
```

For the sequence-to-prediction workflow, edit
`configs/pipeline.example.json` and run `scripts/run_pipeline.py`.

## Data

Processed datasets, pretrained weights, intermediate protein features, and
third-party model files are not included in this repository.
