# SpatPTM

## Model Description

A deep learning model SpatPTM is built for predicting cancer-associated post-translational modification. The position-specific scoring matrix (PSSM) is first extracted for each protein, which is refined by graph attention auto-encoder using a protein spatial network yielded by SPOT-Contact-LM. This operation can transmit the evolutionary information of residues in a protein sequence. The submatrix is extracted from the refined PSSM to represent the PTM site, incorporating the information of full protein sequence. Then, the submatrix is processed by dual 1D convolution kernels, residue-attention, multi-head self-attention, and attention pooling, generating the final feature vector of the PTM site. The fully connected layer is adopted to make prediction.

整体结构见 [Figure1.pdf](docs/images/Figure1.pdf)。

## Requirements

```text
python==3.10.20
numpy==2.0.1
pandas==2.3.3
scikit-learn==1.7.2
torch==2.5.1
openpyxl==3.1.5
```

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python main.py --help
```
