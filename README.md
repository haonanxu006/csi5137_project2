# M2 Range Query Selectivity – Reproduction

This project reproduces the M2 range query selectivity experiments (Table 5) from:

**A Generic Machine Learning Model for Spatial Query Optimization based on Spatial Embeddings**  
Belussi et al., 2024.

We evaluate both:

1. The authors’ pretrained M2 models
2. Our own re-trained M2 models using the same training sets

---

## Project Structure

configs/ # Configs for training our M2 models
tconfigs/ # Configs for testing pretrained models
pretrained/ # Authors’ pretrained models (download; see README inside)
tset/ # Training datasets for M2 (download; see README inside)

myModel_RQ.py # M2 model definitions (DNN & CNN) adapted from authors' code
run_model_all.py# Evaluation helpers extracted from authors' code
train.py # Training pipeline
test_pretrained.py # Script for testing pretrained models

---

## Usage

### 1. Test pretrained models

python test_pretrained.py --cfg tconfigs/aes1_dnn.json

### 2. Train a model

python train.py --cfg configs/aec2_cnn.json

Datasets and pretrained models must be downloaded and renamed before any testing or training (links provided in folder READMEs).

---

## Notes

- We follow the same 80/20 train–test split (seed = 42) used by the authors.
- M2 model architectures are preserved with minimal fixes for compatibility.
- This project focuses **only** on reproducing Table 5 (range query selectivity).
