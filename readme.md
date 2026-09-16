# CASD

Official implementation of our ICME 2025 paper:

**CASD: Counterfactual Augmentation for Social Bot Detection on Twitter**

Pin Xu, Fangfang Yuan, Yueshan Wang, Diandian Guo, Cong Cao, and Yanbing Liu

**IEEE International Conference on Multimedia and Expo (ICME), 2025**

[[Paper / DOI]](https://doi.org/10.1109/ICME59968.2025.11209322)

---

## Overview

Social bot detection aims to identify automated or malicious accounts by jointly modeling user attributes, textual information, and social interactions. However, graph-based social bot detectors can be sensitive to limited training distributions and may fail to capture informative structural variations.

**CASD** introduces a **counterfactual graph augmentation** framework for social bot detection.

Instead of relying on conventional random graph augmentations, CASD learns counterfactual transformations over both:

* **Graph structure**, by learning edge perturbations.
* **Node attributes**, by learning feature masking patterns.

The generated counterfactual graphs provide challenging augmented samples that complement the original social graph.

For downstream detection, CASD jointly encodes the **original graph** and the **counterfactually augmented graph** using relational graph neural networks, then fuses their representations for bot classification.

The current repository contains experiment pipelines for:

* **TwiBot-20**
* **TwiBot-22**
* **MGTAB**

---

# Supported Datasets

The repository contains experimental pipelines for three social bot detection settings.

| Dataset                    | Relation Types | Counterfactual Subgraphs | Generation Epochs |
| -------------------------- | :------------: | :----------------------: | :---------------: |
| TwiBot-20                  |        2       |            200           |         80        |
| TwiBot-22 (`100w` setting) |        2       |            800           |        200        |
| MGTAB                      |        7       |            10            |         80        |

These values correspond to the default configurations in the current augmentation scripts and can be changed through command-line arguments.

---

# Repository Structure

```text
CASD/
│
├── BackBone/
│   └── rgcn.py
│       └── Relational graph convolution implementation
│
├── CounterFactual/
│   ├── Generator.py
│   │   └── Learnable structural perturbation and feature masking
│   │
│   └── Generation.py
│       └── Counterfactual optimization and augmented graph generation
│
├── FW/
│   ├── BotRGCN.py
│   │   └── BotRGCN encoder
│   │
│   ├── Graph_Split.py
│   │   └── Graph partitioning utilities
│   │
│   └── load_data.py
│       └── Dataset loading and graph conversion utilities
│
├── Loggers/
│   └── Logging utilities
│
├── data/
│   └── Dataset / intermediate data directory
│
├── Way_2_Augmented_Generate_twibot20.py
│   └── Counterfactual generation for TwiBot-20
│
├── Way_2_Augmented_Generate_twibot22_100w.py
│   └── Counterfactual generation for TwiBot-22
│
├── Way_2_Augmented_Generate_mgtab.py
│   └── Counterfactual generation for MGTAB
│
├── Main_twibot20_RGCN_Backbone_Concat_DynCrossEntropy.py
│   └── CASD training and evaluation on TwiBot-20
│
├── Main_twibot22_100w_RGCN_Backbone_Concat_DynCrossEntropy.py
│   └── CASD training and evaluation on TwiBot-22
│
├── Main_mgtab_RGCN_Backbone_Concat_DynCrossEntropy.py
│   └── CASD training and evaluation on MGTAB
│
├── Convert_HomoToHetero.py
│   └── Graph conversion utility
│
├── Generate_AugmentedSubGraph.py
├── DP_Main.py
│
├── environment.yml
└── readme.md
```

---

# Requirements

The complete experimental environment is provided in:

```text
environment.yml
```

The main dependencies include:

```text
Python 3.10
PyTorch 2.3.0 + CUDA 12.1
PyTorch Geometric 2.5.3
torch-cluster 1.6.3
torch-scatter 2.1.2
torch-sparse 0.6.18
scikit-learn 1.5.1
Transformers 4.47.1
```

Create the environment using:

```bash
conda env create -f environment.yml
```

Then activate the corresponding Conda environment.

> The provided `environment.yml` was exported from the original experimental environment and may contain machine-specific package or prefix information. Adjust it if necessary for your local system.

---

# Data Preparation

The implementation expects graph datasets to be stored as preprocessed PyTorch tensors.

For example, the TwiBot-20 loader uses files corresponding to:

```text
edge_index.pt
edge_type.pt
num_properties_tensor.pt
tweets_tensor.pt
cat_properties_tensor.pt
des_tensor.pt
label.pt
```

These files represent:

```text
edge_index.pt
    Graph connectivity

edge_type.pt
    Social relation types

num_properties_tensor.pt
    Numerical user properties

cat_properties_tensor.pt
    Categorical user properties

des_tensor.pt
    User-description representations

tweets_tensor.pt
    Tweet representations

label.pt
    Human / bot labels
```

Dataset preprocessing and storage formats differ slightly between TwiBot-20, TwiBot-22, and MGTAB.

Please update the dataset paths in the corresponding scripts before running the experiments.

---

# Running CASD

The experiments consist of two major stages.

## Stage 1: Generate Counterfactual Graphs

### TwiBot-20

```bash
python Way_2_Augmented_Generate_twibot20.py
```

Default counterfactual generation configuration:

```text
GNN              : GIN
GNN layers       : 2
Subgraphs        : 200
Generation lr    : 1e-4
Generation epochs: 80
Gamma            : 0.3
Relation types   : 2
```

---

### TwiBot-22

```bash
python Way_2_Augmented_Generate_twibot22_100w.py
```

The current script uses the `100w` TwiBot-22 experimental setting.

---

### MGTAB

```bash
python Way_2_Augmented_Generate_mgtab.py
```

The MGTAB implementation performs counterfactual augmentation over its multiple relation types.

---

## Stage 2: Social Bot Detection

Once the required original and augmented subgraphs have been prepared, run the downstream detector.

### TwiBot-20

```bash
python Main_twibot20_RGCN_Backbone_Concat_DynCrossEntropy.py
```

The default configuration includes:

```text
Number of classes : 2
Relation types    : 2
Input dimension   : 128
Hidden dimension  : 64
Edge dropout      : 0.2
Dropout           : 0.5
Learning rate     : 0.01
Weight decay      : 3e-3
Training epochs   : 80
```

---

### TwiBot-22

```bash
python Main_twibot22_100w_RGCN_Backbone_Concat_DynCrossEntropy.py
```

---

### MGTAB

```bash
python Main_mgtab_RGCN_Backbone_Concat_DynCrossEntropy.py
```

---

# Evaluation

The implementation reports several metrics for social bot detection:

```text
Accuracy
F1 Score
Precision
Recall
Matthews Correlation Coefficient (MCC)
Confusion Matrix
```

Predictions from individual subgraphs are collected and combined to compute the final dataset-level evaluation results.

---

# Implementation Notes

The repository contains the original research implementation used during experimentation.

Some scripts currently contain machine-specific settings inherited from the original experimental environment, including:

```text
/data/...
/data3/data/...
```

Please replace these paths with your own dataset and intermediate-file locations before running the code.

Some legacy scripts may also contain imports such as:

```python
from UnknownFW.BotRGCN import BotRGCN
```

while the public repository stores the corresponding implementation under:

```text
FW/
```

If necessary, update these imports according to the current repository structure, for example:

```python
from FW.BotRGCN import BotRGCN
```

---

# Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{DBLP:conf/icmcs/XuYWGCL25,
  author       = {Pin Xu and
                  Fangfang Yuan and
                  Yueshan Wang and
                  Diandian Guo and
                  Cong Cao and
                  Yanbing Liu},
  title        = {{CASD:} Counterfactual Augmentation for Social Bot Detection on Twitter},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2025,
                  Nantes, France, June 30 - July 4, 2025},
  pages        = {1--6},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICME59968.2025.11209322},
  doi          = {10.1109/ICME59968.2025.11209322},
  timestamp    = {Sat, 01 Aug 2026 10:47:14 +0200},
  biburl       = {https://dblp.org/rec/conf/icmcs/XuYWGCL25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---

# Acknowledgements

This implementation is built with:

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pyg.org/)
* [scikit-learn](https://scikit-learn.org/)

We thank the authors and maintainers of these open-source projects.

---

# Contact

For questions regarding the implementation or paper, please open an issue in this repository.
