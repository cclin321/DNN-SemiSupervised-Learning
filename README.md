# DNN-SemiSupervised-Learning
Source code for regression experiments with semi-supervised learning.

## Overview
This repository provides the source code and experiments for analyzing **supervised**, **semi-supervised**, and **traditional machine learning** methods within the context of predictive modeling.  
The motivation behind this work is to investigate how unlabeled data can be effectively leveraged through **semi-supervised learning (SSL)** to achieve performance improvements over classical methods, which typically rely solely on limited labeled data.

---

## Folder Structure

- **Deep_Learning/**  
  Contains five deep learning methods:
  - `Fully_Supervised.ipynb`
  - `Hyperparam_Tuning.ipynb`
  - `Label_Propagation.ipynb`
  - `Mean_Teacher.ipynb`
  - `Pseudo-Labeling.ipynb`

- **Traditional_ML/**  
  Contains three traditional machine learning baselines:
  - `svr.py` (Support Vector Regression)
  - `ada.py` (AdaBoost)
  - `knn.py` (K-Nearest Neighbors)

- **main/**  
  Contains the proposed **3-Stage SSL method**, which integrates supervised pretraining, consistency regularization, and iterative pseudo-label refinement.

---

## Methodology of the Proposed SSL Approach
Our **3-Stage SSL method** is designed to combine the strengths of supervised and semi-supervised paradigms:

1. **Stage 1 – Supervised Pretraining**  
   Train a baseline model on labeled data to provide strong initialization.  

2. **Stage 2 – Consistency Regularization**  
   Introduce consistency-based learning to exploit unlabeled data.  

3. **Stage 3 – Iterative Pseudo-Labeling**  
   Refine predictions with progressively updated pseudo-labels, improving decision boundaries.  

This unified pipeline enables systematic comparisons with both **traditional ML methods** (`Traditional_ML/`) and **deep learning baselines** (`Deep_Learning/`), under a consistent evaluation protocol to ensure fairness and reproducibility.

---
