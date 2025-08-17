# DNN-SemiSupervised-Learning
Source code for regression experiments with semi-supervised learning.

Overview

This repository provides the source code and experiments for analyzing supervised, semi-supervised, and traditional machine learning methods within the context of predictive modeling. The motivation behind this work is to investigate how unlabeled data can be effectively leveraged through semi-supervised learning (SSL) to achieve performance improvements over classical approaches, which often rely solely on limited labeled data.

Folder Structure

Deep_Learning/
Contains five methods implemented using neural networks:

Fully_Supervised.ipynb

Hyperparam_Tuning.ipynb

Label_Propagation.ipynb

Mean_Teacher.ipynb

Pseudo-Labeling.ipynb

Traditional_ML/
Contains three classical machine learning baselines:

svr.py (Support Vector Regression)

ada.py (AdaBoost)

knn.py (K-Nearest Neighbors)

main/
Contains the proposed 3-Stage SSL method, which integrates supervised pretraining, consistency regularization, and iterative pseudo-label refinement into a unified framework.

Methodology of the Proposed SSL Approach

Our 3-Stage SSL method is designed to combine the strengths of supervised and semi-supervised paradigms. In Stage 1, a supervised model is trained on labeled data to provide a strong initialization. Stage 2 introduces consistency-based regularization to exploit unlabeled data. Stage 3 refines predictions by iterative pseudo-labeling, progressively improving decision boundaries.

This pipeline allows systematic comparisons with both traditional ML methods (in the Traditional_ML folder) and deep learning baselines (in the Deep_Learning folder). All methods are evaluated under a unified protocol, ensuring fairness and reproducibility.
