# 512_2
Classification of radiological data using deep and shallow leaning approach


Multi-Modal Clinical NLP: Active Learning & Continuous Uncertainty Modeling on Radiology Reports
This repository contains an end-to-end medical data science framework designed to classify unstructured radiology reports along a continuous spectrum of diagnostic confidence. By combining domain-specific transformer embeddings, unsupervised topological clustering, and an active learning feedback loop, this pipeline successfully addresses severe class bias and clinical ambiguity in electronic health records (EHR).

Core Architecture Overview
The repository is structured around a dual-stream classification framework designed to evaluate the trade-offs between computational efficiency and semantic depth:

The Shallow Lexical Stream: Utilizes a high-dimensional TF-IDF feature matrix (1,485 unique clinical unigrams and bigrams) mapped to a sparse linear regressor.

The Deep Contextual Stream: Leverages frozen representation vectors extracted from the base pooling layer of Bio_ClinicalBERT, routed through an optimized classification head governed by a 128-node hidden information bottleneck layer and dropout regularization (p=0.2).

Unsupervised Topology & Dataset Curation Flow
Instead of relying on automated pseudo-labeling (which introduces a high noise floor), the 57,807 raw institutional records were projected into a 2D coordinate manifold via UMAP and segmented using HDBSCAN.

[Raw Radiology Corpus] ──> [Bio_ClinicalBERT Embeddings] ──> [UMAP (768D → 2D)]
                                                                    │
[Balanced Target Dataset] <── [Active Learning Sieve] <── [HDBSCAN Clustering]
This unsupervised topology served as a targeted search light to filter out 19,249 decentralized background noise records, allowing efficient human auditing to track rare minority categories (Normal and Unsure). This active learning intervention balanced the dataset, driving a massive expansion in model generalization (boosting the shallow test R 
2
  score from 0.4797 to 0.5840 and macro F 
1
​
 -score from 0.54 to 0.72).

📁 Repository Directory Structure
├── LMoutput/                       # Sub-sections generated for active learning verification
│   ├── 205_normal_bullets_q.json   # Isolated semantic clusters enriched for "Normal" profiles
│   ├── 502_unsure_bullets_q.json   # Isolated semantic clusters enriched for "Unsure" profiles
│   └── 611_abnormal_bullets_q.json # Isolated semantic clusters enriched for "Abnormal" profiles
│
├── saved_models/                   # Serialized checkpoints and model binary stores
│   ├── clinical_bert_run_1.pt      # PyTorch weights for Deep Learning
│   ├── clinical_bert_run_6.pt      # Peak Model (Optimal Test MSE: 0.0827)
│   ├── clinical_rf_regressor.joblib # Baseline Random Forest snapshot weights
│   └── clinical_tfidf_vectorizer.joblib # Serialized dictionary weights
├── old dataset/                   # File for initial run using 148 sample labelled data
│   ├── semi-supervised.ipynb       
│   ├── E_deeptrainer01.ipynb 
│   ├── E_trainer.ipynb   
│   └── combined_testdata1 # old dataset 149 samples 
│   ├── local_query.ipynb   # code for querying local model on expanded dataset to removed most likely unsure and normal candidates
│   └──
├── Assignment2_F copy.pdf          # Final compiled assignment submission report
├── E_deeptrainer01.ipynb           # Deep learning code (Expanded Dataset)
├── E_semi-supervised.ipynb         # Active learning loops, UMAP landscape mapping, and HDBSCAN runs (Expanded)
├── E_trainer.ipynb                 # shallow learning code Expanded Dataset
├── semi-supervised.ipynb           # Baseline unsupervised landscape mapping and early audit metrics
├── xlsxprocessing.ipynb            # Post-processing slider discretization and continuous label mapping
└── README.md                       # Project documentation (This file)
Note: Notebooks prefixed with E_ represent the secondary phase of research utilizing the curated active learning expanded dataset.


🛠️ Usage and Execution Guide
Dependencies & Requirements
Used Metal for apple silicon.

pip install torch transformers scikit-learn umap-learn hdbscan pandas numpy
Running the Active Learning Curation
To execute the density-based cluster extraction and view the latent space distribution:

Run E_semi-supervised.ipynb to initialize the UMAP projections on your text vector files.


Training the Final Classifiers
To reproduce the deep learning results on the balanced, expanded dataset:

Execute E_trainer.ipynb.
