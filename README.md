# Swallowing Audio Segmentation and Aspiration Classification

This project focuses on the automated analysis of swallowing sounds to aid clinical assessment of aspiration risk. Using audio recordings of swallowing events from participants, the project extracts meaningful acoustic features, segments swallowing and non-swallowing periods, and builds machine learning models to classify aspiration occurrences.

---

## Project Overview

Aspiration during swallowing is a critical clinical concern that can lead to serious health complications such as pneumonia. Traditional assessment methods like videofluoroscopic swallow studies (VFSS) are resource-intensive and not always accessible. This project explores audio-based swallowing analysis as a non-invasive, cost-effective alternative.

Key components include:

-  **Audio segmentation:** Using precise swallow onset and offset labels stored in HDF5 files, the project extracts swallowing segments and non-swallowing gaps from continuous audio recordings.
-  **Gap adjustment:** Swallow segments are extended to include relevant portions of the silence following each swallow, improving feature representation.
-  **Feature extraction:** Spectral Subband Centroid (SSC) features are computed from audio segments and aggregated to characterize swallowing sounds.
-  **Machine learning classification:** Support Vector Machines (SVM) are trained on extracted features to detect aspiration events, with class imbalance handled using SMOTE oversampling.
-  **Evaluation:** Models are validated on separate test datasets to assess classification performance.

---

## Goals

-  Develop a reproducible pipeline for swallowing audio segmentation and feature extraction.
-  Build robust classifiers to identify aspiration from audio features.
-  Provide tools and datasets for further research in audio-based dysphagia assessment.

---

## Repository Contents

-  Audio processing scripts for segment extraction and gap adjustment.
-  Feature extraction and machine learning training code.
-  Sample datasets and label files.
-  Documentation and usage instructions.

---

This project aims to contribute to improved, accessible dysphagia diagnostics through innovative audio analysis and machine learning techniques.


