# Chest X-Ray Images(Pneumonia) - Progress Report

**Team Members:** Jesus Villa, Derek Shin, Momoka Aung, Gauri Chahal, Joshua Encinas
**Project Lead:** (Project leader name here)
**Date:** April 19, 2026

---

## Project Overview

Chest X-Ray Pneumonia detection aims to develop machine learning models that classify chest X-ray images as Normal or Pneumonia using a publicly available medical imaging dataset. Our project employs two modeling approaches: (1) traditional machine learning methods such as Support Vector Machines (SVMs), and (2) deep learning models using Convolutional Neural Networks (CNNs) for image-based classification. By analyzing pixel-level patterns and visual indicators associated with lung abnormalities, we aim to identify key features that contribute to accurate pneumonia detection. This project demonstrates the effectiveness of machine learning techniques in medical image classification and provides insight into model performance, optimization strategies, and real-world healthcare applications.

---

## Team Member Responsibilities
(These are sample roles but you can choose them if you like)
- **(Project leader name here) (Project Lead):** Project coordination, model selection strategy, baseline CNN implementation, ensures integration of all components
- **(team member name here):** Data preprocessing (image resizing, normalization), dataset organization, data augmentation implementation
- **(team member name here):** Exploratory data analysis (class distribution, sample inspection), dataset visualization, preprocessing validation
- **(team member name here):** Model improvement and optimization, hyperparameter tuning, implementation of additional models (CNN variations or SVM)
- **(team member name here):** Performance analysis, confusion matrix and metrics evaluation, visualization of results, documentation and report preparation

---

## Data Information

**Source:** Kaggle - Chest X-Ray Images(Pneumonia)
**Format:** JPEG images (5,863 labeled images across training, validation, and test sets) 
**Key Features:** Pixel-based image data representing chest X-rays; visual patterns associated with lung opacity, consolidation, and other pneumonia-related indicators
**Target Variables:** Image classification label (Normal vs. Pneumonia – binary classification) 
**Data Quality:** Labeled dataset reviewed by medical experts; contains class imbalance with a higher proportion of pneumonia cases; images require preprocessing including resizing, normalization, and augmentation to improve model generalization

---


## Project Status and Progress

We have established the project scope and selected the Chest X-Ray Pneumonia dataset, confirming its structure and suitability for binary classification tasks. Initial analysis identified key challenges, including class imbalance and the need for image preprocessing such as resizing, normalization, and augmentation. The team has defined a clear modeling approach involving both traditional machine learning methods and deep learning techniques, with plans to implement baseline models followed by improved models for performance comparison. The development pipeline is currently being set up, including dataset organization and preprocessing workflows to ensure consistent input for model training. Our next phase includes implementing baseline models (SVM and/or CNN), applying data augmentation and class balancing strategies, evaluating performance using accuracy and confusion matrix metrics, and refining models through tuning and optimization. The project foundation has been established with a structured plan for execution, positioning us to efficiently complete model development and evaluation within the remaining timeline.
