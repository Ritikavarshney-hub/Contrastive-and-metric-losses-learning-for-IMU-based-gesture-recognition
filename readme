# IMU-Based Gesture Recognition Using Deep Representation Learning

## Overview

This project implements an **IMU-based gesture recognition system** using deep learning, with a primary focus on **studying the effect of different loss functions on representation learning**. Instead of relying only on classification accuracy, the project analyzes how various loss functions shape the **embedding space** learned from inertial sensor data.

The system processes **raw IMU signals (accelerometer + gyroscope)** using a **fixed CNN-based architecture**, while experimenting with multiple loss functions under identical training conditions. This controlled setup enables a fair and meaningful comparison of learning objectives.

---

## Motivation

Gesture recognition using IMU data is challenging due to:

- Sensor noise and drift
- Large intra-class variation across users
- Overlapping motion patterns between gestures
- Visual ambiguity between characters (e.g., `O`, `o`, `0`)

Traditional classification losses optimize decision boundaries but do not explicitly enforce structured embeddings. This work explores whether **embedding-based loss functions** can learn more compact, separable, and semantically meaningful representations for IMU gestures.

---

## Dataset Description

### Sensor Modality

- **6-axis IMU data**
  - 3-axis Accelerometer (X, Y, Z)
  - 3-axis Gyroscope (X, Y, Z)

### Sample Format

Each gesture sample is stored as a time-series array:


### Class Configurations

| Configuration | Description |
|---------------|------------|
| 62 Classes | Full alphanumeric character set |
| 47 Classes | Merged classes where visually similar characters are grouped (e.g., `O/o/0`, `C/c`) |

### Dataset Structure

- `X_*.npy`: IMU signals of shape `(N, T, 6)`
- `y_*.npy`: One-hot encoded labels
- Multiple predefined **user-independent splits**
- Each split is treated as an independent experiment

---

## Preprocessing Pipeline

The same preprocessing is applied across all experiments:

1. Channel-wise normalization of IMU signals
2. Each channel normalized independently per sample
3. No handcrafted features
4. Raw time-series used as model input

This ensures that observed differences are due only to the loss function and not preprocessing variations.

---

## Model Architecture

A **fixed CNN-based encoder** is used across all experiments.

### Encoder

- 1D convolution layers for temporal feature extraction
- ReLU activations
- Max pooling for temporal downsampling
- Global average pooling
- Fully connected layer producing a fixed-dimensional embedding

### Classifier Head

- Linear layer on top of embeddings
- Used only during supervised training

**The architecture remains unchanged for all loss functions.**

---

## Loss Functions Evaluated

The core contribution of this project is a **systematic comparison of multiple loss functions**:

### 1. Cross-Entropy Loss (Baseline)
- Standard classification objective
- Optimizes class prediction accuracy
- Does not explicitly structure the embedding space

### 2. Supervised Contrastive Loss (SCL)
- Encourages global clustering of same-class samples
- Maximizes inter-class separation
- Produces highly structured embeddings
- Requires sufficiently diverse batches

### 3. Triplet Loss
- Enforces relative distance constraints between samples
- Sensitive to mining strategy
- Shows variability across splits

### 4. Margin-Based Embedding Loss
- Enforces explicit margins between classes
- Improves boundary separation
- Requires careful optimization

All loss functions are trained using the **same model, optimizer, batch size, learning rate, and number of epochs**.

---

## Training Strategy

- Framework: PyTorch
- Optimizer: Adam
- Fixed learning rate and batch size
- Identical training settings across all experiments
- User-independent evaluation protocol
- No architectural changes between experiments

This guarantees reproducibility and unbiased comparison.

---

## Embedding Visualization

To analyze representation quality, learned embeddings are visualized using **UMAP**.

### Visualization Process

- Embeddings extracted from trained encoder
- Reduced to 2D using UMAP with cosine distance
- Points colored by gesture class
- Cluster centers annotated with gesture characters

### Observations

- Cross-entropy loss produces weaker cluster separation
- Triplet loss shows unstable clusters due to mining sensitivity
- Margin-based loss improves separation but converges slowly
- **Supervised contrastive loss yields the most compact and well-separated clusters**

For the 47-class setup, visually similar characters (e.g., `O`, `o`, `0`) naturally overlap, reflecting real-world ambiguity.

---

## Saved Outputs

The following artifacts are generated:

- Learned embeddings (`.npy`)
- Corresponding labels (`.npy`)
- UMAP plots for qualitative analysis

These allow further analysis without retraining.

---

## Key Contributions

- IMU-based gesture recognition using raw sensor data
- Fair comparison of multiple loss functions
- Controlled experimental design
- Embedding-space visualization and analysis
- Support for merged gesture classes

---

## Conclusion

This work demonstrates that **loss function choice plays a crucial role in representation learning for IMU-based gesture recognition**. While cross-entropy serves as a strong baseline, embedding-based losses—especially **supervised contrastive loss**—produce more structured and semantically meaningful embeddings.

The study highlights the importance of analyzing learned representations, not just classification accuracy, for noisy time-series data such as IMU signals.

---

## Future Work

- Cross-session and cross-device generalization
- Lightweight models for real-time deployment
- Self-supervised or pretraining-based approaches
- Quantitative embedding metrics (silhouette score, Davies–Bouldin index)
