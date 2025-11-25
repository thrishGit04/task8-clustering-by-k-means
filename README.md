# ⭐ Task 8 — Clustering with K-Means

This repository contains **Task 8** of my AIML Internship project.
The goal is to perform **unsupervised learning** using **K-Means clustering** on the Housing dataset, including data preprocessing, PCA-based dimensionality reduction, elbow method for optimal K selection, cluster visualization, silhouette score evaluation, and saving all results in a structured output folder.

---

## 📚 Table of Contents

1. [Repository Structure](#-repository-structure)
2. [Objective](#-objective)
3. [Data Preprocessing](#-data-preprocessing-steps)
4. [Modeling Workflow](#-modeling-workflow-clustering_kmeanspy)
5. [Visualizations](#-generated-visualizations)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [How to Run](#-how-to-run-the-project)
8. [Dataset](#-dataset)
9. [Author](#-author)

---

## 📁 Repository Structure

This structure matches your actual folder layout:

```text
├── outputs/
│   ├── cluster_centers.csv               # Final K-Means cluster centroids
│   ├── cluster_visualization.png         # 2D PCA-based cluster plot
│   ├── elbow_curve.png                   # Elbow method to choose optimal K
│   ├── empty                             # auto-generated placeholder
│   ├── kmeans_model.pkl                  # Saved K-Means model
│   ├── pca_2d_data.csv                   # 2D PCA transformed dataset
│   ├── processed_numeric_data.csv         # Cleaned numeric-only dataset
│   ├── raw_dataset.csv                   # Raw input dataset backup
│   ├── scaler.pkl                        # StandardScaler object
│   ├── silhouette_score.txt              # Silhouette score result
│
├── Housing.csv                            # Original dataset (uploaded)
├── clustering_kmeans.py                   # Complete K-Means pipeline script
├── README.md                              # Documentation (this file)
```

---

## 🎯 Objective

This task focuses on implementing and understanding:

* **Unsupervised learning** using **K-Means clustering**
* **Feature scaling** with StandardScaler
* **Dimensionality reduction** using PCA (for visualization)
* **Optimal K detection** using the Elbow Method
* **Cluster visualization** in 2D
* **Cluster quality evaluation** using the Silhouette Score
* **Saving all outputs** for reproducibility and analysis

---

## 🧹 Data Preprocessing Steps

The following steps are applied to `Housing.csv`:

1. Load the raw dataset and store a clean backup in `outputs/raw_dataset.csv`.
2. Select **numeric columns only** (K-Means requires numeric features).
3. Drop missing values from numeric columns.
4. Standardize the features using **StandardScaler**.
5. Save the processed dataset as `processed_numeric_data.csv`.

This creates a consistent input matrix for clustering.

---

## 🤖 Modeling Workflow (`clustering_kmeans.py`)

This script implements a full K-Means clustering pipeline:

### 1️⃣ Feature Scaling

All numeric data is standardized before clustering.

### 2️⃣ PCA (2D Projection)

The dataset is reduced to two principal components for visualization.

Output:

* `pca_2d_data.csv`

### 3️⃣ Elbow Method

K values from 1 to 10 are evaluated to identify the optimal number of clusters.

Output:

* `elbow_curve.png`

### 4️⃣ K-Means Clustering

Runs K-Means with K=3 (or any K you choose after elbow inspection).
Stores:

* Cluster labels
* Cluster centers
* Fitted model (`kmeans_model.pkl`)

Outputs:

* `cluster_centers.csv`
* `kmeans_model.pkl`

### 5️⃣ Silhouette Score

Measures cluster cohesion and separation.

Output:

* `silhouette_score.txt`

### 6️⃣ Final Visualization

Creates a 2D scatter plot with PCA components and color-coded clusters.

Output:

* `cluster_visualization.png`

---

## 📊 Generated Visualizations

All plots are saved inside the `outputs/` directory.

### ✔ Elbow Curve

**File:** `elbow_curve.png`
Helps identify the optimal K by analyzing inertia drop-off.

### ✔ Cluster Visualization

**File:** `cluster_visualization.png`
Displays K-Means clusters in PCA-reduced 2D space.

---

## 🧪 Evaluation Metrics

The clustering quality is evaluated using:

* **Silhouette Score** — measures how well-separated clusters are.
* **Cluster Centers** — numeric representation of each cluster's centroid.

These metrics give insight into cluster cohesion and interpretability.

---

## 🚀 How to Run the Project

### Option 1 — Google Colab (Recommended)

1. Upload:

   * `Housing.csv`
   * `clustering_kmeans.py`

2. Run:

```python
!python clustering_kmeans.py
```

3. All results will appear inside the **outputs/** folder, including plots, PCA files, cluster centers, silhouette score, and the saved K-Means model.

---

### Option 2 — Local Machine

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Run the script:

```bash
python clustering_kmeans.py
```

Outputs will be automatically generated in the `outputs/` directory.

---

## 📝 Dataset

**Housing Dataset (Unsupervised Task)**

A numeric dataset containing various housing-related features.
Used for exploring **customer segmentation / pattern discovery** using clustering.

---

## ✨ Author

**Thrishool M S**

AIML Internship — *Task 8: K-Means Clustering*
