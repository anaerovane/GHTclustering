# scGHT

This repository provides an enhanced implementation of the scGHT (Single-cell Gomory-Hu Tree) framework for single-cell data clustering optimization. The framework includes two main components: **global optimization** (all) and **single-cluster optimization** (part). The goal is to refine cluster boundaries using a combination of graph-based methods, GAE/VGAE embeddings, and attention mechanisms.

## Features

- **Global Optimization**: Refine cluster boundaries across the entire dataset using Gomory-Hu trees and attention-based node reassignment.
- **Single-Cluster Optimization**: Expand or refine a specific cluster using hyperbolic embeddings and distance thresholds.
- **Hybrid Embedding**: Leverage both node features and graph structure for improved clustering accuracy.
- **GPU Support**: Accelerate GAE/VGAE training using CUDA.

## Installation

```bash
git clone https://github.com/anaerovane/scGHT.git
cd scGHT
conda create -n scGHT python=3.12
conda activate scGHT
pip install -r requirements.txt
```

## Usage

### 1. Initial Clustering Preprocessing (First Step)

Before running the optimization steps, you must first run the **initial clustering** to generate the clustering results and node features.

Run the following script to perform the initial clustering and save the outputs:

```python
from initial_clustering import run_initial_clustering

run_initial_clustering(
    data_file='FCA_lung_0.01.h5ad',
    subsample_fraction=0.005,
    n_pca_components=50,
    leiden_resolution=1.0,
    n_clusters_kmeans=8,
    n_clusters_spectral=8,
    output_clustering_csv='clustering_results_enhanced.csv',
    output_node_features_csv='node_features_pca.csv'
)
```

### 2. Global Optimization (All)

#### Step 1: Run Gomory-Hu Tree Construction

```python
from ghtree import process_clustering_network_all_enhanced

process_clustering_network_all_enhanced(
    csv_path='clustering_results.csv',
    output_gtree_path="Gtree_refined.graphml",
    output_ttree_path="Ttree_refined.graphml",
    node_features_path="node_features.csv",
    embedding_dim=32,
    epochs=150,
    lr=0.01,
    use_vgae=True,
    output_embedding_path="node_embeddings_all.pt"
)
```

#### Step 2: Process Clustering Optimization

```python
from processall import process_clustering_all_enhanced

process_clustering_all_enhanced(
    T_refined_file="Ttree_refined.graphml",
    clustering_file="clustering_results.csv",
    initial_clustering_method="leiden",
    min_cut_threshold_percentile=30,
    attention_confidence_threshold=0.6,
    embedding_path="node_embeddings_all.pt",
    output_csv_path="final_clustering_results_enhanced.csv"
)
```

### 3. Single-Cluster Optimization (Part)

#### Step 1: Run Merged Gomory-Hu Tree Construction

```python
from ghtree import process_clustering_network_part_enhanced

process_clustering_network_part_enhanced(
    csv_path='clustering_results.csv',
    leiden_col='leiden',
    target_cluster='2',
    output_gtree_path="Gtree_part_refined_c2.graphml",
    output_ttree_path="Ttree_part_refined_c2.graphml",
    node_features_path="node_features.csv",
    embedding_dim=32,
    epochs=100,
    lr=0.01,
    use_vgae=True
)
```

#### Step 2: Process Single-Cluster Expansion

```python
from processpart import process_clustering_part_enhanced

process_clustering_part_enhanced(
    graphml_path="Ttree_part_refined_c2.graphml",
    csv_path='clustering_results.csv',
    leiden_col='leiden',
    target_cluster='2',
    hyperbolic_distance_threshold=1.5,
    new_col_name='leiden_expanded_2',
    output_path='final_clustering_results_part_expanded_c2.csv',
    embedding_dim=10
)
```

## Example Workflows

### Example All Workflow

```python
from exampleall import run_example_all

run_example_all(
    data_file='FCA_lung_0.01.h5ad',
    subsample_fraction=0.005,
    n_pca_components=50,
    leiden_resolution=1.0,
    embedding_dim=32,
    gae_epochs=150,
    gae_lr=0.01,
    use_vgae=True
)
```

### Example Part Workflow

```python
from examplepart import run_example_part

run_example_part(
    clustering_csv='clustering_results_enhanced.csv',
    node_features_csv='node_features_pca.csv',
    target_col='leiden',
    target_cluster_id='2',
    hyperbolic_dist_thresh=1.5,
    new_cluster_col_name='leiden_expanded_2',
    embedding_dim_part=10
)
```

## Result Shown

After running the optimization workflows, the final clustering results are saved to CSV files. You can visualize the results using tools like `scanpy` or `matplotlib`.

## Notes

- We conducted the tests on both Windows CPU systems and Ubuntu 22.04LTS GPU systems (with an Nvidia Tesla M40 GPU VRAM12G)
- We are still making significant modifications and optimizations to the project

## Contact

For questions or issues, please contact:
- Wuzheng Dong (2210454@mail.nankai.edu.cn)
- Yifan Luo
- Weiyi Chen
