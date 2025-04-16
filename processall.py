import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # Using GAT directly might be complex here
                                      # Let's implement a simpler attention mechanism based on embeddings
from tqdm import tqdm

# --- Simple Attention Mechanism ---
# This is a simplified concept, not a full GAT layer training loop.
# It calculates attention based on pre-trained embeddings.
def calculate_attention_scores(node_idx, neighbor_indices, embeddings, cluster_map, T_refined):
    """
    Calculates attention scores of a node towards its neighboring clusters in T_refined.
    Uses dot product of embeddings as a proxy for attention weight.
    """
    scores = {}
    node_emb = embeddings[node_idx]
    total_score = 0
    epsilon = 1e-9 # for numerical stability

    neighbors_in_T = list(T_refined.neighbors(node_idx)) # Use neighbors in T for cluster context

    for neighbor_idx in neighbors_in_T:
        if neighbor_idx not in cluster_map: continue # Skip if neighbor has no cluster assigned yet

        neighbor_cluster = cluster_map[neighbor_idx]
        neighbor_emb = embeddings[neighbor_idx]

        # Simple attention: dot product (or cosine similarity) scaled by edge weight in T?
        # Using dot product here. Could use exp(dot_product) for softmax-like behavior
        # weight_in_T = T_refined[node_idx][neighbor_idx].get('weight', 1.0) # Min cut value
        # score = np.dot(node_emb, neighbor_emb) * weight_in_T # Incorporate GHT edge weight?

        score = np.dot(node_emb, neighbor_emb)
        exp_score = np.exp(score) # Use exp for softmax-like scaling

        if neighbor_cluster not in scores:
            scores[neighbor_cluster] = 0
        # Sum scores for neighbors belonging to the same cluster
        scores[neighbor_cluster] += exp_score
        total_score += exp_score

    # Normalize scores (like softmax)
    if total_score > epsilon:
        for cluster in scores:
            scores[cluster] /= total_score
    else: # Handle case with no valid neighbors or zero scores
         # Assign uniform low probability or handle as outlier/keep original
         pass # Decision to be made in the main loop

    return scores


def process_clustering_all_enhanced(T_refined_file, clustering_file, initial_clustering_method,
                                    min_cut_threshold_percentile=25, # Use percentile for threshold1
                                    attention_confidence_threshold=0.5, # Min attention score to reassign
                                    embedding_path="node_embeddings_all.pt",
                                    output_csv_path="final_clustering_results_enhanced.csv"):
    """
    Optimizes clustering using the refined Gomory-Hu tree and an attention mechanism.
    """
    print("Loading refined Gomory-Hu tree...")
    T_refined_orig_ids = nx.read_graphml(T_refined_file)
    print("Loading initial clustering data...")
    clustering_df = pd.read_csv(clustering_file, index_col=0)
    clustering_df.index = clustering_df.index.astype(str) # Ensure string index

    # Load node embeddings
    print(f"Loading node embeddings from {embedding_path}...")
    embeddings_tensor = torch.load(embedding_path)
    embeddings = embeddings_tensor.numpy()

    # Map original node IDs to integer indices used during training/embedding
    node_list = sorted(list(T_refined_orig_ids.nodes()))
    node_map = {node: i for i, node in enumerate(node_list)}
    print(f"Node map length: {len(node_map)}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Relabel T to use integer indices for processing with embeddings
    T_refined = nx.relabel_nodes(T_refined_orig_ids, node_map, copy=True)
    print(f"Refined T nodes: {T_refined.number_of_nodes()}, Embeddings shape: {embeddings.shape}")
    if T_refined.number_of_nodes() != embeddings.shape[0]:
        raise ValueError("Mismatch between number of nodes in T_refined and embeddings!")


    # --- Get initial cluster assignments and map ---
    if initial_clustering_method not in clustering_df.columns:
        raise ValueError(f"Initial clustering method '{initial_clustering_method}' not found.")

    # Ensure initial method column is string
    clustering_df[initial_clustering_method] = clustering_df[initial_clustering_method].astype(str)
    # Map initial clusters using original node IDs
    initial_clusters_orig_ids = clustering_df[initial_clustering_method].to_dict()
    # Create mapping using integer indices for processing
    initial_clusters_mapped = {node_map[node_id]: cluster
                               for node_id, cluster in initial_clusters_orig_ids.items() if node_id in node_map}

    unique_clusters = sorted(list(clustering_df[initial_clustering_method].unique()))
    print(f"Found {len(unique_clusters)} unique initial clusters: {unique_clusters}")

    optimized_clusters_mapped = initial_clusters_mapped.copy() # Work with mapped IDs
    changed_cells_mapped = []
    discrete_cells_mapped = []
    processed_nodes = set() # Keep track of nodes whose assignment is finalized in an iteration

    # --- Identify low min-cut edges ---
    edge_weights = [data['weight'] for u, v, data in T_refined.edges(data=True)]
    if not edge_weights:
        print("Warning: No edge weights found in the Gomory-Hu tree. Skipping optimization.")
        min_cut_threshold = 0
    else:
        min_cut_threshold = np.percentile(edge_weights, min_cut_threshold_percentile)
    print(f"Using min-cut threshold (Percentile {min_cut_threshold_percentile}%): {min_cut_threshold:.4f}")

    # Iterate multiple times? Or just once? Let's try one pass for now.
    print("Processing nodes near low min-cut edges...")
    nodes_to_evaluate = set()
    for u, v, data in T_refined.edges(data=True):
         if data.get('weight', float('inf')) < min_cut_threshold:
             # Get clusters of endpoints if available
             u_cluster = initial_clusters_mapped.get(u)
             v_cluster = initial_clusters_mapped.get(v)
             # Consider nodes near the cut if they belong to different initial clusters
             if u_cluster is not None and v_cluster is not None and u_cluster != v_cluster:
                  # Add nodes and their immediate neighbors in T to the evaluation set
                  nodes_to_evaluate.add(u)
                  nodes_to_evaluate.add(v)
                  # Optional: Add neighbors as well for broader context
                  # for neighbor in T_refined.neighbors(u): nodes_to_evaluate.add(neighbor)
                  # for neighbor in T_refined.neighbors(v): nodes_to_evaluate.add(neighbor)

    print(f"Evaluating {len(nodes_to_evaluate)} nodes near low-cut boundaries.")

    for node_idx in tqdm(list(nodes_to_evaluate)):
        if node_idx in processed_nodes: continue
        if node_idx not in initial_clusters_mapped: continue # Skip if node somehow lost mapping

        current_cluster = initial_clusters_mapped[node_idx]

        # Calculate attention scores towards neighboring clusters based on embeddings
        attention_scores = calculate_attention_scores(node_idx, list(T_refined.neighbors(node_idx)), embeddings, initial_clusters_mapped, T_refined)

        if not attention_scores: # Handle nodes with no scored neighbors
            # discrete_cells_mapped.append(node_idx)
            # processed_nodes.add(node_idx)
            # Keep original assignment for now, could mark as discrete later
            continue

        # Find cluster with highest attention
        best_cluster = max(attention_scores, key=attention_scores.get)
        max_attention = attention_scores[best_cluster]

        # Decision logic based on attention
        if max_attention > attention_confidence_threshold and best_cluster != current_cluster:
             # Reassign node if attention is high enough and to a different cluster
            optimized_clusters_mapped[node_idx] = best_cluster
            changed_cells_mapped.append((node_idx, current_cluster, best_cluster))
            processed_nodes.add(node_idx)
            # print(f"Node {node_idx} changed from {current_cluster} to {best_cluster} (Attention: {max_attention:.3f})")

        elif max_attention < (attention_confidence_threshold / 2): # Example: Mark as discrete if max attention is very low
            # Mark as discrete or uncertain (assign a special label?)
            discrete_cells_mapped.append(node_idx)
            optimized_clusters_mapped[node_idx] = f"discrete_{current_cluster}" # Assign a temporary discrete label
            processed_nodes.add(node_idx)
        else:
            # Keep original assignment if attention is not decisive or points to the same cluster
             processed_nodes.add(node_idx) # Mark as processed even if not changed

    # --- Finalize and Save ---
    # Map optimized clusters back to original node IDs
    optimized_clusters_orig_ids = {node_id: optimized_clusters_mapped.get(node_map[node_id], initial_clusters_orig_ids.get(node_id)) # Fallback to initial if error
                                   for node_id in clustering_df.index if node_id in node_map}

    # Update the DataFrame
    clustering_df['optimized_' + initial_clustering_method] = clustering_df.index.map(optimized_clusters_orig_ids)

    # Print summary
    reverse_node_map = {v: k for k, v in node_map.items()}
    print("\n--- Optimization Summary ---")
    print(f"Nodes evaluated: {len(nodes_to_evaluate)}")
    print(f"Nodes changed cluster: {len(changed_cells_mapped)}")
    # for node_idx, old, new in changed_cells_mapped[:20]: # Print first few changes
    #     print(f"  Cell {reverse_node_map[node_idx]} changed from {old} to {new}")

    print(f"Nodes marked as discrete: {len(discrete_cells_mapped)}")
    # for node_idx in discrete_cells_mapped[:20]:
    #      print(f"  Cell {reverse_node_map[node_idx]} marked as discrete (Original: {initial_clusters_mapped.get(node_idx)})")


    final_clustering_results = clustering_df[[initial_clustering_method, 'optimized_' + initial_clustering_method]]
    final_clustering_results.to_csv(output_csv_path, index=True)

    print(f"\nEnhanced final clustering results saved to '{output_csv_path}'.")
    print("Summary of changes (first 5 rows):")
    print(final_clustering_results.head())