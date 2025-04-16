import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
# Optional: Use geoopt for more sophisticated hyperbolic operations if needed
try:
    import geoopt
    import geoopt.manifolds.poincare.math as pmath
    use_geoopt = True
except ImportError:
    print("Warning: geoopt library not found. Using basic numpy implementation for hyperbolic distance.")
    use_geoopt = False
use_geoopt = False # Keep it simple for now

# --- Basic Hyperbolic Functions (Poincare Ball Model) ---
# Based on https://github.com/facebookresearch/poincare-embeddings

def project_to_poincare_ball(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    epsilon = 1e-9
    norms[norms < epsilon] = epsilon
    poincare_embeddings = embeddings / (1 + norms)
    return poincare_embeddings

def poincare_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # Compute the squared Euclidean distances
    diff = x - y
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    # Compute the Poincaré distance using the formula
    dist = np.arccosh(1 + 2 * np.dot(diff, diff) / ((1 - norm_x**2) * (1 - norm_y**2)))

    return dist

def embed_graph_hyperbolic(graph, embedding_dim=10, walks_per_node=10, walk_length=40, p=1, q=1):
    """
    Placeholder: Uses node2vec to get initial Euclidean embeddings, then projects.
    Requires 'node2vec' library: pip install node2vec
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        print("Error: node2vec library not found. Run 'pip install node2vec'")
        print("Cannot perform hyperbolic embedding placeholder. Returning None.")
        return None, None

    print("Running Node2Vec as a placeholder for hyperbolic embedding...")

    # 清洗图中每条边的权重，确保都是有限值
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        if not np.isfinite(w):
            print(f"[警告] 边 ({u}, {v}) 权重非法：{w}，已设为 1.0")
            w = 1.0
        graph[u][v]['weight'] = w  # 将当前边的权重更新为有限的值

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(graph, dimensions=embedding_dim, walk_length=walk_length,
                        num_walks=walks_per_node, p=p, q=q, workers=4, quiet=True)

    # 训练 Node2Vec 模型
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 获取节点嵌入（需要确保图中节点ID为字符串）
    euclidean_embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])

    # 投影到 Poincare 球面（简单的投影）
    hyperbolic_embeddings = project_to_poincare_ball(euclidean_embeddings)
    print("Node2Vec embeddings projected to Poincare ball.")

    node_list = list(graph.nodes())
    embedding_map = {node: hyperbolic_embeddings[i] for i, node in enumerate(node_list)}

    return embedding_map, node_list



def process_clustering_part_enhanced(graphml_path, csv_path, leiden_col, target_cluster,
                                     # start_node is implicitly the supernode ID
                                     hyperbolic_distance_threshold, # Use distance threshold
                                     new_col_name, output_path,
                                     embedding_dim=10): # Low dim often used for hyperbolic
    """
    Optimizes a single cluster using the refined merged Gomory-Hu tree
    and hyperbolic distance.
    """
    print(f"Loading refined merged Gomory-Hu tree from {graphml_path}")
    T_merged_refined_orig_ids = nx.read_graphml(graphml_path)

    print(f"Loading clustering data from {csv_path}")
    clustering_df = pd.read_csv(csv_path, index_col=0)
    clustering_df.index = clustering_df.index.astype(str) # Ensure string index

    print("Original clustering data head:")
    print(clustering_df.head())

    if leiden_col not in clustering_df.columns:
        raise ValueError(f"Column '{leiden_col}' not found.")
    clustering_df[leiden_col] = clustering_df[leiden_col].astype(str) # Ensure target col is string
    target_cluster_str = str(target_cluster)

    # Identify the supernode ID (assuming the format used in ghtree_enhanced)
    super_node_id = f"supernode_{target_cluster_str}"
    if super_node_id not in T_merged_refined_orig_ids.nodes():
        raise ValueError(f"Supernode '{super_node_id}' not found in the loaded graphml tree.")

    print(f"Target cluster: {target_cluster_str}, Supernode ID: {super_node_id}")

    # --- Embed the Merged GHT into Hyperbolic Space ---
    # Use the placeholder node2vec + projection for now
    # Ensure the graph passed to node2vec has string node IDs if required by the library

    


    T_embed_graph = nx.DiGraph(T_merged_refined_orig_ids) # Node2vec often prefers directed? Check lib docs. Or use original T'.
    
    
    # Ensure nodes are strings for node2vec compatibility
    nx.relabel_nodes(T_embed_graph, {node: str(node) for node in T_embed_graph.nodes()}, copy=False)

    hyperbolic_embedding_map, embedded_node_list = embed_graph_hyperbolic(T_embed_graph, embedding_dim=embedding_dim)

    if hyperbolic_embedding_map is None:
         print("Failed to generate embeddings. Aborting part enhancement.")
         return

    # --- Identify Nodes Close to Supernode in Hyperbolic Space ---
    added_nodes_orig_ids = set() # Store original IDs of nodes to add
    # Get the embedding of the supernode (ensure it was embedded)
    super_node_emb_str = str(super_node_id) # Ensure string key if needed
    if super_node_emb_str not in hyperbolic_embedding_map:
         print(f"Warning: Supernode '{super_node_id}' not found in embedding map. Skipping distance calculation.")
    else:
         supernode_embedding = hyperbolic_embedding_map[super_node_emb_str]

         print(f"\nCalculating hyperbolic distances from supernode (Threshold: {hyperbolic_distance_threshold:.4f})...")
         # Iterate through all *other* nodes in the embedding map
         for node_str, node_embedding in tqdm(hyperbolic_embedding_map.items()):
            if node_str == super_node_emb_str:
                continue # Skip self-comparison

            # Calculate hyperbolic distance
            distance = poincare_distance(supernode_embedding, node_embedding)

            # Check if distance is below threshold
            if distance < hyperbolic_distance_threshold:
                # Map node_str back to original ID (might be integer or string depending on graph loading)
                # Find original ID corresponding to node_str (could be complex if relabeling happened)
                # Assuming node_str IS the original ID (or supernode ID) here
                original_node_id = node_str # Adjust if relabeling was complex

                # Add the original node ID (not the supernode itself)
                if original_node_id != super_node_id:
                    added_nodes_orig_ids.add(original_node_id)

    print(f"\nNodes identified for addition (within hyperbolic distance {hyperbolic_distance_threshold:.4f}): {len(added_nodes_orig_ids)}")
    # print("Nodes to add:", added_nodes_orig_ids) # Can be very long

    # --- Update Clustering Results ---
    print("Updating clustering results...")
    # Initialize the new column with the original leiden assignments
    clustering_df[new_col_name] = clustering_df[leiden_col]

    # Find valid nodes that exist in the original dataframe index
    valid_nodes_to_update = [node_id for node_id in added_nodes_orig_ids if node_id in clustering_df.index]

    if valid_nodes_to_update:
        print(f"Assigning {len(valid_nodes_to_update)} valid nodes to cluster {target_cluster_str}.")
        # Update the cluster assignment for the identified nodes
        clustering_df.loc[valid_nodes_to_update, new_col_name] = target_cluster_str
    else:
        print("No valid nodes found to add to the target cluster based on hyperbolic distance.")


    print("\nClustering results after update (first 5 rows):")
    print(clustering_df.head())

    # Save the updated clustering results
    clustering_df.to_csv(output_path)
    print(f"\nEnhanced clustering results for part optimization saved to '{output_path}'")