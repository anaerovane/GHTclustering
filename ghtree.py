import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- GAE/VGAE Model Definition ---
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Reduced complexity for potentially large graphs
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# --- Helper Function for Training ---
def train_gae_vgae(model, optimizer, data, num_epochs=200, is_vgae=False):
    model.train()
    print("Training GAE/VGAE...")
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index if hasattr(data, 'pos_edge_label_index') else data.edge_index)
        if is_vgae:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        # Optional: Add logging for loss
        # if (epoch + 1) % 20 == 0:
        #     print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')
    model.eval()
    print("Training complete.")
    return model

# --- Main Processing Functions ---
def build_initial_graph(clustering_df):
    """Builds the initial agreement graph based on clustering results."""
    G_init = nx.Graph()
    print("Building initial agreement graph...")
    for col in tqdm(clustering_df.columns):
        # Convert potential non-string cluster labels to strings for consistency
        clustering_df[col] = clustering_df[col].astype(str)
        unique_clusters = clustering_df[col].unique()
        for cluster in unique_clusters:
            # Ensure index is string for networkx compatibility if needed
            cells_in_cluster = clustering_df[clustering_df[col] == cluster].index.astype(str).tolist()
            for i in range(len(cells_in_cluster)):
                for j in range(i + 1, len(cells_in_cluster)):
                    cell1, cell2 = cells_in_cluster[i], cells_in_cluster[j]
                    if G_init.has_edge(cell1, cell2):
                        G_init[cell1][cell2]['weight'] += 1
                    else:
                        G_init.add_edge(cell1, cell2, weight=1)
    print(f"Initial graph nodes: {G_init.number_of_nodes()}, edges: {G_init.number_of_edges()}")
    # Ensure all nodes exist, even if isolated initially based on clustering results
    all_cells = clustering_df.index.astype(str).tolist()
    for cell in all_cells:
        if not G_init.has_node(cell):
            G_init.add_node(cell)

    # Check for disconnected components or isolated nodes
    if not nx.is_connected(G_init):
        print("Warning: Initial graph is not connected. Adding minimal edges to connect components.")
        # A simple strategy: connect the largest component to others via single edges
        components = list(nx.connected_components(G_init))
        if len(components) > 1:
            largest_component = max(components, key=len)
            node_from_largest = next(iter(largest_component))
            for i in range(1, len(components)):
                 node_from_other = next(iter(components[i]))
                 # Add edge with minimal weight, indicating it's artificial
                 G_init.add_edge(node_from_largest, node_from_other, weight=0.1)
            print(f"Graph connected. Edges after connection: {G_init.number_of_edges()}")

    return G_init


def process_clustering_network_all_enhanced(csv_path, output_gtree_path, output_ttree_path,
                                             node_features_path=None, embedding_dim=64, epochs=200, lr=0.01, use_vgae=True,
                                             output_embedding_path="node_embeddings_all.pt"):
    """
    Builds G, computes GHT using GAE/VGAE refined edge weights for the 'all' scenario.
    Optionally uses node features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    clustering_df = pd.read_csv(csv_path, index_col=0)
    # Ensure index is string type for node names
    clustering_df.index = clustering_df.index.astype(str)
    print("Reading clustering data")
    print(clustering_df.head())

    # 1. Build Initial Graph (based on clustering agreement)
    G_initial = build_initial_graph(clustering_df)

    # Ensure node order consistency
    # node_list = sorted(list(G_initial.nodes()))
    # node_map = {node: i for i, node in enumerate(node_list)}
    # nx.relabel_nodes(G_initial, node_map, copy=False) # Relabel in place

    node_list = sorted(list(G_initial.nodes()))
    node_map = {node: i for i, node in enumerate(node_list)}
    reverse_node_map = {v: k for k, v in node_map.items()}
    nx.relabel_nodes(G_initial, node_map, copy=False) 

    # 2. Prepare Data for PyG
    if node_features_path:
        # Load features (e.g., PCA results, ensure order matches node_list)
        features_df = pd.read_csv(node_features_path, index_col=0)
        features_df.index = features_df.index.astype(str)
        # Align features with the node_list order
        features_df = features_df.reindex(node_list)
        # Handle potential missing features (e.g., fill with zeros or mean)
        if features_df.isnull().values.any():
            print("Warning: Missing values found in node features. Filling with zeros.")
            features_df = features_df.fillna(0)
        node_features = torch.tensor(features_df.values, dtype=torch.float).to(device)
        in_channels = node_features.shape[1]
        print(f"Loaded node features with shape: {node_features.shape}")
    else:
        # Use identity matrix or node degrees if no features provided
        print("No node features provided. Using node degrees as features.")
        degrees = torch.tensor([deg for node, deg in G_initial.degree()], dtype=torch.float).unsqueeze(1).to(device)
        node_features = degrees
        in_channels = 1 # Degree is a single feature

    pyg_data = from_networkx(G_initial, group_node_attrs=['weight'] if 'weight' in G_initial.edges(data=True) else None) # Keep initial weights if needed later?
    pyg_data.x = node_features
    pyg_data = pyg_data.to(device) # Ensure edge_index is also on device

    # 3. Initialize and Train GAE/VGAE
    if use_vgae:
        model = VGAE(VariationalGCNEncoder(in_channels, embedding_dim)).to(device)
    else:
        model = GAE(GCNEncoder(in_channels, embedding_dim)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_gae_vgae(model, optimizer, pyg_data, num_epochs=epochs, is_vgae=use_vgae)

# ... (previous code in the function) ...

    # 4. Get Node Embeddings
    # 4. Get Node Embeddings
    with torch.no_grad():
        z = model.encode(pyg_data.x, pyg_data.edge_index)
        if use_vgae and isinstance(z, tuple):  # Only unpack if it's VGAE output
            z = z[0]  # Use mu
    print(z)
    print(f"Z shape: {z.shape}")
    embeddings = z.cpu().numpy()
    torch.save(torch.tensor(embeddings), output_embedding_path)
    print(f"Embeddings shape: {embeddings.shape}")


    print(f"Node embeddings saved to {output_embedding_path}")
    # Add print statement for debugging shape
    print(f"Shape of embeddings BEFORE potential reshape: {embeddings.shape}")

    # --- Start of FIX ---
    # Ensure embeddings is at least 2D
    if embeddings.ndim == 1:
        print(f"Warning: Embeddings array is 1D (shape {embeddings.shape}). Reshaping to 2D.")
        # Logic to handle reshaping based on whether it's one node or dim=1
        # Assuming embedding_dim is defined in the function scope
        if embeddings.shape[0] == embedding_dim: # Shape (embedding_dim,) -> (1, embedding_dim) for single node
            embeddings = embeddings.reshape(1, -1)
        else: # Shape (num_nodes,) -> (num_nodes, 1) for embedding_dim=1
             embeddings = embeddings.reshape(-1, 1)
        print(f"Shape of embeddings AFTER reshape: {embeddings.shape}")
    # --- End of FIX ---


    # 5. Refine Graph Weights based on Embeddings
    print("Refining graph weights using learned embeddings...")
    G_refined = nx.Graph()
    G_refined.add_nodes_from(G_initial.nodes()) # Keep the mapped integer nodes

    num_nodes = embeddings.shape[0]

    # Use cosine similarity for weights (can explore other metrics)
    # Normalize embeddings for cosine similarity calculation

    # --- Start of FIX for Normalization ---
    # Calculate norms, avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Replace zero norms with a small value (epsilon) before division
    epsilon = 1e-9
    norms[norms < epsilon] = epsilon
    norm_embeddings = embeddings / norms
    # --- End of FIX for Normalization ---


    # Efficiently calculate cosine similarity for potential edges (can be memory intensive)
    # Consider using KNN or other methods for large graphs if full pairwise is too slow
    # Here, we iterate through initial edges + potentially add new ones based on similarity threshold
    similarity_threshold = 0.7 # Example threshold to add new edges (adjust as needed)

    processed_edges = set()
    # Use integer node indices (0 to num_nodes-1) for iteration with embeddings
    node_indices = list(range(num_nodes))

    for i in tqdm(node_indices):
        for j in node_indices[i + 1:]: # Correct loop for pairs
            # Calculate similarity between node i and node j using normalized embeddings
            similarity = np.dot(norm_embeddings[i], norm_embeddings[j])

            # Option 2: Use similarity directly as weight, add edges above threshold
            # Ensure similarity is positive and potentially filter weak edges further
            if similarity > 1e-6: # Use similarity as weight if positive
                # Check if the edge existed in the *original mapped* graph OR if similarity is high
                # G_initial uses mapped integer nodes at this stage
                if G_initial.has_edge(i, j) or similarity > similarity_threshold:
                    G_refined.add_edge(i, j, weight=similarity)
                    # processed_edges set might not be strictly necessary here anymore
                    # processed_edges.add(tuple(sorted((i, j))))

    print(f"Refined graph nodes: {G_refined.number_of_nodes()}, edges: {G_refined.number_of_edges()}")
    # Check connectivity again after refinement
    if not nx.is_connected(G_refined):
        print("Warning: Refined graph is not connected. Gomory-Hu tree requires a connected graph.")
        components = list(nx.connected_components(G_refined))
        if len(components) > 1:
            largest_component = max(components, key=len)
            node_from_largest = next(iter(largest_component))
            for k in range(1, len(components)):
                node_from_other = next(iter(components[k]))
                G_refined.add_edge(node_from_largest, node_from_other, weight=1e-5)
            print(f"Graph re-connected. Edges: {G_refined.number_of_edges()}")


    G_refined_orig_ids = nx.relabel_nodes(G_refined, reverse_node_map, copy=True) # Use it here
    print("Saving G_refined (with original IDs)")
    nx.write_graphml(G_refined_orig_ids, output_gtree_path)
    print(f"Refined graph G saved to {output_gtree_path}")


    # --- Section 6: Compute Gomory-Hu Tree on Refined Graph ---
    print("Calculating Gomory-Hu tree on refined graph...")
    # Ensure 'capacity' attribute is set from 'weight' for gomory_hu_tree function
    for u, v, data in G_refined.edges(data=True):
         # Use .get() with a default, ensure capacity exists
         G_refined[u][v]['capacity'] = data.get('weight', 1.0)

    # Handle potential disconnected graph for Gomory-Hu Tree if connectivity check failed/was bypassed
    if not nx.is_connected(G_refined):
         print("Error: G_refined is still not connected before Gomory-Hu tree calculation. Cannot proceed.")
         # Option: raise error, or return gracefully
         raise nx.NetworkXError("Input graph for Gomory-Hu tree must be connected.")
         # Alternatively, return None or handle appropriately
         # return None

    T_refined = nx.gomory_hu_tree(G_refined, capacity='capacity')

    # Map T_refined nodes back to original IDs for the Tree
    T_refined_orig_ids = nx.relabel_nodes(T_refined, reverse_node_map, copy=True) # Use reverse_node_map here

    print("Saving T_refined (with original IDs)")
    nx.write_graphml(T_refined_orig_ids, output_ttree_path)
    print(f"Refined Gomory-Hu tree T saved to {output_ttree_path}")
    print(f"G_refined nodes: {G_refined.number_of_nodes()}")
    print(f"T_refined nodes: {T_refined.number_of_nodes()}")
    
    missing = set(node_list) - set(features_df.index)
    if missing:
        print(f"Warning: {len(missing)} node(s) in graph missing from features.")
    print(f"[DEBUG] node_features shape: {node_features.shape}")  # 应该是 (726, 50)
    print(f"[DEBUG] pyg_data.edge_index shape: {pyg_data.edge_index.shape}")  # 应该有很多条边
    print(f"[DEBUG] pyg_data.x shape: {pyg_data.x.shape}")  # 应该是 (726, 50)



# --- Placeholder for 'part' version ---
def process_clustering_network_part_enhanced(csv_path, leiden_col, target_cluster,
                                             output_gtree_path, output_ttree_path,
                                             node_features_path=None, embedding_dim=64, epochs=200, lr=0.01, use_vgae=True,
                                             output_embedding_path="node_embeddings_part.pt"):
    """
    Builds G', computes GHT' using GAE/VGAE for the 'part' scenario.
    Merges target_cluster nodes *before* GAE/VGAE training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    clustering_df = pd.read_csv(csv_path, index_col=0)
    clustering_df.index = clustering_df.index.astype(str) # Ensure string index
    print("previous: ")
    print(clustering_df.head())

    if leiden_col not in clustering_df.columns:
         raise ValueError(f"Column '{leiden_col}' not found in clustering data.")
    clustering_df[leiden_col] = clustering_df[leiden_col].astype(str) # Ensure target col is string
    target_cluster_str = str(target_cluster) # Ensure target is string

    # --- 1. Build Initial Graph with Merged Node ---
    print(f"Building initial graph, merging nodes from cluster '{target_cluster_str}' in column '{leiden_col}'...")
    G_merged = nx.Graph()
    cells_in_target_cluster = clustering_df[clustering_df[leiden_col] == target_cluster_str].index.tolist()
    super_node_id = "supernode_" + target_cluster_str # Unique ID for the merged node
    print(f"Super node for cluster {target_cluster_str}: {super_node_id}")

    # Add nodes, mapping target cells to the supernode
    node_list_merged = [super_node_id] + [cell for cell in clustering_df.index if cell not in cells_in_target_cluster]
    node_map_merged = {node: i for i, node in enumerate(node_list_merged)}
    reverse_node_map_merged = {v: k for k, v in node_map_merged.items()}

    for node in node_list_merged:
         G_merged.add_node(node_map_merged[node]) # Add mapped integer nodes

    # Add edges based on clustering agreement, redirecting target cluster edges
    for col in tqdm(clustering_df.columns):
        clustering_df[col] = clustering_df[col].astype(str)
        unique_clusters = clustering_df[col].unique()
        for cluster in unique_clusters:
            cells_in_cluster_col = clustering_df[clustering_df[col] == cluster].index.tolist()
            for i in range(len(cells_in_cluster_col)):
                for j in range(i + 1, len(cells_in_cluster_col)):
                    cell1_orig, cell2_orig = cells_in_cluster_col[i], cells_in_cluster_col[j]

                    # Map to supernode if needed
                    node1 = super_node_id if cell1_orig in cells_in_target_cluster else cell1_orig
                    node2 = super_node_id if cell2_orig in cells_in_target_cluster else cell2_orig

                    # Skip self-loops for the supernode
                    if node1 == node2:
                        continue

                    # Get mapped integer IDs
                    node1_mapped = node_map_merged[node1]
                    node2_mapped = node_map_merged[node2]

                    if G_merged.has_edge(node1_mapped, node2_mapped):
                        G_merged[node1_mapped][node2_mapped]['weight'] += 1
                    else:
                        G_merged.add_edge(node1_mapped, node2_mapped, weight=1)

    print(f"Initial merged graph nodes: {G_merged.number_of_nodes()}, edges: {G_merged.number_of_edges()}")
    if not nx.is_connected(G_merged):
         print("Warning: Initial merged graph is not connected. Adding minimal edges.")
         # Add connection logic similar to the 'all' case
         # ... (omitted for brevity, assume connection or add connection logic)
         pass

    # --- 2. Prepare Data for PyG (Features need careful handling for supernode) ---
    if node_features_path:
        features_df = pd.read_csv(node_features_path, index_col=0)
        features_df.index = features_df.index.astype(str)
        # Create feature vector for the supernode (e.g., mean of merged nodes)
        target_features = features_df.loc[cells_in_target_cluster]
        if target_features.empty:
             print(f"Warning: No features found for target cluster {target_cluster_str}. Using zero vector for supernode.")
             super_node_feature_vec = np.zeros(features_df.shape[1])
        else:
             super_node_feature_vec = target_features.mean(axis=0).values

        # Align features for other nodes
        other_cells = [cell for cell in node_list_merged if cell != super_node_id]
        other_features = features_df.reindex(other_cells)
        if other_features.isnull().values.any():
             print("Warning: Missing values found in non-target node features. Filling with zeros.")
             other_features = other_features.fillna(0)

        # Combine features in the correct order
        aligned_features_np = np.vstack([super_node_feature_vec] + [other_features.loc[cell].values for cell in other_cells])
        node_features = torch.tensor(aligned_features_np, dtype=torch.float).to(device)
        in_channels = node_features.shape[1]
        print(f"Loaded and processed node features for merged graph. Shape: {node_features.shape}")

    else:
        print("No node features provided. Using node degrees for merged graph.")
        degrees = torch.tensor([deg for node, deg in G_merged.degree()], dtype=torch.float).unsqueeze(1).to(device)
        node_features = degrees
        in_channels = 1

    pyg_data_merged = from_networkx(G_merged, group_node_attrs=['weight'] if 'weight' in G_merged.edges(data=True) else None)
    pyg_data_merged.x = node_features
    pyg_data_merged = pyg_data_merged.to(device)

    # --- 3. Initialize and Train GAE/VGAE on Merged Graph ---
    if use_vgae:
        model_part = VGAE(VariationalGCNEncoder(in_channels, embedding_dim)).to(device)
    else:
        model_part = GAE(GCNEncoder(in_channels, embedding_dim)).to(device)

    optimizer_part = torch.optim.Adam(model_part.parameters(), lr=lr)
    model_part = train_gae_vgae(model_part, optimizer_part, pyg_data_merged, num_epochs=epochs, is_vgae=use_vgae)

    # 4. Get Node Embeddings
    with torch.no_grad():
        z = model_part.encode(pyg_data_merged.x, pyg_data_merged.edge_index)
        if use_vgae and isinstance(z, tuple):  # Only unpack if it's VGAE output
            z = z[0]  
    print(z)
    print(f"Z shape: {z.shape}")
    embeddings = z.cpu().numpy()
    torch.save(torch.tensor(embeddings), output_embedding_path)
    print(f"Node embeddings saved to {output_embedding_path}")
    print(f"Embeddings shape: {embeddings.shape}")


    # with torch.no_grad():
    #     z = model_part.encode(pyg_data_merged.x, pyg_data_merged.edge_index)
    #     print(z,z.shape)
    #     if use_vgae: # For VGAE, use the mean embedding
    #          z = z[0] # z is (mu, logstd)
    # embeddings = z.cpu().numpy()
    # torch.save(torch.tensor(embeddings), output_embedding_path)
    # print(f"Node embeddings saved to {output_embedding_path}")
    # # Add print statement for debugging shape
    # print(f"Shape of embeddings BEFORE potential reshape: {embeddings.shape}")

    # --- Start of FIX ---
    # Ensure embeddings is at least 2D
    if embeddings.ndim == 1:
        print(f"Warning: Embeddings array is 1D (shape {embeddings.shape}). Reshaping to 2D.")
        # Logic to handle reshaping based on whether it's one node or dim=1
        # Assuming embedding_dim is defined in the function scope
        if embeddings.shape[0] == embedding_dim: # Shape (embedding_dim,) -> (1, embedding_dim) for single node
            embeddings = embeddings.reshape(1, -1)
        else: # Shape (num_nodes,) -> (num_nodes, 1) for embedding_dim=1
             embeddings = embeddings.reshape(-1, 1)
        print(f"Shape of embeddings AFTER reshape: {embeddings.shape}")
    # --- End of FIX ---


    # 5. Refine Graph Weights based on Embeddings
    print("Refining graph weights using learned embeddings...")
    G_merged_refined = nx.Graph()
    G_merged_refined.add_nodes_from(G_merged.nodes()) # Keep the mapped integer nodes

    num_nodes = embeddings.shape[0]

    # Use cosine similarity for weights (can explore other metrics)
    # Normalize embeddings for cosine similarity calculation

    # --- Start of FIX for Normalization ---
    # Calculate norms, avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Replace zero norms with a small value (epsilon) before division
    epsilon = 1e-9
    norms[norms < epsilon] = epsilon
    norm_embeddings = embeddings / norms
    # --- End of FIX for Normalization ---


    # Efficiently calculate cosine similarity for potential edges (can be memory intensive)
    # Consider using KNN or other methods for large graphs if full pairwise is too slow
    # Here, we iterate through initial edges + potentially add new ones based on similarity threshold
    similarity_threshold = 0.7 # Example threshold to add new edges (adjust as needed)

    processed_edges = set()
    # Use integer node indices (0 to num_nodes-1) for iteration with embeddings
    node_indices = list(range(num_nodes))

    for i in tqdm(node_indices):
        for j in node_indices[i + 1:]: # Correct loop for pairs
            # Calculate similarity between node i and node j using normalized embeddings
            similarity = np.dot(norm_embeddings[i], norm_embeddings[j])
            if not np.isfinite(similarity):
                print(f"[Warning] Similarity between node {i} and node {j} is not finite: {similarity}")
                continue
            if similarity > 1e-6: # Use similarity as weight if positive
                if G_merged.has_edge(i, j) or similarity > similarity_threshold:
                    G_merged_refined.add_edge(i, j, weight=similarity)
                    # processed_edges set might not be strictly necessary here anymore
                    # processed_edges.add(tuple(sorted((i, j))))

    print(f"Refined graph nodes: {G_merged_refined.number_of_nodes()}, edges: {G_merged_refined.number_of_edges()}")


    # Map nodes back to original IDs (including supernode) before saving G
    G_merged_refined_orig_ids = nx.relabel_nodes(G_merged_refined, reverse_node_map_merged, copy=True)
    print("Saving G'_refined (merged graph with original IDs/supernode ID)")
    nx.write_graphml(G_merged_refined_orig_ids, output_gtree_path)
    print(f"Refined merged graph G' saved to {output_gtree_path}")


    # --- 6. Compute Gomory-Hu Tree on Refined Merged Graph ---
    print("Calculating Gomory-Hu tree on refined merged graph...")
    for u, v, data in G_merged_refined.edges(data=True):
        G_merged_refined[u][v]['capacity'] = data.get('weight', 1.0)

    T_merged_refined = nx.gomory_hu_tree(G_merged_refined, capacity='capacity')

    # Map nodes back to original IDs (including supernode) for the Tree
    T_merged_refined_orig_ids = nx.relabel_nodes(T_merged_refined, reverse_node_map_merged, copy=True)

    print("Saving T'_refined (merged GHT with original IDs/supernode ID)")
    nx.write_graphml(T_merged_refined_orig_ids, output_ttree_path)
    print(f"Refined merged Gomory-Hu tree T' saved to {output_ttree_path}")

