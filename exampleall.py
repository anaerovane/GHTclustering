from ghtree import process_clustering_network_all_enhanced
from processall import process_clustering_all_enhanced

clustering_csv = 'clustering_results_enhanced.csv'
node_features_csv = 'node_features_pca.csv'
embedding_dim = 32
gae_epochs = 150
gae_lr = 0.01
use_vgae_flag = True
embedding_path = "node_embeddings_all.pt"
gtree_path = "Gtree_refined.graphml"
ttree_path = "Ttree_refined.graphml"
final_csv = 'final_clustering_results_enhanced.csv'

process_clustering_network_all_enhanced(
    csv_path=clustering_csv,
    output_gtree_path=gtree_path,
    output_ttree_path=ttree_path,
    node_features_path=node_features_csv,
    embedding_dim=embedding_dim,
    epochs=gae_epochs,
    lr=gae_lr,
    use_vgae=use_vgae_flag,
    output_embedding_path=embedding_path
)

process_clustering_all_enhanced(
    T_refined_file=ttree_path,
    clustering_file=clustering_csv,
    initial_clustering_method="leiden",
    min_cut_threshold_percentile=30,
    attention_confidence_threshold=0.6,
    embedding_path=embedding_path,
    output_csv_path=final_csv
)

print("All optimization complete.")
