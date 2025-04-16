from ghtree import process_clustering_network_part_enhanced
from processpart import process_clustering_part_enhanced

clustering_csv = 'clustering_results_enhanced.csv'
node_features_csv = 'node_features_pca.csv'
embedding_dim = 10
hyperbolic_distance_threshold = 1.5
new_col_name = 'leiden_expanded_2'
final_csv_part = 'final_clustering_results_part_expanded_c2.csv'

process_clustering_network_part_enhanced(
    csv_path=clustering_csv,
    leiden_col='leiden',
    target_cluster='2',
    output_gtree_path="Gtree_part_refined_c2.graphml",
    output_ttree_path="Ttree_part_refined_c2.graphml",
    node_features_path=node_features_csv,
    embedding_dim=embedding_dim,
    epochs=100,
    lr=0.01,
    use_vgae=True
)

process_clustering_part_enhanced(
    graphml_path="Ttree_part_refined_c2.graphml",
    csv_path=clustering_csv,
    leiden_col='leiden',
    target_cluster='2',
    hyperbolic_distance_threshold=hyperbolic_distance_threshold,
    new_col_name=new_col_name,
    output_path=final_csv_part,
    embedding_dim=embedding_dim
)

print("Part optimization complete.")
