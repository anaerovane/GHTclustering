import scanpy as sc
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering

def run_initial_clustering(
    data_file,
    subsample_fraction,
    n_pca_components,
    leiden_resolution,
    n_clusters_kmeans,
    n_clusters_spectral,
    output_clustering_csv,
    output_node_features_csv
):
    print("Loading and preprocessing data...")
    adata = sc.read_h5ad(data_file)
    if subsample_fraction < 1.0:
        sc.pp.subsample(adata, fraction=subsample_fraction, random_state=42)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
    adata = adata[:, adata.var['highly_variable']]
    sc.tl.pca(adata, n_comps=n_pca_components, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pca_components)
    
    pca_df = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs.index.astype(str))
    pca_df.to_csv(output_node_features_csv)

    sc.tl.leiden(adata, resolution=leiden_resolution, key_added='leiden')
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
    spectral = SpectralClustering(n_clusters=n_clusters_spectral, affinity='rbf', random_state=42)
    adata.obs['spectral'] = spectral.fit_predict(adata.obsm['X_pca']).astype(str)

    clustering_df = adata.obs[['leiden', 'kmeans', 'spectral']]
    clustering_df.to_csv(output_clustering_csv)

    print(f"Clustering saved to {output_clustering_csv}")
    print(f"PCA features saved to {output_node_features_csv}")

run_initial_clustering(
    data_file='E:\\newscCOD2\\scGHT-master\\FCA_lung_0.01.h5ad',
    subsample_fraction=0.001,
    n_pca_components=50,
    leiden_resolution=1.0,
    n_clusters_kmeans=8,
    n_clusters_spectral=8,
    output_clustering_csv='clustering_results_enhanced.csv',
    output_node_features_csv='node_features_pca.csv'
)