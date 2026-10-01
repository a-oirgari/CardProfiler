import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.config import RANDOM_STATE


def grid_kmeans(X_pca_full, comps=range(2, 6), ks=range(2, 8)):
    rows = []
    for n in comps:
        X = X_pca_full[:, :n]
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
            tailles = pd.Series(km.labels_).value_counts(normalize=True)
            rows.append({
                "n_components": n, "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
                "min_pct": tailles.min() * 100,
                "max_pct": tailles.max() * 100,
            })
    return pd.DataFrame(rows)

def fit_kmeans(X_pca_full, n_components, k):
    X = X_pca_full[:, :n_components]
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    return km, X