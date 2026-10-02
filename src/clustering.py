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

def score_clustering(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if len(set(labels[mask])) < 2:      # les métriques exigent au moins 2 clusters
        return None
    return {
        "n_clusters": len(set(labels[mask])),
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "pct_bruit": (~mask).mean() * 100,
    }

def grid_dbscan(X_pca_full, comps=range(2, 6),
                eps_list=(0.3, 0.5, 0.7, 1.0, 1.5), min_samples_list=(5, 10, 20, 50)):
    rows = []
    for n in comps:
        X = X_pca_full[:, :n]
        for eps in eps_list:
            for ms in min_samples_list:
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                s = score_clustering(X, labels)
                if s is None:
                    continue
                rows.append({"n_components": n, "eps": eps, "min_samples": ms,
                             "n_clusters": s["n_clusters"],
                             "silhouette": s["silhouette"], "pct_bruit": s["pct_bruit"]})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)

def fit_dbscan(X_pca_full, n_components, eps, min_samples):
    X = X_pca_full[:, :n_components]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels, X