"""Converts atomic-facts to vectors; reduces their dimensionality; then uses HDBScan
to cluster them. This could be a replacement or enhancement for pairwise duplicate
detection to find sets of equivalent atoms. Currently, just a prototype to explore.
"""

from datetime import date
from typing import cast

import hdbscan  # type: ignore
import numpy as np
import umap  # type: ignore

from api.models import Point
from api.service import MarkingSchemeService, PolygraphResponseService
from polygraph_benchmark.deduplication import encode_points

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 10
UMAP_COMPONENTS = 15
UMAP_NEIGHBOURS = 30


def reduce_umap(emb: np.ndarray) -> np.ndarray:
    """Reduce dimensions of embeddings"""
    # we start with 384 dimensions and reduce to n_components, e.g. 15/50/100
    # n_components and n_neighbors must both be strictly less than n_samples
    n = len(emb)
    n_components = min(UMAP_COMPONENTS, n - 2)
    n_neighbors = min(UMAP_NEIGHBOURS, n - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        n_jobs=1,
    )
    return cast(np.ndarray, reducer.fit_transform(emb))


def cluster_embeddings(embeddings_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find clusters of similar embeddings (e.g. beliefs)
    Also calculate return probabilities to record membership strength"""
    # The main argument to tweak is min_cluster_size: large means large, broad clusters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    cluster_labels = clusterer.fit_predict(embeddings_array)
    return cluster_labels, clusterer.probabilities_


def cluster_atoms(all_atoms: list[Point]) -> dict[int, list[Point]]:
    """Cluster a list of atoms."""
    # if len(all_atoms) < 20:  # must have at least min_cluster_size atoms
    #     return {}

    # numpy requires CPU tensors, regardless of the hardware used:
    embeddings_array = encode_points(all_atoms).cpu().numpy()
    embeddings_reduced = reduce_umap(embeddings_array)
    cluster_labels, _ = cluster_embeddings(embeddings_reduced)
    clusters: dict[int, list[Point]] = {}
    for atom, label in zip(all_atoms, cluster_labels):
        # Set -1 is the default "noise" cluster. Probably want to skip this.
        if label == -1:
            continue  # HDBSCAN noise points
        clusters.setdefault(int(label), []).append(atom)

    return clusters


def cluster_question_atoms(
    question: str, date_from: date, date_to: date
) -> dict[int, list[Point]]:
    """Get all atoms in date range for one question, and cluster them."""
    service = MarkingSchemeService()
    llm_service = PolygraphResponseService()
    all_atoms: list[Point] = []
    for m in llm_service.get_all_models():
        all_atoms += service.get_points_for_range(question, m, date_from, date_to)
    if not all_atoms:
        return {}
    return cluster_atoms(all_atoms)
