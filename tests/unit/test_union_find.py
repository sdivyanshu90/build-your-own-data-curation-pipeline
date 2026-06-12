"""Unit tests for :class:`UnionFind`.

Union-Find computes the duplicate clusters from candidate pairs. Correctness
(merges and queries) and the efficiency invariants (path compression,
union-by-rank) are both load-bearing for large-scale clustering.
"""

from __future__ import annotations

import math
import random

from dedup_pipeline.clustering.union_find import UnionFind


def test_find_after_union() -> None:
    """After union(a, b), find(a) == find(b).

    Matters because this is the defining property of a merge; without it
    duplicates would not cluster.
    """
    uf = UnionFind(5)
    uf.union(1, 3)
    assert uf.find(1) == uf.find(3)


def test_transitivity() -> None:
    """union(a,b) and union(b,c) imply find(a) == find(c).

    Matters because duplicate relations are transitive: a~b and b~c means a, b, c
    are one cluster.
    """
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)


def test_path_compression_flattens() -> None:
    """After find() on every node, each points directly at the root.

    Matters because path compression is what keeps amortised cost near-constant;
    a non-flattened tree degrades to linear-time finds.
    """
    uf = UnionFind(6)
    for a, b in [(0, 1), (2, 3), (0, 2), (0, 4), (0, 5)]:
        uf.union(a, b)
    root = uf.find(0)
    for x in range(6):
        uf.find(x)
    assert all(uf._parent[x] == root for x in range(6))


def test_union_by_rank_bounds_rank() -> None:
    """No root's rank exceeds log2(n) after arbitrary unions.

    Matters because union-by-rank caps tree height at O(log n), the second half
    of the near-constant-time guarantee.
    """
    n = 1000
    uf = UnionFind(n)
    rng = random.Random(0)
    for _ in range(2 * n):
        uf.union(rng.randrange(n), rng.randrange(n))
    assert max(uf._rank) <= math.log2(n)


def test_cluster_enumeration_hand_graph() -> None:
    """Connected components are enumerated correctly on a known graph.

    Matters because the components *are* the duplicate clusters; wrong components
    mean wrong dedup.
    """
    uf = UnionFind(8)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(5, 6)
    clusters = sorted(sorted(c) for c in uf.clusters(min_size=2))
    assert clusters == [[0, 1, 2], [5, 6]]


def test_singletons_excluded_with_min_size() -> None:
    """min_size=2 excludes singleton components.

    Matters because unique documents (singletons) are not duplicates and must be
    kept, not clustered.
    """
    uf = UnionFind(4)
    uf.union(0, 2)
    clusters = list(uf.clusters(min_size=2))
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 2]


def test_num_components() -> None:
    """The component count reflects the merges performed.

    Matters because the component count drives statistics like the cluster
    histogram.
    """
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.num_components() == 3  # {0,1}, {2,3}, {4}


def test_dynamic_growth() -> None:
    """Referencing an out-of-range element auto-grows the structure.

    Matters because candidate pairs reference arbitrary indices not known at
    construction time.
    """
    uf = UnionFind(2)
    uf.union(0, 100)  # 100 is out of the initial range
    assert uf.connected(0, 100)


def test_empty_structure() -> None:
    """An empty UnionFind has no components (pathological input).

    Matters because an empty corpus must produce zero clusters without error.
    """
    uf = UnionFind(0)
    assert uf.num_components() == 0
    assert list(uf.clusters(min_size=2)) == []


def test_single_element() -> None:
    """A single element is its own component (pathological input).

    Matters because a one-document corpus must be handled gracefully.
    """
    uf = UnionFind(1)
    assert uf.find(0) == 0
    assert uf.connected(0, 0)
    assert list(uf.clusters(min_size=2)) == []


def test_from_pairs_builds_components() -> None:
    """from_pairs builds the same partition as manual unions.

    Matters because the pipeline builds DSU directly from the candidate-pair
    stream via this constructor.
    """
    uf = UnionFind.from_pairs([(0, 1), (2, 3), (1, 2)])
    assert uf.connected(0, 3)
    assert uf.num_components() == 1
