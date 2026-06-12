"""Property-based invariants for :class:`UnionFind`.

For random sequences of union operations, the production Union-Find must agree
with a simple independent reference implementation on connectivity, component
count, and idempotence of :meth:`find`.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dedup_pipeline.clustering.union_find import UnionFind


def _reference_components(n: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    """A minimal, independent union-find used as ground truth.

    Returns a mapping ``element -> canonical root`` computed without rank, so it
    shares no code with the implementation under test.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return {x: find(x) for x in range(n)}


@settings(max_examples=200, deadline=None, derandomize=True)
@given(data=st.data())
def test_union_find_matches_reference(data: st.DataObject) -> None:
    """Connectivity, component count, and idempotence match the reference.

    Matters because clustering correctness is exactly the correctness of these
    three properties; a divergence would merge or split duplicate clusters.
    """
    n = data.draw(st.integers(min_value=2, max_value=500))
    num_edges = data.draw(st.integers(min_value=0, max_value=2 * n))
    edges = [
        (
            data.draw(st.integers(min_value=0, max_value=n - 1)),
            data.draw(st.integers(min_value=0, max_value=n - 1)),
        )
        for _ in range(num_edges)
    ]

    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)

    reference = _reference_components(n, edges)

    # 1. find(a) == find(b) iff a and b share a reference root.
    for _ in range(min(50, n)):
        x = data.draw(st.integers(min_value=0, max_value=n - 1))
        y = data.draw(st.integers(min_value=0, max_value=n - 1))
        same_under_test = uf.find(x) == uf.find(y)
        same_under_reference = reference[x] == reference[y]
        assert same_under_test == same_under_reference

    # 2. The number of distinct roots equals the number of components.
    distinct_roots = {uf.find(x) for x in range(n)}
    reference_components = set(reference.values())
    assert len(distinct_roots) == len(reference_components)

    # 3. find() is idempotent: find(find(x)) == find(x).
    for x in range(min(n, 100)):
        root = uf.find(x)
        assert uf.find(root) == root
