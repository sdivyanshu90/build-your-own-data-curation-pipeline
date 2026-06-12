"""Disjoint Set Union (Union-Find) for clustering duplicate documents.

Candidate pairs form an undirected graph; its connected components are the
duplicate clusters. Union-Find computes those components in near-constant
amortised time per operation when it combines **path compression** with
**union-by-rank**, giving an inverse-Ackermann bound ``O(alpha(n))`` [Tarjan
1975] — effectively constant for any realistic ``n``.

Responsibility:
    * Provide :class:`UnionFind` with array-backed parent/rank storage, dynamic
      growth, component iteration, and a thread-safety guarantee.

Inputs:
    * Integer element ids and ``(a, b)`` union pairs.

Outputs:
    * Connected-component membership and cluster lists.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator


class UnionFind:
    """Array-backed Union-Find with path compression and union-by-rank.

    Elements are non-negative integers. The structure grows automatically when
    an out-of-range element is referenced, so the element count need not be
    known in advance.

    Thread-safety:
        **Thread-safe.** Every public operation (:meth:`find`, :meth:`union`,
        :meth:`connected`, :meth:`clusters`, :meth:`num_components`) acquires an
        internal re-entrant lock, so concurrent unions from multiple threads
        produce a correct, well-defined final partition. The lock serialises
        operations; for single-producer clustering (the pipeline's default) it
        adds negligible overhead.

    Args:
        size: Initial number of elements ``0..size-1`` (default 0; grows on use).

    Example:
        >>> uf = UnionFind(5)
        >>> uf.union(0, 1)
        >>> uf.union(1, 2)
        >>> uf.connected(0, 2)
        True
        >>> uf.connected(0, 3)
        False
        >>> sorted(sorted(c) for c in uf.clusters(min_size=2))
        [[0, 1, 2]]
    """

    def __init__(self, size: int = 0) -> None:
        self._parent: list[int] = list(range(size))
        self._rank: list[int] = [0] * size
        self._lock = threading.RLock()

    def _ensure_capacity(self, index: int) -> None:
        """Grow the parent/rank arrays so ``index`` is addressable.

        Args:
            index: The element id that must exist after this call.
        """
        current = len(self._parent)
        if index >= current:
            # New elements are their own parent (singletons) with rank 0.
            self._parent.extend(range(current, index + 1))
            self._rank.extend([0] * (index + 1 - current))

    def _find(self, x: int) -> int:
        """Find the root of ``x`` with full (two-pass) path compression.

        After this call every node on the path from ``x`` to the root points
        directly at the root.

        Args:
            x: The element id (must already be in range).

        Returns:
            The representative (root) id of ``x``'s set.
        """
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Second pass: repoint every node on the path straight to the root.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def find(self, x: int) -> int:
        """Return the representative of ``x``'s set (thread-safe).

        Args:
            x: The element id (auto-added if out of range).

        Returns:
            The root id of the set containing ``x``.

        Example:
            >>> uf = UnionFind()
            >>> uf.find(10)  # auto-grows; a fresh element is its own root
            10
        """
        with self._lock:
            self._ensure_capacity(x)
            return self._find(x)

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing ``a`` and ``b`` (thread-safe).

        Attaches the lower-rank tree under the higher-rank root (union-by-rank),
        keeping trees shallow.

        Args:
            a: First element id (auto-added if out of range).
            b: Second element id (auto-added if out of range).

        Example:
            >>> uf = UnionFind()
            >>> uf.union(3, 8)
            >>> uf.connected(3, 8)
            True
        """
        with self._lock:
            self._ensure_capacity(max(a, b))
            root_a, root_b = self._find(a), self._find(b)
            if root_a == root_b:
                return
            # Union by rank: keep the taller tree as the new root.
            if self._rank[root_a] < self._rank[root_b]:
                root_a, root_b = root_b, root_a
            self._parent[root_b] = root_a
            if self._rank[root_a] == self._rank[root_b]:
                self._rank[root_a] += 1

    def connected(self, a: int, b: int) -> bool:
        """Return whether ``a`` and ``b`` are in the same set (thread-safe).

        Args:
            a: First element id.
            b: Second element id.

        Returns:
            ``True`` if they share a root.
        """
        with self._lock:
            self._ensure_capacity(max(a, b))
            return self._find(a) == self._find(b)

    @property
    def num_elements(self) -> int:
        """The number of elements currently tracked."""
        with self._lock:
            return len(self._parent)

    def num_components(self) -> int:
        """Return the number of disjoint sets (thread-safe).

        Returns:
            The count of distinct roots among all tracked elements.
        """
        with self._lock:
            return sum(1 for x in range(len(self._parent)) if self._find(x) == x)

    def clusters(self, min_size: int = 1) -> Iterator[list[int]]:
        """Yield connected components with at least ``min_size`` members.

        Args:
            min_size: Minimum component size to emit (use 2 to drop singletons).

        Yields:
            Lists of element ids, one per qualifying component.

        Example:
            >>> uf = UnionFind(4)
            >>> uf.union(0, 2)
            >>> sorted(sorted(c) for c in uf.clusters(min_size=2))
            [[0, 2]]
        """
        with self._lock:
            groups: dict[int, list[int]] = {}
            for x in range(len(self._parent)):
                groups.setdefault(self._find(x), []).append(x)
        # Yielding happens after the snapshot is built under the lock.
        for members in groups.values():
            if len(members) >= min_size:
                yield members

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, int]], size: int = 0) -> UnionFind:
        """Build a :class:`UnionFind` by unioning a stream of pairs.

        Args:
            pairs: An iterable of ``(a, b)`` element pairs to union.
            size: Optional initial capacity.

        Returns:
            A populated :class:`UnionFind`.

        Example:
            >>> uf = UnionFind.from_pairs([(0, 1), (2, 3), (1, 2)])
            >>> uf.connected(0, 3)
            True
        """
        uf = cls(size)
        for a, b in pairs:
            uf.union(a, b)
        return uf
