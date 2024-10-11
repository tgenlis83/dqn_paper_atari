import operator
from typing import Callable, List, Union


class SegmentTree:
    """
    Segment Tree data structure for efficient range queries and updates.
    """

    def __init__(
        self,
        capacity: int,
        operation: Callable[[float, float], float],
        init_value: float,
    ):
        assert capacity > 0 and (
            capacity & (capacity - 1) == 0
        ), "Capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Recursively apply the operation over the range [start, end].

        Args:
            start (int): The start index of the range.
            end (int): The end index of the range.
            node (int): The current node in the segment tree.
            node_start (int): The start index of the current node's range.
            node_end (int): The end index of the current node's range.

        Returns:
            float: The result of the operation over the range.
        """
        if start > node_end or end < node_start:
            return 0 if self.operation == operator.add else float("inf")
        if start <= node_start and node_end <= end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        left = self._operate(start, end, 2 * node, node_start, mid)
        right = self._operate(start, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(left, right)

    def operate(self, start: int = 0, end: int = None) -> float:
        """Apply the operation over the range [start, end].

        Args:
            start (int): The start index of the range.
            end (int): The end index of the range.

        Returns:
            float: The result of the operation over the range.
        """
        if end is None:
            end = self.capacity - 1
        return self._operate(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Update the value at index `idx` and propagate the change up the tree.

        Args:
            idx (int): The index to update.
            val (float): The new value.
        """
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            left = self.tree[2 * idx]
            right = self.tree[2 * idx + 1]
            self.tree[idx] = self.operation(left, right)
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get the value at index `idx`.

        Args:
            idx (int): The index to retrieve.

        Returns:
            float: The value at the specified index.
        """
        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """Segment Tree specialized for sum operations."""

    def __init__(self, capacity: int):
        super().__init__(capacity, operator.add, 0.0)

    def sum(self, start: int = 0, end: int = None) -> float:
        """Calculate the sum over the range [start, end].

        Args:
            start (int): The start index of the range.
            end (int): The end index of the range.

        Returns:
            float: The sum over the range.
        """
        return super().operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Retrieve the index where the prefix sum is just greater than `upperbound`.

        Args:
            upperbound (float): The upper bound for the prefix sum.

        Returns:
            int: The index where the prefix sum is just greater than `upperbound`.
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """Segment Tree specialized for minimum operations."""

    def __init__(self, capacity: int):
        super().__init__(capacity, min, float("inf"))

    def min(self, start: int = 0, end: int = None) -> float:
        """Calculate the minimum over the range [start, end].

        Args:
            start (int): The start index of the range.
            end (int): The end index of the range.

        Returns:
            float: The minimum value over the range.
        """
        return super().operate(start, end)
