"""
Huffman compression for video frame transmission over WiFi.

Used to reduce bandwidth usage when streaming from SJ4000 action cameras
mounted on the Glaideron AUV to the shore-side receiver station.

Based on: bachelor's thesis, SamGTU, 2023 (Babaev B.G.)
"""

import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ─── Tree node ────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _Node:
    freq: int
    symbol: Optional[int] = field(default=None, compare=False)
    left:  Optional["_Node"] = field(default=None, compare=False)
    right: Optional["_Node"] = field(default=None, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ─── Build tree ───────────────────────────────────────────────────────────────

def _build_tree(data: bytes) -> _Node:
    """Build Huffman tree from byte frequency table."""
    freq = Counter(data)
    heap = [_Node(f, sym) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _Node(lo.freq + hi.freq, left=lo, right=hi))

    return heap[0]


def _build_codes(node: _Node, prefix: str = "", table: dict | None = None) -> dict:
    """Recursively generate binary codes for each symbol."""
    if table is None:
        table = {}
    if node.is_leaf:
        table[node.symbol] = prefix or "0"
    else:
        _build_codes(node.left,  prefix + "0", table)
        _build_codes(node.right, prefix + "1", table)
    return table


# ─── Public API ───────────────────────────────────────────────────────────────

def compress(data: bytes) -> tuple[bytes, dict]:
    """
    Compress raw bytes using Huffman coding.

    Returns:
        compressed  – packed bytes (bit-padded to full bytes)
        code_table  – {byte_value: bit_string} needed for decompression
    """
    if not data:
        return b"", {}

    tree = _build_tree(data)
    codes = _build_codes(tree)

    bit_string = "".join(codes[b] for b in data)

    # Pad to multiple of 8
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding

    compressed = bytes(
        int(bit_string[i : i + 8], 2) for i in range(0, len(bit_string), 8)
    )
    # Prepend padding length as first byte
    return bytes([padding]) + compressed, codes


def decompress(data: bytes, codes: dict) -> bytes:
    """
    Decompress Huffman-encoded bytes back to original data.

    Args:
        data       – compressed bytes (first byte = padding length)
        codes      – code table returned by compress()
    """
    if not data or not codes:
        return b""

    padding = data[0]
    reverse = {v: k for k, v in codes.items()}

    bit_string = "".join(f"{byte:08b}" for byte in data[1:])
    if padding:
        bit_string = bit_string[:-padding]

    result, current = [], ""
    for bit in bit_string:
        current += bit
        if current in reverse:
            result.append(reverse[current])
            current = ""

    return bytes(result)


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    sample = b"Glaideron AUV underwater defect detection system " * 50
    compressed, table = compress(sample)
    restored = decompress(compressed, table)

    ratio = len(compressed) / len(sample) * 100
    print(f"Original : {len(sample):>6} bytes")
    print(f"Compressed: {len(compressed):>6} bytes  ({ratio:.1f}%)")
    print(f"Restored : {'OK' if restored == sample else 'FAIL'}")
