"""
Create a quimb TensorNetwork from an adjacency matrix where each node is a factor
whose tensor values follow a Boltzmann-like weight:

    value = exp(beta * (product of all incident binary variables))

Here each bond corresponds to a binary variable (0 or 1). The product of all
incident variables equals 1 only when every incident variable is 1; otherwise 0.
So the tensor's entries are 1 for all configurations except the all-ones entry,
which is exp(beta).

Functions provided:
- lattice_adjacency(n, periodic=False) -> adjacency matrix for n x n lattice
- build_tn_from_adj(adj, beta=1.0) -> quimb TensorNetwork
- contract_tn(tn) -> contracted scalar (partition function)

Requires: quimb (qutip/quimb) and numpy.
"""

import math
import numpy as np
import quimb.tensor as qtn



def kagome_adjacency(n):
    """Return adjacency matrix for an n x n kagome lattice."""
    assert n % 2 == 0

    N = n * n
    adj = np.zeros((N, N), dtype=int)

    def idx(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            if r % 2 == 1:
                i = idx(r, c)
                # diagonal
                if c % 2 == 1:
                    if r - 1 >= 0 and c - 1 >= 0:
                        upleft = idx(r - 1, c - 1)
                        adj[i, upleft] = adj[upleft, i] = 1
                    if r + 1 < n and c + 1 < n:
                        downright = idx(r + 1, c + 1)
                        adj[i, downright] = adj[downright, i] = 1
                else:
                    for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if (r + x) >= 0 and r + x < n and (c + y) >= 0 and c + y < n:
                            neighbour = idx(r + x, c + y)
                            adj[i, neighbour] = adj[neighbour, i] = 1
    rows_to_delete = []  # delete 1st and 3rd rows
    for r in range(n):
        for c in range(n):
            if r % 2 == 0 and c % 2 == 1:
                i = idx(r, c)
                rows_to_delete.append(i)

    adj = np.delete(adj, rows_to_delete, axis=0)
    adj = np.delete(adj, rows_to_delete, axis=1)

    return adj


def lattice_adjacency(n, periodic=False):
    """Return adjacency matrix for an n x n 2D square lattice.

    Nodes are numbered row-major: 0..n*n-1. Edges to up/down/left/right.
    If periodic=True, implements periodic boundary conditions.
    """
    N = n * n
    adj = np.zeros((N, N), dtype=int)

    def idx(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            i = idx(r, c)
            # right neighbor
            if c + 1 < n:
                j = idx(r, c + 1)
                adj[i, j] = adj[j, i] = 1
            elif periodic:
                j = idx(r, 0)
                adj[i, j] = adj[j, i] = 1
            # down neighbor
            if r + 1 < n:
                j = idx(r + 1, c)
                adj[i, j] = adj[j, i] = 1
            elif periodic:
                j = idx(0, c)
                adj[i, j] = adj[j, i] = 1
    return adj


def bbcode_adjacency(n, offset, periodic=False):
    """Return adjacency matrix for an n x n 2D square lattice plus long range connections."""
    N = n * n
    adj = np.zeros((N, N), dtype=int)

    def idx(r, c):
        return (r % n) * n + (c % n)

    for r in range(n):
        for c in range(n):
            i = idx(r, c)

            if c + 1 < n or periodic:
                j = idx(r, c + 1)
                adj[i, j] = adj[j, i] = 1
            if r + 1 < n or periodic:
                j = idx(r + 1, c)
                adj[i, j] = adj[j, i] = 1

            if r % 2 == c % 2:
                if (r + offset < n and c - offset >= 0) or periodic:
                    j = idx(r + offset, c - offset)
                    adj[i, j] = adj[j, i] = 1

    return adj


def boltzmann_tensor(d, beta):
    """
    Construct a tensor for a node with `len(inds)` legs.

    - Shape: (2,)*d where d = len(inds)
    - Entries: all ones, except the all-ones index (1,...,1) gets 'val'.

    Parameters
    ----------
    inds : list[str]
        Indices (one per leg).
    val : float
        Value to place at the all-ones entry (default: e).

    Returns
    -------
    data : np.ndarray
        Tensor array with shape (2,)*len(inds).
    """
    if d == 0:
        # scalar case
        return np.array(val, dtype=float)

    data = np.ones((2,) * d, dtype=float)
    data[(1,) * d] = np.exp(-beta)
    return data


def build_tn_from_adj_coordinate_names(adj, beta=1.0):
    """Build a quimb TensorNetwork from adjacency matrix `adj`.

    - `adj` should be a symmetric 2D numpy array of 0/1 entries.
    - Each undirected edge becomes a bond with dimension 2 (binary variable).
    - For node i with degree d, its tensor has shape (2,)*d and entries:
          T[all zeros..] = 1, ..., T[all ones] = exp(beta).

    Returns: quimb.tensor.TensorNetwork
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")

    N = adj.shape[0]

    # canonical bond name for edge (i,j) with i<j
    def bond_name(i, j):
        a, b = (i, j) if i < j else (j, i)
        return f"bond_{a}_{b}"

    tensors = []

    for i in range(N):
        neigh = list(np.nonzero(adj[i])[0])
        deg = len(neigh)

        if deg == 0:
            # Node with no incident edges: treat as scalar factor exp(beta * 1)
            data = np.array(np.exp(-beta), dtype=float)
            inds = [f"node_{i}_scalar"]
            tensors.append(qtn.Tensor(data, inds=inds, tags={f"N{i}"}))
            continue

        # Determine bond names for this node's legs in a deterministic order
        # We'll sort neighbor indices to make bond ordering reproducible
        neigh_sorted = sorted(neigh)
        inds = [bond_name(i, j) for j in neigh_sorted]

        # Create tensor data: ones everywhere, exp(beta) at the all-ones index
        shape = (2,) * deg
        data = np.ones(shape, dtype=float)
        data[(1,) * deg] = np.exp(-beta)

        data /= np.sum(data)

        side = np.sqrt(N)
        tensors.append(
            qtn.Tensor(data, inds=inds, tags={f"N{math.floor(i / side)}_{i % side}"})
        )

    tn = qtn.TensorNetwork(tensors)
    return tn


def build_tn_from_adj(adj, beta=1.0, normalized=True):
    """Build a quimb TensorNetwork from adjacency matrix `adj`.

    - `adj` should be a symmetric 2D numpy array of 0/1 entries.
    - Each undirected edge becomes a bond with dimension 2 (binary variable).
    - For node i with degree d, its tensor has shape (2,)*d and entries:
          T[all zeros..] = 1, ..., T[all ones] = exp(beta).

    Returns: quimb.tensor.TensorNetwork
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")

    N = adj.shape[0]

    # canonical bond name for edge (i,j) with i<j
    def bond_name(i, j):
        a, b = (i, j) if i < j else (j, i)
        return f"bond_{a}_{b}"

    tensors = []

    for i in range(N):
        neigh = list(np.nonzero(adj[i])[0])
        deg = len(neigh)

        if deg == 0:
            # Node with no incident edges: treat as scalar factor exp(beta * 1)
            data = np.array(np.exp(-beta), dtype=float)
            inds = [f"node_{i}_scalar"]
            tensors.append(qtn.Tensor(data, inds=inds, tags={f"N{i}"}))
            continue

        # Determine bond names for this node's legs in a deterministic order
        # We'll sort neighbor indices to make bond ordering reproducible
        neigh_sorted = sorted(neigh)
        inds = [bond_name(i, j) for j in neigh_sorted]

        # Create tensor data: ones everywhere, exp(beta) at the all-ones index
        shape = (2,) * deg
        data = np.ones(shape, dtype=float)
        data[(1,) * deg] = np.exp(-beta)

        # data /= np.sum(data)

        tensors.append(qtn.Tensor(data, inds=inds, tags={f"N{i}"}))

    tn = qtn.TensorNetwork(tensors)
    if normalized:
        Z = tn.contract()
        tn = tn.multiply(1 / Z)
    return tn


def contract_tn(tn, optimize="auto"):
    """Contract the tensor network to a scalar. Use with small graphs only.

    The contraction may be expensive (exponential) for large graphs.
    `optimize` passed to quimb's contract method.
    """
    # make a copy so we do not destroy original indexing metadata
    tn_copy = tn.copy()
    result = tn_copy.contract(all, optimize=optimize)
    return result


def adj_from_file(filename):
    edges = []
    max_node = -1

    with open(filename, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                u, v = map(int, line.split())
                edges.append((u, v))
                max_node = max(max_node, u, v)

    n = max_node + 1
    adj_matrix = np.zeros((n, n), dtype=int)

    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return adj_matrix
