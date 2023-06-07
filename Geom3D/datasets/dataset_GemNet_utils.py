import numpy as np
import scipy.sparse as sp

'''
Credit to https://github.com/TUM-DAML/GemNet_pytorch/blob/master/GemNet/training/data_container.py
'''


def get_id_data_single(data, cutoff, int_cutoff, index_keys, triplets_only):
    N = len(data.x)

    adj_matrices = []
    adj_matrices_int = []

    R = data.positions

    D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    adj_mat = sp.csr_matrix(D_ij <= cutoff)
    adj_mat -= sp.eye(N, dtype=np.bool)
    adj_matrices.append(adj_mat)

    if not triplets_only:
        # get adjacency matrix for interaction
        adj_mat = sp.csr_matrix(D_ij <= int_cutoff)
        adj_mat -= sp.eye(N, dtype=np.bool)
        adj_matrices_int.append(adj_mat)
    
    #### Indices of the moleule structure
    idx_data = {key: None for key in index_keys}
    # Entry A_ij is edge j -> i (!)
    adj_matrix = _bmat_fast(adj_matrices)
    idx_t, idx_s = adj_matrix.nonzero()  # target and source nodes

    if not triplets_only:
        # Entry A_ij is edge j -> i (!)
        adj_matrix_int = _bmat_fast(adj_matrices_int)
        idx_int_t, idx_int_s = adj_matrix_int.nonzero()  # target and source nodes

    # catch no edge case
    if len(idx_t) == 0:
        for key in idx_data.keys():
            idx_data[key] = np.array([], dtype="int32")
        return idx_data

    # Get mask for undirected edges               0     1      nEdges/2  nEdges/2+1
    # change order of indices such that edge = [[0,1],[0,2], ..., [1,0], [2,0], ...]
    edges = np.stack([idx_t, idx_s], axis=0)
    mask = edges[0] < edges[1]
    edges = edges[:, mask]
    edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
    idx_t, idx_s = edges[0], edges[1]
    indices = np.arange(len(mask) / 2, dtype="int32")
    idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")

    idx_data["id_c"] = idx_s  # node c is source
    idx_data["id_a"] = idx_t  # node a is target

    if not triplets_only:
        idx_data["id4_int_a"] = idx_int_t
        idx_data["id4_int_b"] = idx_int_s
    #                                    0         1       ... nEdges/2  nEdges/2+1
    ## swap indices a->c to c->a:   [nEdges/2  nEdges/2+1  ...     0        1 ...   ]
    N_undir_edges = int(len(idx_s) / 2)
    ind = np.arange(N_undir_edges, dtype="int32")
    id_swap = np.concatenate([ind + N_undir_edges, ind])
    idx_data["id_swap"] = id_swap

    # assign an edge_id to each edge
    edge_ids = sp.csr_matrix((np.arange(len(idx_s)), (idx_t, idx_s)), shape=adj_matrix.shape, dtype="int32",)
    
    #### ------------------------------------ Triplets ------------------------------------ ####
    id3_expand_ba, id3_reduce_ca = get_triplets(idx_s, idx_t, edge_ids)
    # embed msg from c -> a with all quadruplets for k and l: c -> a <- k <- l
    # id3_reduce_ca is for k -> a -> c but we want c -> a <- k
    id3_reduce_ca = id_swap[id3_reduce_ca]

    # --------------------- Needed for efficient implementation --------------------- #
    if len(id3_reduce_ca) > 0:
        # id_reduce_ca must be sorted (i.e. grouped would suffice) for ragged_range !
        idx_sorted = np.argsort(id3_reduce_ca)
        id3_reduce_ca = id3_reduce_ca[idx_sorted]
        id3_expand_ba = id3_expand_ba[idx_sorted]
        _, K = np.unique(id3_reduce_ca, return_counts=True)
        idx_data["Kidx3"] = ragged_range(K)  # K = [1 4 2 3] -> Kidx3 = [0  0 1 2 3  0 1  0 1 2] , (nTriplets,)
    else:
        idx_data["Kidx3"] = np.array([], dtype="int32")
    # ------------------------------------------------------------------------------- #

    idx_data["id3_expand_ba"] = id3_expand_ba  # (nTriplets,)
    idx_data["id3_reduce_ca"] = id3_reduce_ca  # (nTriplets,)

    if triplets_only:
        return idx_data

    output = get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t)
    (
        id4_reduce_ca,
        id4_expand_db,
        id4_reduce_cab,
        id4_expand_abd,
        id4_reduce_intm_ca,
        id4_expand_intm_db,
        id4_reduce_intm_ab,
        id4_expand_intm_ab,
    ) = output

    if len(id4_reduce_ca) > 0:
        # id4_reduce_ca has to be sorted (i.e. grouped would suffice) for ragged range !
        sorted_idx = np.argsort(id4_reduce_ca)
        id4_reduce_ca = id4_reduce_ca[sorted_idx]
        id4_expand_db = id4_expand_db[sorted_idx]
        id4_reduce_cab = id4_reduce_cab[sorted_idx]
        id4_expand_abd = id4_expand_abd[sorted_idx]

        _, K = np.unique(id4_reduce_ca, return_counts=True)
        # K = [1 4 2 3] -> Kidx4 = [0  0 1 2 3  0 1  0 1 2]
        idx_data["Kidx4"] = ragged_range(K)  # (nQuadruplets,)
    else:
        idx_data["Kidx4"] = np.array([], dtype="int32")

    idx_data["id4_reduce_ca"] = id4_reduce_ca  # (nQuadruplets,)
    idx_data["id4_expand_db"] = id4_expand_db  # (nQuadruplets,)
    idx_data["id4_reduce_cab"] = id4_reduce_cab  # (nQuadruplets,)
    idx_data["id4_expand_abd"] = id4_expand_abd  # (nQuadruplets,)
    idx_data["id4_reduce_intm_ca"] = id4_reduce_intm_ca  # (intmTriplets,)
    idx_data["id4_expand_intm_db"] = id4_expand_intm_db  # (intmTriplets,)
    idx_data["id4_reduce_intm_ab"] = id4_reduce_intm_ab  # (intmTriplets,)
    idx_data["id4_expand_intm_ab"] = id4_expand_intm_ab  # (intmTriplets,)

    return idx_data


def get_id_data_list(data_list, cutoff, int_cutoff, index_keys, triplets_only):
    adj_matrices = []
    adj_matrices_int = []
    for data in data_list:
        N = len(data.x)  
        R = data.positions

        D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        # get adjacency matrix for embeddings
        adj_mat = sp.csr_matrix(D_ij <= cutoff)
        adj_mat -= sp.eye(N, dtype=np.bool)
        adj_matrices.append(adj_mat)

        if not triplets_only:
            # get adjacency matrix for interaction
            adj_mat = sp.csr_matrix(D_ij <= int_cutoff)
            adj_mat -= sp.eye(N, dtype=np.bool)
            adj_matrices_int.append(adj_mat)
    
    #### Indices of the moleule structure
    idx_data = {key: None for key in index_keys}
    # Entry A_ij is edge j -> i (!)
    adj_matrix = _bmat_fast(adj_matrices)
    idx_t, idx_s = adj_matrix.nonzero()  # target and source nodes

    if not triplets_only:
        # Entry A_ij is edge j -> i (!)
        adj_matrix_int = _bmat_fast(adj_matrices_int)
        idx_int_t, idx_int_s = adj_matrix_int.nonzero()  # target and source nodes

    # catch no edge case
    if len(idx_t) == 0:
        for key in idx_data.keys():
            idx_data[key] = np.array([], dtype="int32")
        return idx_data

    # Get mask for undirected edges               0     1      nEdges/2  nEdges/2+1
    # change order of indices such that edge = [[0,1],[0,2], ..., [1,0], [2,0], ...]
    edges = np.stack([idx_t, idx_s], axis=0)
    mask = edges[0] < edges[1]
    edges = edges[:, mask]
    edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
    idx_t, idx_s = edges[0], edges[1]
    indices = np.arange(len(mask) / 2, dtype="int32")
    idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")

    idx_data["id_c"] = idx_s  # node c is source
    idx_data["id_a"] = idx_t  # node a is target

    if not triplets_only:
        idx_data["id4_int_a"] = idx_int_t
        idx_data["id4_int_b"] = idx_int_s
    #                                    0         1       ... nEdges/2  nEdges/2+1
    ## swap indices a->c to c->a:   [nEdges/2  nEdges/2+1  ...     0        1 ...   ]
    N_undir_edges = int(len(idx_s) / 2)
    ind = np.arange(N_undir_edges, dtype="int32")
    id_swap = np.concatenate([ind + N_undir_edges, ind])
    idx_data["id_swap"] = id_swap

    # assign an edge_id to each edge
    edge_ids = sp.csr_matrix((np.arange(len(idx_s)), (idx_t, idx_s)), shape=adj_matrix.shape, dtype="int32",)
    
    #### ------------------------------------ Triplets ------------------------------------ ####
    id3_expand_ba, id3_reduce_ca = get_triplets(idx_s, idx_t, edge_ids)
    # embed msg from c -> a with all quadruplets for k and l: c -> a <- k <- l
    # id3_reduce_ca is for k -> a -> c but we want c -> a <- k
    id3_reduce_ca = id_swap[id3_reduce_ca]

    # --------------------- Needed for efficient implementation --------------------- #
    if len(id3_reduce_ca) > 0:
        # id_reduce_ca must be sorted (i.e. grouped would suffice) for ragged_range !
        idx_sorted = np.argsort(id3_reduce_ca)
        id3_reduce_ca = id3_reduce_ca[idx_sorted]
        id3_expand_ba = id3_expand_ba[idx_sorted]
        _, K = np.unique(id3_reduce_ca, return_counts=True)
        idx_data["Kidx3"] = ragged_range(K)  # K = [1 4 2 3] -> Kidx3 = [0  0 1 2 3  0 1  0 1 2] , (nTriplets,)
    else:
        idx_data["Kidx3"] = np.array([], dtype="int32")
    # ------------------------------------------------------------------------------- #

    idx_data["id3_expand_ba"] = id3_expand_ba  # (nTriplets,)
    idx_data["id3_reduce_ca"] = id3_reduce_ca  # (nTriplets,)

    if triplets_only:
        return idx_data

    output = get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t)
    (
        id4_reduce_ca,
        id4_expand_db,
        id4_reduce_cab,
        id4_expand_abd,
        id4_reduce_intm_ca,
        id4_expand_intm_db,
        id4_reduce_intm_ab,
        id4_expand_intm_ab,
    ) = output

    if len(id4_reduce_ca) > 0:
        # id4_reduce_ca has to be sorted (i.e. grouped would suffice) for ragged range !
        sorted_idx = np.argsort(id4_reduce_ca)
        id4_reduce_ca = id4_reduce_ca[sorted_idx]
        id4_expand_db = id4_expand_db[sorted_idx]
        id4_reduce_cab = id4_reduce_cab[sorted_idx]
        id4_expand_abd = id4_expand_abd[sorted_idx]

        _, K = np.unique(id4_reduce_ca, return_counts=True)
        # K = [1 4 2 3] -> Kidx4 = [0  0 1 2 3  0 1  0 1 2]
        idx_data["Kidx4"] = ragged_range(K)  # (nQuadruplets,)
    else:
        idx_data["Kidx4"] = np.array([], dtype="int32")

    idx_data["id4_reduce_ca"] = id4_reduce_ca  # (nQuadruplets,)
    idx_data["id4_expand_db"] = id4_expand_db  # (nQuadruplets,)
    idx_data["id4_reduce_cab"] = id4_reduce_cab  # (nQuadruplets,)
    idx_data["id4_expand_abd"] = id4_expand_abd  # (nQuadruplets,)
    idx_data["id4_reduce_intm_ca"] = id4_reduce_intm_ca  # (intmTriplets,)
    idx_data["id4_expand_intm_db"] = id4_expand_intm_db  # (intmTriplets,)
    idx_data["id4_reduce_intm_ab"] = id4_reduce_intm_ab  # (intmTriplets,)
    idx_data["id4_expand_intm_ab"] = id4_expand_intm_ab  # (intmTriplets,)

    return idx_data


def get_id_data_list_for_material(data_list, index_keys):
    adj_matrices = []
    adj_matrices_int = []
    for data in data_list:
        N = len(data.x)
        R = data.positions
        edge_index = data.edge_index.numpy()
        M = len(edge_index[0])

        data = [1 for _ in range(M)]
        adj_mat = sp.csr_matrix((data, edge_index), shape=(N, N))
        adj_matrices.append(adj_mat)

    #### Indices of the moleule structure
    idx_data = {key: None for key in index_keys}
    # Entry A_ij is edge j -> i (!)
    adj_matrix = _bmat_fast(adj_matrices)
    idx_t, idx_s = adj_matrix.nonzero()  # target and source nodes

    # catch no edge case
    if len(idx_t) == 0:
        for key in idx_data.keys():
            idx_data[key] = np.array([], dtype="int32")
        return idx_data

    # Get mask for undirected edges               0     1      nEdges/2  nEdges/2+1
    # change order of indices such that edge = [[0,1],[0,2], ..., [1,0], [2,0], ...]
    edges = np.stack([idx_t, idx_s], axis=0)
    mask = edges[0] < edges[1]
    edges = edges[:, mask]
    edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
    idx_t, idx_s = edges[0], edges[1]
    indices = np.arange(len(mask) / 2, dtype="int32")
    idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")

    idx_data["id_c"] = idx_s  # node c is source
    idx_data["id_a"] = idx_t  # node a is target

    #                                    0         1       ... nEdges/2  nEdges/2+1
    ## swap indices a->c to c->a:   [nEdges/2  nEdges/2+1  ...     0        1 ...   ]
    N_undir_edges = int(len(idx_s) / 2)
    ind = np.arange(N_undir_edges, dtype="int32")
    id_swap = np.concatenate([ind + N_undir_edges, ind])
    idx_data["id_swap"] = id_swap

    # assign an edge_id to each edge
    edge_ids = sp.csr_matrix((np.arange(len(idx_s)), (idx_t, idx_s)), shape=adj_matrix.shape, dtype="int32",)
    
    #### ------------------------------------ Triplets ------------------------------------ ####
    id3_expand_ba, id3_reduce_ca = get_triplets(idx_s, idx_t, edge_ids)
    # embed msg from c -> a with all quadruplets for k and l: c -> a <- k <- l
    # id3_reduce_ca is for k -> a -> c but we want c -> a <- k
    id3_reduce_ca = id_swap[id3_reduce_ca]

    # --------------------- Needed for efficient implementation --------------------- #
    if len(id3_reduce_ca) > 0:
        # id_reduce_ca must be sorted (i.e. grouped would suffice) for ragged_range !
        idx_sorted = np.argsort(id3_reduce_ca)
        id3_reduce_ca = id3_reduce_ca[idx_sorted]
        id3_expand_ba = id3_expand_ba[idx_sorted]
        _, K = np.unique(id3_reduce_ca, return_counts=True)
        idx_data["Kidx3"] = ragged_range(K)  # K = [1 4 2 3] -> Kidx3 = [0  0 1 2 3  0 1  0 1 2] , (nTriplets,)
    else:
        idx_data["Kidx3"] = np.array([], dtype="int32")
    # ------------------------------------------------------------------------------- #

    idx_data["id3_expand_ba"] = id3_expand_ba  # (nTriplets,)
    idx_data["id3_reduce_ca"] = id3_reduce_ca  # (nTriplets,)

    return idx_data


def _bmat_fast(mats):
    """Combines multiple adjacency matrices into single sparse block matrix.
    Parameters
    ----------
        mats: list
            Has adjacency matrices as elements.
    Returns
    -------
        adj_matrix: sp.csr_matrix
            Combined adjacency matrix (sparse block matrix)
    """
    assert len(mats) > 0
    new_data = np.concatenate([mat.data for mat in mats])

    ind_offset = np.zeros(1 + len(mats), dtype="int32")
    ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
    new_indices = np.concatenate(
        [mats[i].indices + ind_offset[i] for i in range(len(mats))]
    )

    indptr_offset = np.zeros(1 + len(mats))
    indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
    new_indptr = np.concatenate(
        [mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))]
    )

    # Resulting matrix shape: sum of matrices
    shape = (ind_offset[-1], ind_offset[-1])

    # catch case with no edges
    if len(new_data) == 0:
        return sp.csr_matrix(shape)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=shape)


def get_triplets(idx_s, idx_t, edge_ids):
    """
    Get triplets c -> a <- b
    """
    # Edge indices of triplets k -> a -> i
    id3_expand_ba = edge_ids[idx_s].data.astype("int32").flatten()
    id3_reduce_ca = edge_ids[idx_s].tocoo().row.astype("int32").flatten()

    id3_i = idx_t[id3_reduce_ca]
    id3_k = idx_s[id3_expand_ba]
    mask = id3_i != id3_k
    id3_expand_ba = id3_expand_ba[mask]
    id3_reduce_ca = id3_reduce_ca[mask]

    return id3_expand_ba, id3_reduce_ca


def get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t):
    """
    c -> a - b <- d where D_ab <= int_cutoff; D_ca & D_db <= cutoff
    """
    # Number of incoming edges to target and source node of interaction edges
    nNeighbors_t = adj_matrix[idx_int_t].sum(axis=1).A1.astype("int32")
    nNeighbors_s = adj_matrix[idx_int_s].sum(axis=1).A1.astype("int32")
    id4_reduce_intm_ca = (
        edge_ids[idx_int_t].data.astype("int32").flatten()
    )  # (intmTriplets,)
    id4_expand_intm_db = (
        edge_ids[idx_int_s].data.astype("int32").flatten()
    )  # (intmTriplets,)
    # note that id4_reduce_intm_ca and id4_expand_intm_db have the same shape but
    # id4_reduce_intm_ca[i] and id4_expand_intm_db[i] may not belong to the same interacting quadruplet !

    # each reduce edge (c->a) has to be repeated as often as there are neighbors for node b
    # vice verca for the edges of the source node (d->b) and node a
    id4_reduce_cab = repeat_blocks(
        nNeighbors_t, nNeighbors_s
    )  # (nQuadruplets,)
    id4_reduce_ca = id4_reduce_intm_ca[id4_reduce_cab]  # intmTriplets -> nQuadruplets

    N = np.repeat(nNeighbors_t, nNeighbors_s)
    id4_expand_abd = np.repeat(
        np.arange(len(id4_expand_intm_db)), N
    )  # (nQuadruplets,)
    id4_expand_db = id4_expand_intm_db[id4_expand_abd]  # intmTriplets -> nQuadruplets

    id4_reduce_intm_ab = np.repeat(
        np.arange(len(idx_int_t)), nNeighbors_t
    )  # (intmTriplets,)
    id4_expand_intm_ab = np.repeat(
        np.arange(len(idx_int_t)), nNeighbors_s
    )  # (intmTriplets,)

    # Mask out all quadruplets where nodes appear more than once
    idx_c = idx_s[id4_reduce_ca]
    idx_a = idx_t[id4_reduce_ca]
    idx_b = idx_t[id4_expand_db]
    idx_d = idx_s[id4_expand_db]

    mask1 = idx_c != idx_b
    mask2 = idx_a != idx_d
    mask3 = idx_c != idx_d
    mask = mask1 * mask2 * mask3  # logical and

    id4_reduce_ca = id4_reduce_ca[mask]
    id4_expand_db = id4_expand_db[mask]
    id4_reduce_cab = id4_reduce_cab[mask]
    id4_expand_abd = id4_expand_abd[mask]

    return (
        id4_reduce_ca,
        id4_expand_db,
        id4_reduce_cab,
        id4_expand_abd,
        id4_reduce_intm_ca,
        id4_expand_intm_db,
        id4_reduce_intm_ab,
        id4_expand_intm_ab,
    )


def repeat_blocks(sizes, repeats):
    """Repeat blocks of indices.
    From https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements
    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3]
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [0,3,2] ; repeats = [3,2,3]
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2]
        Return: [0 1 0 1  5 6 5 6]
    """
    a = np.arange(np.sum(sizes))
    indices = np.empty((sizes * repeats).sum(), dtype=np.int32)
    start = 0
    oi = 0
    for i, size in enumerate(sizes):
        end = start + size
        for _ in range(repeats[i]):
            oe = oi + size
            indices[oi:oe] = a[start:end]
            oi = oe
        start = end
    return indices


def ragged_range(sizes):
    """
    -------
    Example
    -------
        sizes = [1,3,2] ;
        Return: [0  0 1 2  0 1]
    """
    a = np.arange(sizes.max())
    indices = np.empty(sizes.sum(), dtype=np.int32)
    start = 0
    for size in sizes:
        end = start + size
        indices[start:end] = a[:size]
        start = end
    return indices
