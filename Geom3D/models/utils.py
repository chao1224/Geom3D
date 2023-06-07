"""
Credit to https://github.com/FabianFuchsML/se3-transformer-public
"""

import torch

from .from_se3cnn import utils_steerable


def get_basis(cloned_d, max_degree):
    """Precompute the SE(3)-equivariant weight basis, W_J^lk(x)
    Args:
        G: DGL graph instance of type dgl.DGLGraph
        max_degree: non-negative int for degree of highest feature type
    Returns:
        dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
        tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
        where the 1's will later be broadcast to the number of output and input
        channels
    """
    with torch.no_grad():
        # Relative positional encodings (vector)
        r_ij = utils_steerable.get_spherical_from_cartesian_torch(cloned_d)
        # Spherical harmonic basis
        Y = utils_steerable.precompute_sh(r_ij, 2 * max_degree)
        device = Y[0].device

        basis = {}
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_in - d_out), d_in + d_out + 1):
                    # Get spherical harmonic projection matrices
                    Q_J = utils_steerable._basis_transformation_Q_J(J, d_in, d_out)
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (-1, 1, 2 * d_out + 1, 1, 2 * d_in + 1, 2 * min(d_in, d_out) + 1)
                basis[f"{d_in},{d_out}"] = torch.stack(K_Js, -1).view(*size)
        return basis
