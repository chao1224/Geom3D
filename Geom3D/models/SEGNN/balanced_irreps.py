from e3nn.o3 import (FullyConnectedTensorProduct, Irreps, Linear,
                     spherical_harmonics)


def BalancedIrreps(lmax, vec_dim, sh_type=True):
    """ Allocates irreps equally along channel budget, resulting
        in unequal numbers of irreps in ratios of 2l_i + 1 to 2l_j + 1.

    Parameters
    ----------
    lmax : int
        Maximum order of irreps.
    vec_dim : int
        Dim of feature vector.
    sh_type : bool
        if true, use spherical harmonics. Else the full set of irreps (with redundance).

    Returns
    -------
    Irreps
        Resulting irreps for feature vectors.

    """
    irrep_spec = "0e"
    for l in range(1, lmax + 1):
        if sh_type:
            irrep_spec += " + {0}".format(l) + ('e' if (l % 2) == 0 else 'o')
        else:
            irrep_spec += " + {0}e + {0}o".format(l)
    irrep_spec_split = irrep_spec.split(" + ")
    dims = [int(irrep[0]) * 2 + 1 for irrep in irrep_spec_split]
    # Compute ratios
    ratios = [1 / dim for dim in dims]
    # Determine how many copies per irrep
    irrep_copies = [int(vec_dim * r / len(ratios)) for r in ratios]
    # Determine the current effective irrep sizes
    irrep_dims = [n * dim for (n, dim) in zip(irrep_copies, dims)]
    # Add trivial irreps until the desired size is reached
    irrep_copies[0] += vec_dim - sum(irrep_dims)

    # Convert to string
    str_out = ''
    for (spec, dim) in zip(irrep_spec_split, irrep_copies):
        str_out += str(dim) + 'x' + spec
        str_out += ' + '
    str_out = str_out[:-3]
    # Generate the irrep
    return Irreps(str_out)


def WeightBalancedIrreps(irreps_in1_scalar, irreps_in2, sh=True, lmax=None):
    """Determines an irreps_in1 type of order irreps_in2.lmax that when used in a tensor product
    irreps_in1 x irreps_in2 -> irreps_in1
    would have the same number of weights as for a standard linear layer, e.g. a tensor product
    irreps_in1_scalar x "1x0e" -> irreps_in1_scalar

    Parameters
    ----------
    irreps_in1_scalar : o3.Irreps
        Number of hidden features, represented by zeroth order irreps.
    irreps_in2 : o3.Irreps
        Irreps related to edge attributes.
    sh : bool
        if true, yields equal number of every order. Else returns balanced irrep.
    lmax : int
        Maximum order irreps to be considered.

    Returns
    -------
    o3.Irreps
        Irreps for hidden feaure vectors. 

    """

    n = 1
    if lmax == None:
        lmax = irreps_in2.lmax
    irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
    weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    weight_numel_scalar = FullyConnectedTensorProduct(irreps_in1_scalar, Irreps("1x0e"), irreps_in1_scalar).weight_numel
    while weight_numel1 < weight_numel_scalar:  # TODO: somewhat suboptimal implementation...
        n += 1
        irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
        weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    print('Determined irrep type:', irreps_in1)
    return Irreps(irreps_in1)
