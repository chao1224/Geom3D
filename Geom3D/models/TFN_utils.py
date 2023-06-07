import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""

    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization

        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(self, num_freq, in_dim, out_dim, edge_dim: int = 0):
        """NN parameterized radial profile function.

        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.net = nn.Sequential(
            nn.Linear(self.edge_dim + 1, self.mid_dim),
            BN(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            BN(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.num_freq * in_dim * out_dim),
        )

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)

    def __repr__(self):
        return f"RadialFunc(edge_dim={self.edge_dim}, in_dim={self.in_dim}, out_dim={self.out_dim})"


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""

    def __init__(
        self,
        degree_in: int,
        nc_in: int,
        degree_out: int,
        nc_out: int,
        edge_dim: int = 0,
    ):
        """SE(3)-equivariant convolution between a pair of feature types.

        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.

        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        # Log settings
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        # Functions of the degree
        self.num_freq = 2 * min(degree_in, degree_out) + 1
        self.d_out = 2 * degree_out + 1
        self.edge_dim = edge_dim

        # Radial profile function
        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, self.edge_dim)

    def forward(self, feat, basis):
        # Get radial weights
        R = self.rp(feat)
        kernel = torch.sum(R * basis[f"{self.degree_in},{self.degree_out}"], -1)
        return kernel.view(kernel.shape[0], self.d_out * self.nc_out, -1)

    def __repr__(self):
        return f"PairwiseConv(edge_dim={self.edge_dim}, degree_in={self.degree_in}, degree_out={self.degree_out})"


class GConvSE3(nn.Module):
    """A tensor field network layer as a DGL module.
    GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """

    def __init__(self, f_in, f_out, self_interaction=False, edge_dim=0, flavor="skip"):
        """SE(3)-equivariant Graph Conv Layer
        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
            self_interaction: include self-interaction in convolution
            edge_dim: number of dimensions for edge embedding
            flavor: allows ['TFN', 'skip'], where 'skip' adds a skip connection
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.flavor = flavor

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f"({di},{do})"] = PairwiseConv(
                    di, mi, do, mo, edge_dim=edge_dim
                )

        # Center -> center weights
        self.kernel_self = nn.ParameterDict()
        if self_interaction:
            assert self.flavor in ["TFN", "skip"]
            if self.flavor == "TFN":
                for m_out, d_out in self.f_out.structure:
                    W = nn.Parameter(torch.randn(1, m_out, m_out) / np.sqrt(m_out))
                    self.kernel_self[f"{d_out}"] = W
            elif self.flavor == "skip":
                for m_in, d_in in self.f_in.structure:
                    if d_in in self.f_out.degrees:
                        m_out = self.f_out.structure_dict[d_in]
                        W = nn.Parameter(torch.randn(1, m_out, m_in) / np.sqrt(m_in))
                        self.kernel_self[f"{d_in}"] = W

    # def forward(self, h, G=None, r=None, basis=None, **kwargs):
    def forward(self, h, r, basis, edge_index, edge_feat=None, **kwargs):
        """Forward pass of the linear layer
        Args:
            G: minibatch of (homo)graphs
            h: dict of features
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        # Add node features to local graph scope
        G = {}
        N = h["0"].size()[0]
        for k, v in h.items():
            G[k] = v

        if edge_feat is not None:
            feat = torch.cat([edge_feat, r], -1)
        else:
            feat = r

        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                etype = f"({di},{do})"
                G[etype] = self.kernel_unary[etype](feat, basis)

        row, col = edge_index
        result = {}
        # Perform message-passing for each output feature type
        for d_out in self.f_out.degrees:
            msg = 0

            # Neighbor -> center messages
            for m_in, d_in in self.f_in.structure:
                h_ = G[f"{d_in}"]
                src = h_[row].view(-1, m_in * (2 * d_in + 1), 1)
                edge = G[f"({d_in},{d_out})"]
                msg += torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            # Center -> center messages
            if self.self_interaction:
                if f"{d_out}" in self.kernel_self.keys():
                    if self.flavor == "TFN":
                        W = self.kernel_self[f"{d_out}"]
                        msg = torch.matmul(W, msg)
                    if self.flavor == "skip":
                        h_ = G[f"{d_out}"]
                        dst = h_[col]
                        W = self.kernel_self[f"{d_out}"]
                        msg = msg + torch.matmul(W, dst)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            # TODO: row or col?
            result[f"{d_out}"] = scatter_mean(msg, col, dim=0, dim_size=N)
        return result

    def __repr__(self):
        return f"GConvSE3(structure={self.f_out}, self_interaction={self.self_interaction})"


class GNormSE3(nn.Module):
    """Graph Norm-based SE(3)-equivariant nonlinearity.
    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:
    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase
    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """

    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int = 0):
        """Initializer.
        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform[str(d)] = self._build_net(int(m))

    def _build_net(self, m: int):
        net = []
        for i in range(self.num_layers):
            net.append(BN(int(m)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(m, m, bias=(i == self.num_layers - 1)))
            nn.init.kaiming_uniform_(net[-1].weight)
        if self.num_layers == 0:
            net.append(BN(int(m)))
            net.append(self.nonlin)
        return nn.Sequential(*net)

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            transformed = self.transform[str(k)](norm[..., 0]).unsqueeze(-1)

            # Nonlinearity on norm
            output[k] = (transformed * phase).view(*v.shape)

        return output

    def __repr__(self):
        return f"GNormSE3(num_layers={self.num_layers}, nonlin={self.nonlin})"
