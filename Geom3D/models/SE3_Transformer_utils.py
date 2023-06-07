import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from .fibers import Fiber, fiber2head
from .TFN_utils import PairwiseConv


class GSE3Res(nn.Module):
    """Graph attention block with SE(3)-equivariance and skip connection"""

    def __init__(
        self,
        f_in,
        f_out,
        edge_dim=0,
        div=4,
        n_heads=1,
        learnable_skip=True,
        skip="cat",
        selfint="1x1",
        x_ij=None,
    ):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        self.n_heads = n_heads
        self.skip = skip  # valid: 'cat', 'sum', None

        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'
        # this will be used for the values
        f_mid_out = {k: int(v // div) for k, v in self.f_out.structure_dict.items()}
        self.f_mid_out = Fiber(dictionary=f_mid_out)

        # f_mid_in has same structure as f_mid_out, but only degrees which are in f_in
        # this will be used for keys and queries
        # (queries are merely projected, hence degrees have to match input)
        f_mid_in = {d: m for d, m in f_mid_out.items() if d in self.f_in.degrees}
        self.f_mid_in = Fiber(dictionary=f_mid_in)

        self.edge_dim = edge_dim

        self.GMAB = nn.ModuleDict()

        # Projections
        self.GMAB["v"] = GConvSE3Partial(
            f_in, self.f_mid_out, edge_dim=edge_dim, x_ij=x_ij
        )
        self.GMAB["k"] = GConvSE3Partial(
            f_in, self.f_mid_in, edge_dim=edge_dim, x_ij=x_ij
        )
        self.GMAB["q"] = G1x1SE3(f_in, self.f_mid_in)

        # Attention
        self.GMAB["attn"] = GMABSE3(self.f_mid_out, self.f_mid_in, n_heads=n_heads)

        # Skip connections
        if self.skip == "cat":
            self.cat = GCat(self.f_mid_out, f_in)
            if selfint == "att":
                self.project = GAttentiveSelfInt(self.cat.f_out, f_out)
            elif selfint == "1x1":
                self.project = G1x1SE3(self.cat.f_out, f_out, learnable=learnable_skip)
            elif self.skip == "sum":
                self.project = G1x1SE3(self.f_mid_out, f_out, learnable=learnable_skip)
                self.add = GSum(f_out, f_in)
                # the following checks whether the skip connection would change
                # the output fibre strucure; the reason can be that the input has
                # more channels than the ouput (for at least one degree); this would
                # then cause a (hard to debug) error in the next layer
                assert (
                    self.add.f_out.structure_dict == f_out.structure_dict
                ), "skip connection would change output structure"
        return

    def forward(self, h, edge_index, **kwargs):
        # Embeddings
        v = self.GMAB["v"](h, edge_index=edge_index, **kwargs)
        k = self.GMAB["k"](h, edge_index=edge_index, **kwargs)
        q = self.GMAB["q"](h, edge_index=edge_index)

        # Attention
        z = self.GMAB["attn"](v=v, k=k, q=q, h=h, edge_index=edge_index)

        if self.skip == "cat":
            z = self.cat(z, h)
            z = self.project(z)
        elif self.skip == "sum":
            # Skip + residual
            z = self.project(z)
            z = self.add(z, h)
        return z


class GConvSE3Partial(nn.Module):
    """Graph SE(3)-equivariant node -> edge layer"""

    def __init__(self, f_in, f_out, edge_dim: int = 0, x_ij=None):
        """SE(3)-equivariant partial convolution.
        A partial convolution computes the inner product between a kernel and
        each input channel, without summing over the result from each input
        channel. This unfolded structure makes it amenable to be used for
        computing the value-embeddings of the attention mechanism.
        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
        """
        super().__init__()
        self.f_out = f_out
        self.edge_dim = edge_dim

        # adding/concatinating relative position to feature vectors
        # 'cat' concatenates relative position & existing feature vector
        # 'add' adds it, but only if multiplicity > 1
        assert x_ij in [None, "cat", "add"]
        self.x_ij = x_ij
        if x_ij == "cat":
            self.f_in = Fiber.combine(f_in, Fiber(structure=[(1, 1)]))
        else:
            self.f_in = f_in

        # Node -> edge weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f"({di},{do})"] = PairwiseConv(
                    di, mi, do, mo, edge_dim=edge_dim
                )

    def forward(self, h, positions, r, basis, edge_index, edge_feat=None):
        """Forward pass of the linear layer
        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        # Add node features to local graph scope
        G = {}
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
            for m_in, d_in in self.f_in.structure:

                # if type 1 and flag set, add relative position as feature
                if self.x_ij == "cat" and d_in == 1:
                    # TODO: double-check
                    # relative positions
                    rel = (positions[row] - positions[col]).view(-1, 3, 1)
                    m_ori = m_in - 1

                    if m_ori == 0:
                        # no type 1 input feature, just use relative position
                        src = rel
                    else:
                        # TODO: double-check
                        h_ = G[f"{d_in}"]
                        # features of src node, shape [edges, m_in*(2l+1), 1]
                        src = h_[row].view(-1, m_ori * (2 * d_in + 1), 1)
                        # add to feature vector
                        src = torch.cat([src, rel], dim=1)
                elif self.x_ij == "add" and d_in == 1 and m_in > 1:
                    # TODO: double-check
                    h_ = G[f"{d_in}"]
                    src = h_[row].view(-1, m_in * (2 * d_in + 1), 1)
                    rel = (positions[row] - positions[col]).view(-1, 3, 1)
                    src[..., :3, :1] = src[..., :3, :1] + rel
                else:
                    h_ = G[f"{d_in}"]
                    src = h_[row].view(-1, m_in * (2 * d_in + 1), 1)

                edge = G[f"({d_in},{d_out})"]
                msg = msg + torch.matmul(edge, src)

            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)
            result[f"{d_out}"] = msg

        return result

    def __repr__(self):
        return f"GConvSE3Partial(structure={self.f_out})"


class G1x1SE3(nn.Module):
    """Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    This is equivalent to a self-interaction layer in TensorField Networks.
    """

    def __init__(self, f_in, f_out, learnable=True):
        """SE(3)-equivariant 1x1 convolution.
        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        for m_out, d_out in self.f_out.structure:
            m_in = self.f_in.structure_dict[d_out]
            self.transform[str(d_out)] = nn.Parameter(
                torch.randn(m_out, m_in) / np.sqrt(m_in), requires_grad=learnable
            )

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            if str(k) in self.transform.keys():
                output[k] = torch.matmul(self.transform[str(k)], v)
        return output

    def __repr__(self):
        return f"G1x1SE3(structure={self.f_out})"


class GMABSE3(nn.Module):
    """An SE(3)-equivariant multi-headed self-attention module for DGL graphs."""

    def __init__(self, f_value, f_key, n_heads):
        """SE(3)-equivariant MAB (multi-headed attention block) layer.
        Args:
            f_value: Fiber() object for value-embeddings
            f_key: Fiber() object for key-embeddings
            n_heads: number of heads
        """
        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads

    def forward(self, v, k, q, edge_index, **kwargs):
        """Forward pass of the linear layer
        Args:
            G: minibatch of (homo)graphs
            v: dict of value edge-features
            k: dict of key edge-features
            q: dict of query node-features
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """

        # Add node features to local graph scope
        ## We use the stacked tensor representation for attention
        G = {}
        for m, d in self.f_value.structure:
            G[f"v{d}"] = v[f"{d}"].view(-1, self.n_heads, m // self.n_heads, 2 * d + 1)
        G["k"] = fiber2head(
            k, self.n_heads, self.f_key, squeeze=True
        )  # [edges, heads, channels](?)
        G["q"] = fiber2head(
            q, self.n_heads, self.f_key, squeeze=True
        )  # [nodes, heads, channels](?)
        N = G["q"].size()[0]

        # Compute attention weights
        ## Inner product between (key) neighborhood and (query) center
        row, col = edge_index
        e = (G["k"] * G["q"][col]).sum(2)

        # TODO: need double-check
        e = e / np.sqrt(self.f_key.n_features)
        e = e.exp()
        s = scatter_add(e, row, dim=0, dim_size=N)
        a = e / s[row]

        result = {}
        # Perform attention-weighted message-passing
        for d_out in self.f_value.degrees:
            attn = a.unsqueeze(-1).unsqueeze(-1)
            value = G[f"v{d_out}"]
            msg = attn * value

            # TODO: row or col?
            G[f"out{d_out}"] = scatter_add(msg, col, dim=0, dim_size=N)

        result = {}
        for m, d in self.f_value.structure:
            result[f"{d}"] = G[f"out{d}"].view(-1, m, 2 * d + 1)

        return result

    def __repr__(self):
        return f"GMABSE3(n_heads={self.n_heads}, structure={self.f_value})"


class GAttentiveSelfInt(nn.Module):
    def __init__(self, f_in, f_out):
        """SE(3)-equivariant 1x1 convolution.
        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.nonlin = nn.LeakyReLU()
        self.num_layers = 2
        self.eps = 1e-12  # regularisation for phase: gradients explode otherwise

        # one network for attention weights per degree
        self.transform = nn.ModuleDict()
        for o, m_in in self.f_in.structure_dict.items():
            m_out = self.f_out.structure_dict[o]
            self.transform[str(o)] = self._build_net(m_in, m_out)

    def _build_net(self, m_in: int, m_out):
        n_hidden = m_in * m_out
        cur_inpt = m_in * m_in
        net = []
        for i in range(1, self.num_layers):
            net.append(nn.LayerNorm(int(cur_inpt)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(cur_inpt, n_hidden, bias=(i == self.num_layers - 1)))
            nn.init.kaiming_uniform_(net[-1].weight)
            cur_inpt = n_hidden
        return nn.Sequential(*net)

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # v shape: [..., m, 2*k+1]
            first_dims = v.shape[:-2]
            m_in = self.f_in.structure_dict[int(k)]
            m_out = self.f_out.structure_dict[int(k)]
            assert v.shape[-2] == m_in
            assert v.shape[-1] == 2 * int(k) + 1

            # Compute the norms and normalized features
            # norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(v)
            # phase = v / norm # [..., m, 2*k+1]
            scalars = torch.einsum("...ac,...bc->...ab", [v, v])  # [..., m_in, m_in]
            scalars = scalars.view(*first_dims, m_in * m_in)  # [..., m_in*m_in]
            sign = scalars.sign()
            scalars = scalars.abs_().clamp_min(self.eps)
            scalars *= sign

            # perform attention
            att_weights = self.transform[str(k)](scalars)  # [..., m_out*m_in]
            att_weights = att_weights.view(
                *first_dims, m_out, m_in
            )  # [..., m_out, m_in]
            att_weights = F.softmax(input=att_weights, dim=-1)
            # shape [..., m_out, 2*k+1]
            # output[k] = torch.einsum('...nm,...md->...nd', [att_weights, phase])
            output[k] = torch.einsum("...nm,...md->...nd", [att_weights, v])

        return output

    def __repr__(self):
        return f"AttentiveSelfInteractionSE3(in={self.f_in}, out={self.f_out})"


class GSum(nn.Module):
    """SE(3)-equvariant graph residual sum function."""

    def __init__(self, f_x: Fiber, f_y: Fiber):
        """SE(3)-equvariant graph residual sum function.

        Args:
            f_x: Fiber() object for fiber of summands
            f_y: Fiber() object for fiber of summands
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        self.f_out = Fiber.combine_max(f_x, f_y)

    def __repr__(self):
        return f"GSum(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if (k in x) and (k in y):
                if x[k].shape[1] > y[k].shape[1]:
                    diff = x[k].shape[1] - y[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(
                        y[k].device
                    )
                    y[k] = torch.cat([y[k], zeros], 1)
                elif x[k].shape[1] < y[k].shape[1]:
                    diff = y[k].shape[1] - x[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(
                        y[k].device
                    )
                    x[k] = torch.cat([x[k], zeros], 1)

                out[k] = x[k] + y[k]
            elif k in x:
                out[k] = x[k]
            elif k in y:
                out[k] = y[k]
        return out


class GCat(nn.Module):
    """Concat only degrees which are in f_x"""

    def __init__(self, f_x: Fiber, f_y: Fiber):
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        f_out = {}
        for k in f_x.degrees:
            f_out[k] = f_x.dict[k]
            if k in f_y.degrees:
                f_out[k] += f_y.dict[k]
        self.f_out = Fiber(dictionary=f_out)

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if k in y:
                out[k] = torch.cat([x[k], y[k]], 1)
            else:
                out[k] = x[k]
        return out

    def __repr__(self):
        return f"GCat(structure={self.f_out})"
