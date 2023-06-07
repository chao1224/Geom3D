import torch
import math
import sys

from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from typing import Optional, Tuple, Union
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from math import sqrt
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from math import pi
try:
    import sympy as sym
except ImportError:
    sym = None
from math import pi as PI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class rbf_emb(nn.Module):
    '''
    modified: delete cutoff with r
    '''
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False, **kwargs):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist=dist.unsqueeze(-1)
        rbounds = 0.5 * \
                  (torch.cos(dist * PI / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        a = self.betas
        d = self.means
        c= torch.square((torch.exp(-dist) - self.means))
        b = torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))
        return rbounds*torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        #self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.dist_emb = rbf_emb(num_radial, cutoff)
        # self.first_emb = dist_emb2(num_radial2, cutoff, envelope_exponent)
        # self.second_emb = dist_emb2(num_radial2, cutoff, envelope_exponent)
        # self.vertical_emb = dist_emb2(num_radial2, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()
        # self.first_emb.reset_parameters()
        # self.second_emb.reset_parameters()
        # self.vertical_emb.reset_parameters()

    def forward(self, dist):
        dist_emb = self.dist_emb(dist)
        # first_emb = self.first_emb(first)
        # second_emb = self.first_emb(second)
        # vertical_emb = self.first_emb(vertical)

        return dist_emb  # , first_emb, second_emb, vertical_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        else:  # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _, _ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class ClofNet(torch.nn.Module):
    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=64, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
            num_radial=12, num_radial2=80, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3, heads=1,
            act=swish, output_init='GlorotOrthogonal', use_node_features=True, **kwargs):
        super(ClofNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.init = nn.Linear(num_radial, hidden_channels)
        self.ln_emb = nn.LayerNorm(hidden_channels,
                                   elementwise_affine=False)  # kwargs['ln_learnable']) #if ln_emb else nn.Identity()
        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.heads = heads
        # self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.emblin = Embedding(95, hidden_channels)
        self.emblinfinal = Embedding(95, 1)
        self.final = nn.Linear(hidden_channels,1)
        self.emb = emb(num_radial, self.cutoff, envelope_exponent)
        self.neighbor_emb = NeighborEmb(hidden_channels, **kwargs)
        self.s2v = CFConvS2V(hidden_channels, **kwargs)
        self.freq1 = torch.nn.Parameter(torch.Tensor(32))
        self.lin1 = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            #nn.LayerNorm(hidden_channels, elementwise_affine=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        # self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            #nn.LayerNorm(hidden_channels, elementwise_affine=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin3 = nn.Sequential(
            nn.Linear(3, hidden_channels//2),
            #nn.LayerNorm(hidden_channels//4, elementwise_affine=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels//2, 32))
        self.lin4 = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            #nn.LayerNorm(hidden_channels // 4, elementwise_affine=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 3))
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dir_proj = nn.Sequential(
            nn.Linear(4 * hidden_channels + num_radial, hidden_channels), nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), )
        self.edge_proj = nn.Sequential(
             nn.Linear(3, hidden_channels//2),nn.LayerNorm(hidden_channels // 2, elementwise_affine=False), nn.SiLU(inplace=True),
             nn.Linear(hidden_channels//2, 32), )
        self.interaction1 = TransformerConv(hidden_channels, hidden_channels, heads=self.heads,
                                            edge_dim=self.hidden_channels, **kwargs)

        self.interactions = nn.ModuleList()
        for i in range(num_layers-1):
            self.interactions.append(TransformerConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, edge_dim=self.hidden_channels, **kwargs))

        self.mlp = nn.Sequential(nn.LayerNorm(hidden_channels * self.heads, elementwise_affine=False),
                                 nn.Linear(hidden_channels * self.heads, 1, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        self.freq1.data = torch.arange(0, self.freq1.numel()).float().mul_(pi)

    def global_add_pool(x: Tensor, batch: Optional[Tensor],
                        size: Optional[int] = None) -> Tensor:

        if batch is None:
            return x.sum(dim=-2, keepdim=x.dim() == 2)
        size = int(batch.max().item() + 1) if size is None else size
        return scatter(x, batch, dim=-2, dim_size=size, reduce='add')

    def forward(self, z, pos, batch):
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes = z.size(0)
        z_emb = self.emblin(z)
        z_emb = self.ln_emb(z_emb)
        i,j = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        coord_diff = pos[i] - pos[j]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(pos[i], pos[j])
        norm = torch.sqrt(radial) + 0.0001
        coord_diff = coord_diff / norm
        cross_norm = (torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1))) + 0.0001
        coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)
        frame = torch.cat((coord_diff.unsqueeze(-1), coord_cross.unsqueeze(-1), coord_vertical.unsqueeze(-1)), dim=-1)
        # dist, first, second, vertical, i, j, idx_kj, idx_ji = self.xyz_to_dat(pos, edge_index, num_nodes)
        emb = self.emb(dist)
        f = self.lin1(emb)
        rbounds = 0.5 * \
                  (torch.cos(dist * pi / self.cutoff) + 1.0)
        frequency = (self.freq1.detach() * dist.unsqueeze(-1)/ self.cutoff).cos()

        f = rbounds.unsqueeze(-1) * f
        g=f
        # g = self.lin2(emb)
        # s, ea, ev, ef = self.mol2graph(z, pos)
        # mask = self.ef_proj(ef) * ea.unsqueeze(-1)
        s = self.neighbor_emb(z, z_emb, edge_index, f)
        #NE = self.s2v(s, frame, edge_index, g)
        NE1 = self.s2v(s, coord_diff.unsqueeze(-1), edge_index, f)
        scalrization1 = torch.sum(NE1[i].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(NE1[j].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        # scalrization1[:, 2, :] = torch.abs(scalrization1[:, 2, :])
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())
        # calculate scalarization of(k - j)
        # scalar1 = torch.sum(NE*a.unsqueeze(-1), dim=1)
        # scalar2 = torch.sum(NE * b.unsqueeze(-1), dim=1)
        #lin3,lin4 maps vector to multiple frequencies
        scalar3 = (self.lin3(torch.permute(scalrization1, (0,2,1)))+ torch.permute(scalrization1, (0,2,1))[:,:,0].unsqueeze(2))
        scalar4 = (self.lin3(torch.permute(scalrization2, (0,2,1)))+ torch.permute(scalrization2, (0,2,1))[:,:,0].unsqueeze(2))
        #(E,hidden)
        scalar3 = torch.einsum('ijk,ik->ij', [scalar3, frequency])
        #scalar3 = torch.matmul(scalar3,frequency)
        scalar4 = torch.einsum('ijk,ik->ij', [scalar4, frequency])
        #
        scalar5 = (self.lin4(torch.permute(scalrization1, (0, 2, 1))) + torch.permute(scalrization1, (0, 2, 1))).permute((0, 2, 1))
        scalar6 = (self.lin4(torch.permute(scalrization2, (0, 2, 1))) + torch.permute(scalrization2, (0, 2, 1))).permute((0, 2, 1))

        Aij = torch.sum(self.q_proj(scalar5) * self.k_proj(scalar6), dim=1)
        #ev_decay
        edgeweight = torch.cat((scalar3,scalar4, Aij), dim=-1) * rbounds.unsqueeze(-1)

        edgeweight = torch.cat((edgeweight,g), dim=-1)
        # add distance embedding
        edgeweight = torch.cat((edgeweight, emb), dim=-1)
        edgeweight = f * self.dir_proj(edgeweight)
        edgefeature = self.edge_proj((scalrization1.detach() - scalrization2.detach()).permute(0, 2, 1))
        edgefeature[:, 1, :] = torch.abs(edgefeature[:, 1, :].clone())
        edgefeature = torch.einsum('ijk,ik->ij', [edgefeature, frequency])* rbounds.unsqueeze(-1)
        edgefeature = edgefeature #+ edgefeature1
        z_emb = self.interaction1(z_emb, edge_index, edge_attr=edgefeature, edgeweight=edgeweight)
        i = 0
        for interaction in self.interactions:
            z_emb = interaction(z_emb, edge_index, edge_attr=edgefeature, edgeweight=edgeweight)  # + z_emb
            i = i + 1

        s = self.mlp(z_emb.view(-1, self.heads * self.hidden_channels)) + self.emblinfinal(z)
        size = int(batch.max().item() + 1)
        s = scatter(s, batch, dim=-2, dim_size=size, reduce='sum')
        return s


class NeighborEmb(MessagePassing):

    def __init__(self, hid_dim: int, **kwargs):  # ln_emb: bool, **kwargs):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        # self.conv = CFConv()
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim,
                                   elementwise_affine=False)  # kwargs['ln_learnable']) #if ln_emb else nn.Identity()

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        # s_neighbors = self.conv(s_neighbors, mask)
        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class CFConvS2V(MessagePassing):

    def __init__(self, hid_dim: int, **kwargs):  # ln_s2v: bool,
        # lin1_tailact: bool, nolin1: bool=False, **kwargs):
        super(CFConvS2V, self).__init__(aggr='add')
        # super().__init__()
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(  # nn.Identity() if nolin1 else nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
             nn.SiLU())  # kwargs['ln_learnable'])
        # if ln_s2v else nn.Identity(),
        # kwargs["act"] if lin1_tailact else nn.Identity())
        #self.lin2 = nn.Linear(3, hid_dim)

    def forward(self, s, v, edge_index, emb):
        '''
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        v (BN, 3, 1)
        emb (BN, hid_dim)
        '''
        s = self.lin1(s)
        #v = self.lin2(v)
        # sv = s*v
        emb = emb.unsqueeze(1) * v

        # s= s.view(-1, 1, self.hid_dim)
        v = self.propagate(edge_index, x=s, norm=emb)

        # s_neighbors = self.conv(s_neighbors, mask)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class TransformerConv(MessagePassing):
    r"""


    """
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 2,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Sequential(
            nn.Linear(in_channels[0], in_channels[0]),
            nn.LayerNorm(in_channels[0], elementwise_affine=False),
            nn.SiLU())
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()

        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edgeweight: OptTensor = None, emb: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, edgeweight=edgeweight, emb=emb, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            #x_r = self.lin_skip(x[1])
            x_r = x[1]
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, edgeweight: OptTensor, emb: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out *= edgeweight.unsqueeze(1)
        if edge_attr is not None:
            #out += emb* edge_attr
            out += edgeweight.unsqueeze(1) * edge_attr

        #out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')