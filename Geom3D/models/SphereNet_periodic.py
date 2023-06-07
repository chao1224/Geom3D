from math import pi as PI
from math import sqrt

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .DimeNet import ResidualLayer
from .SphereNet_utils import angle_emb, dist_emb, torsion_emb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REDUCE = "mean"


class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size_angle, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(
            num_spherical * num_spherical * num_radial,
            basis_emb_size_torsion,
            bias=False,
        )
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0), reduce=REDUCE)
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == "zeros":
            self.lin.weight.data.fill_(0)
        if self.output_init == "GlorotOrthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, dim_size=None):
        if dim_size == None:
            dim_size = i.max().item() + 1
        _, e2 = e
        v = scatter(e2, i, dim=0, dim_size=dim_size, reduce=REDUCE)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0, reduce=REDUCE)
        return u


class SphereNetPeriodic(torch.nn.Module):
    r"""
     The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>`_ paper.

    Args:
        energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
        num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
        out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
        int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
        basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
        basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
        basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
        out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
        num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
        envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion. (default: :obj:`swish`)
        output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)

    """

    def __init__(
        self,
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        output_init="GlorotOrthogonal",
    ):
        super(SphereNetPeriodic, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(
            hidden_channels,
            out_emb_channels,
            out_channels,
            num_output_layers,
            act,
            output_init,
        )
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList(
            [
                update_v(
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_es = torch.nn.ModuleList(
            [
                update_e(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size_dist,
                    basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def triplets(self, pos, edge_index, num_nodes, use_torsion=False):
        """
        Compute the diatance, angle, and torsion from geometric information.
        Args:
            pos: Geometric information for every node in the graph.
            edgee_index: Edge index of the graph.
            number_nodes: Number of nodes in the graph.
            use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
        """
        j, i = edge_index  # j->i

        # Calculate distances. # number of edges
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        value = torch.arange(j.size(0), device=j.device)
        adj_t = SparseTensor(
            row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[j]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = i.repeat_interleave(num_triplets)
        idx_j = j.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        # Calculate angles. 0 to pi
        pos_ji = pos[idx_i] - pos[idx_j]
        pos_jk = pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
        b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
        angle = torch.atan2(b, a)

        idx_batch = torch.arange(len(idx_i), device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = (
            idx_i_t[mask],
            idx_j_t[mask],
            idx_k_t[mask],
            idx_k_n[mask],
            idx_batch_t[mask],
        )

        # Calculate torsions.
        if use_torsion:
            pos_j0 = pos[idx_k_t] - pos[idx_j_t]
            pos_ji = pos[idx_i_t] - pos[idx_j_t]
            pos_jk = pos[idx_k_n] - pos[idx_j_t]
            dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
            plane1 = torch.cross(pos_ji, pos_j0)
            plane2 = torch.cross(pos_ji, pos_jk)
            a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
            b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
            torsion1 = torch.atan2(b, a)  # -pi to pi
            torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
            torsion = scatter(torsion1, idx_batch_t, reduce="min")

            return dist, angle, torsion, i, j, idx_kj, idx_ji

        else:
            return dist, angle, i, j, idx_kj, idx_ji

    def forward(self, z, pos, edge_index, batch):
        if self.energy_and_force:
            pos.requires_grad_()
        num_nodes = z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = self.triplets(
            pos, edge_index, num_nodes, use_torsion=True
        )

        emb = self.emb(dist, angle, torsion, idx_kj)
        node_num = batch.size()[0]

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, dim_size=node_num)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0, reduce=REDUCE)), v, batch)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i, dim_size=node_num)
            u = update_u(u, v, batch)

        return u

    def forward_with_gathered_index(self, z, pos, edge_index, batch, periodic_index_mapping):
        if self.energy_and_force:
            pos.requires_grad_()
        num_nodes = pos.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = self.triplets(pos, edge_index, num_nodes, use_torsion=True)
        
        i = periodic_index_mapping[i]
        j = periodic_index_mapping[j]

        emb = self.emb(dist, angle, torsion, idx_kj)
        node_num = batch.size()[0]

        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, dim_size=node_num)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0, reduce=REDUCE)), v, batch)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i, dim_size=node_num)
            u = update_u(u, v, batch)

        return u
