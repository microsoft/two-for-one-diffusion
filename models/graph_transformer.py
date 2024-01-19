"""
This code has been imported from:
(lucidrains - graph-transformer-pytorch)
https://github.com/lucidrains/graph-transformer-pytorch
"""
import torch
from torch import nn
from typing import List, Optional
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils import center_zero
from contextlib import nullcontext
from torch import nn, einsum
from einops import rearrange, repeat


class GraphTransformer(nn.Module):
    """
    Graph neural network moldule.
    """

    def __init__(
        self,
        num_beads,
        hidden_nf,
        device="cpu",
        n_layers=4,
        use_intrinsic_coords: bool = False,
        use_abs_coords: bool = True,
        use_distances: bool = True,
        conservative: bool = True,
    ):
        """_summary_

        Args:
            num_beads (_type_): _description_
            hidden_nf (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            n_layers (int, optional): _description_. Defaults to 4.
            use_intrinsic_coords (bool, optional): _description_. Defaults to False.
            use_abs_coords (bool, optional): _description_. Defaults to True.
            use_distances (bool, optional): _description_. Defaults to True.
            conservative (bool, optional): _description_. Defaults to True.
        """
        super(GraphTransformer, self).__init__()
        self.device = device
        self.use_intrinsic_coords = use_intrinsic_coords
        self.use_distances = use_distances
        self.use_abs_coords = use_abs_coords
        self.conservative = conservative

        in_node_nf = num_beads + 1 + use_abs_coords * 3
        in_edge_nf = (
            3 * use_intrinsic_coords
            + use_distances
            + 1 * (not use_intrinsic_coords) * (not use_distances)
        )
        self.node_embedding = nn.Linear(in_node_nf, hidden_nf)
        self.edge_embedding = nn.Linear(in_edge_nf, hidden_nf)

        if self.conservative:
            self.node_decoder = nn.Linear(hidden_nf, 1)
        else:
            self.node_decoder = nn.Linear(hidden_nf, 3)

        self.graphtransformer = GraphTransformerLucid(
            dim=hidden_nf,
            depth=n_layers,
            edge_dim=hidden_nf,  # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards=True,  # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual=True,  # to use the gated residual to prevent over-smoothing
        )

        self.to(self.device)

    def forward(
        self,
        x,
        h,
        t,
        return_energy=False,
        alphas: Optional[torch.Tensor] = None,
    ):
        # alphas: Optional tensor of shape [Batch_size,]
        # Center at 0 to be translation invariant
        x = center_zero(x)
        x = x.requires_grad_(requires_grad=True)
        with torch.enable_grad() if self.conservative else nullcontext():
            bs, n_nodes, _ = x.shape
            t = t.reshape(-1, 1, 1).repeat(1, x.shape[1], 1)
            h = h.unsqueeze(0).repeat(bs, 1, 1).to(x.device)

            # Compute edge attributes if necessary
            edge_attr = self.get_edge_attr(x)
            edge_attr = self.edge_embedding(edge_attr)

            # Concatenate node inputs
            if self.use_abs_coords:
                nodes = torch.cat((h, x, t), dim=2)
            else:
                nodes = torch.cat((h, t), dim=2)
            nodes = self.node_embedding(nodes)
            mask = torch.ones(x.size(0), x.size(1)).bool().to(x.device)
            nodes, _ = self.graphtransformer(nodes, edge_attr, mask=mask)
            output = self.node_decoder(nodes)
            if self.conservative:
                energy = output
                if return_energy:
                    return energy
                forces = compute_forces(energy, x, self.training)
            else:
                forces = output
            return forces

    def get_edge_attr(self, x):
        if self.use_distances and not self.use_intrinsic_coords:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            dist = torch.sum(
                diff**2, dim=3, keepdim=True
            )  # torch.norm(diff, dim=3).unsqueeze(3)
            return dist
        elif self.use_intrinsic_coords and not self.use_distances:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            return diff
        elif self.use_intrinsic_coords and self.use_distances:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            dist = torch.sum(
                diff**2, dim=3, keepdim=True
            )  # torch.norm(diff, dim=3).unsqueeze(3)
            return torch.cat([diff, dist], dim=3)
        else:
            bs, n_nodes, _ = x.size()
            return torch.zeros(bs, n_nodes, n_nodes, 1).to(x.device).detach()


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training=True
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        only_inputs=True,  # Diff only w.r.t. inputs
        allow_unused=True,
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        raise Exception("Gradient after computing forces is None.")
    return -1 * gradient


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


List = nn.ModuleList

# normalizations


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# gated residual


class Residual(nn.Module):
    def forward(self, x, res):
        return x + res


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, 1, bias=False), nn.Sigmoid())

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):
    # NOTE: we don't use positional embeddings
    def __init__(self, dim, pos_emb=None, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask=None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim=-1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(
            lambda t: rearrange(t, "b ... (h d) -> (b h) ... d", h=h), (q, k, v, e_kv)
        )

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, "b j d -> b () j d "), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum("b i d, b i j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# optional feedforward


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim)
    )


# classes


class GraphTransformerLucid(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        gated_residual=True,
        with_feedforwards=False,
        norm_edges=False,
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        pos_emb = None

        for _ in range(depth):
            self.layers.append(
                List(
                    [
                        List(
                            [
                                PreNorm(
                                    dim,
                                    Attention(
                                        dim,
                                        pos_emb=pos_emb,
                                        edge_dim=edge_dim,
                                        dim_head=dim_head,
                                        heads=heads,
                                    ),
                                ),
                                GatedResidual(dim),
                            ]
                        ),
                        List([PreNorm(dim, FeedForward(dim)), GatedResidual(dim)])
                        if with_feedforwards
                        else None,
                    ]
                )
            )

    def forward(self, nodes, edges, mask=None):
        edges = self.norm_edges(edges)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, edges, mask=mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges


if __name__ == "__main__":
    import time

    # Init variables
    n_nodes = 10
    hidden_nf = 256
    bs = 128

    # Init model
    model = GraphTransformer(
        n_nodes, hidden_nf=hidden_nf, n_layers=5, conservative=False
    ).cuda()

    # Init parameters
    x = torch.ones(bs, n_nodes, 3).cuda()
    h = torch.ones(n_nodes, n_nodes).cuda()
    t = torch.ones(bs).cuda()

    # Run model
    t1 = time.time()
    n_iterations = 10
    for i in range(n_iterations):
        forces = model(x, h, t)
    t2 = time.time()

    # Print output shape
    print(forces.shape)
    print(f"Elapsed time {(t2 - t1)/n_iterations}")
