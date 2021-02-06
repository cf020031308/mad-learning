import torch
import torch.nn as nn
from utils import gpu


class MADGraph(nn.Module):
    def __init__(
        self, n_nodes, node_feats, src, dst,
        edge_feats=1, n_samples=8, directed=True,
        nearest=0, sentinel=0,
    ):
        super(self.__class__, self).__init__()
        self.n_nodes = n_nodes
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.n_samples = n_samples
        self.nearest = nearest
        self.sentinel = sentinel
        self.position = nn.Parameter(gpu(
            torch.rand((n_nodes, node_feats))))
        self.src_field = nn.Parameter(gpu(
            torch.rand((n_nodes, node_feats))))
        self.dst_field = (
            self.src_field if not directed
            else nn.Parameter(gpu(torch.rand((n_nodes, node_feats)))))
        self.uncertainty = nn.Parameter(gpu(torch.ones(1, 1) * 5))
        self.edge = None
        edge = -1 * torch.ones(n_nodes, n_nodes)
        edge[src, dst] = 1
        if not directed:
            edge[dst, src] = 1
        self.edge = gpu(edge)

    def forward(self, src, dst=None):
        if dst is None:
            src, dst = src.T
        n = src.shape[0]
        if self.nearest and not self.training:
            mid0 = (
                (self.position[src].unsqueeze(1) - self.position.unsqueeze(2))
                .norm(dim=2)
                .topk(1 + self.nearest, largest=False)
                .indices[:, 1:])
            mid1 = (
                (self.position[dst].unsqueeze(1) - self.position.unsqueeze(2))
                .norm(dim=2)
                .topk(1 + self.nearest, largest=False)
                .indices[:, 1:])
        else:
            mid0 = torch.randint(0, self.n_nodes, (n, self.n_samples))
            mid1 = torch.randint(0, self.n_nodes, (n, self.n_samples))
        srcdiff = self.position[src].unsqueeze(1) - self.position[mid0]
        logits1 = (
            (srcdiff.unsqueeze(2)
             @ self.dst_field[dst].unsqueeze(1).unsqueeze(3)
             ).squeeze(2).squeeze(2)
            + self.uncertainty * self.edge[mid0, dst.unsqueeze(1)]
        )
        dstdiff = self.position[dst].unsqueeze(1) - self.position[mid1]
        logits2 = (
            (dstdiff.unsqueeze(2)
             @ self.src_field[src].unsqueeze(1).unsqueeze(3)
             ).squeeze(2).squeeze(2)
            + self.uncertainty * self.edge[src.unsqueeze(1), mid1]
        )
        logits = torch.cat((logits1, logits2), dim=1)
        dist = torch.cat((srcdiff, dstdiff), dim=1).norm(dim=2)
        if self.sentinel:
            logits = torch.cat(
                (logits, gpu(torch.zeros(n, self.sentinel))), dim=1)
            dist = torch.cat(
                (dist, gpu(torch.ones(n, self.sentinel))), dim=1)
        return (
            logits.unsqueeze(1) @ torch.softmax(1 - dist, dim=1).unsqueeze(2)
        ).squeeze()
