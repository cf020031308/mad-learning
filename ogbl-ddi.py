# !pip install ogb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset, Evaluator

from logger import Logger


hasGPU = torch.cuda.is_available()


def gpu(x):
    return x.cuda() if hasGPU else x


class MADGraph(nn.Module):
    def __init__(
            self,
            n_nodes,
            node_feats,
            src,
            dst,
            in_feats=32,
            n_heads=4,
            n_samples=256,
            n_sentinels=8,
            memory=-1,
            softmin=True,
            n_nearest=0
    ):
        super(self.__class__, self).__init__()
        self.n_nodes = n_nodes
        self.node_feats = node_feats
        self.n_samples = n_samples
        self.n_heads = n_heads
        self.n_sentinels = n_sentinels
        self.memory = memory
        self.softmin = softmin
        self.n_nearest = n_nearest

        self.pos = nn.Parameter(gpu(
            torch.rand((n_heads, n_nodes, node_feats))))
        self.field = nn.Parameter(gpu(
            torch.rand((n_heads, n_nodes, node_feats))))
        self.uncertainty = nn.Parameter(gpu(torch.ones(1, 1, 1) * 5))

        edge = -1 * torch.ones(n_nodes, n_nodes)
        edge[src, dst] = 1
        edge[dst, src] = 1
        self.edge = gpu(edge)

    def forward(self, edge):
        src, dst = edge.T
        n = edge.shape[0]
        mid0 = torch.randint(
            0, self.n_nodes, (self.n_heads, n, self.n_samples))
        mid1 = torch.randint(
            0, self.n_nodes, (self.n_heads, n, self.n_samples))
        if self.n_nearest and not self.training:
            mid0 = (
                self.pos[:, src].unsqueeze(2)
                - self.pos.unsqueeze(1)
            ).norm(dim=3).topk(
                1 + self.n_nearest, largest=False).indices[:, :, 1:]
            mid1 = (
                self.pos[:, dst].unsqueeze(2)
                - self.pos.unsqueeze(1)
            ).norm(dim=3).topk(
                1 + self.n_nearest, largest=False).indices[:, :, 1:]
        srcdiff = self.pos[:, src].unsqueeze(2) - self.pos[
            torch.arange(self.n_heads).unsqueeze(1).unsqueeze(2), mid0]
        logits1 = (
            (
                srcdiff.unsqueeze(3)
                @ (self.field[:, dst].unsqueeze(2).unsqueeze(4))
            ).squeeze(3).squeeze(3)
            + [
                0,
                self.uncertainty,
                self.uncertainty * self.edge[
                    mid0, dst.unsqueeze(0).unsqueeze(2)],
            ][self.memory]
        )
        dstdiff = self.pos[:, dst].unsqueeze(2) - self.pos[
            torch.arange(self.n_heads).unsqueeze(1).unsqueeze(2), mid1]
        logits2 = (
            (
                dstdiff.unsqueeze(3)
                @ (self.field[:, src].unsqueeze(2).unsqueeze(4))
            ).squeeze(3).squeeze(3)
            + [
                0,
                self.uncertainty,
                self.uncertainty * self.edge[
                    src.unsqueeze(0).unsqueeze(2), mid1],
            ][self.memory]
        )
        logits = torch.cat((logits1, logits2), dim=2)
        if not self.softmin:
            return logits.mean(2).mean(0)
        dist = torch.cat((srcdiff, dstdiff), dim=2).norm(dim=3)
        if self.n_sentinels:
            logits = torch.cat((
                logits, gpu(torch.zeros(self.n_heads, n, self.n_sentinels))
            ), dim=2)
            dist = torch.cat((
                dist, gpu(torch.ones(self.n_heads, n, self.n_sentinels))
            ), dim=2)
        return (
            logits.unsqueeze(2) @ torch.softmax(1-dist, dim=2).unsqueeze(3)
        ).squeeze(2).squeeze(2).mean(0)


def sample(edge, batch_size=1024):
    for perm in DataLoader(range(edge.shape[0]), batch_size, shuffle=True):
        yield edge[perm]


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (MADGraph)')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4 * 1024)
    parser.add_argument('--dim', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--nearest', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sentinels', type=int, default=8)
    parser.add_argument('--memory', type=str, default='all')
    parser.add_argument('--softmin', type=bool, default=True)
    parser.add_argument('--output_csv', type=str, default='')
    args = parser.parse_args()
    print(args)

    DNAME = 'ogbl-ddi'
    dataset = LinkPropPredDataset(name=DNAME)
    graph = dataset[0]
    n_nodes = graph['num_nodes']

    data = dataset.get_edge_split()
    for group in 'train valid test'.split():
        if group in data:
            sets = data[group]
            for key in ('edge', 'edge_neg'):
                if key in sets:
                    sets[key] = gpu(torch.from_numpy(sets[key]))
    data['eval_train'] = {'edge': data['train']['edge'][
        torch.randperm(data['train']['edge'].shape[0])[
            :data['valid']['edge'].shape[0]]]}

    model = MADGraph(
        n_nodes=n_nodes,
        node_feats=args.dim,
        src=data['train']['edge'][:, 0],
        dst=data['train']['edge'][:, 1],
        n_samples=args.samples,
        n_heads=args.heads,
        n_sentinels=args.sentinels,
        memory=['none', 'stat', 'all'].index(args.memory),
        softmin=args.softmin,
        n_nearest=args.nearest,
    )
    params = [p for net in [model] for p in net.parameters()]
    print('params:', sum(p.numel() for p in params))

    evaluator = Evaluator(name=DNAME)
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    for run in range(args.runs):
        torch.manual_seed(args.seed + run)
        opt = optim.Adam(params, lr=args.lr)

        torch.nn.init.xavier_uniform_(model.pos.data)
        torch.nn.init.xavier_uniform_(model.field.data)
        model.uncertainty.data = model.uncertainty.data * 0 + 1

        for epoch in range(1, args.epochs + 1):
            model.train()
            for chunk in sample(data['train']['edge'], args.batch_size):
                opt.zero_grad()
                p_edge = torch.sigmoid(model(chunk))
                edge_neg_chunk = gpu(
                    torch.randint(0, n_nodes, chunk.shape))
                p_edge_neg = torch.sigmoid(model(edge_neg_chunk))
                loss = (
                    -torch.log(1e-5 + 1 - p_edge_neg).mean()
                    - torch.log(1e-5 + p_edge).mean())
                loss.backward()
                opt.step()

            if epoch % args.eval_steps:
                continue

            with torch.no_grad():
                model.eval()
                p_train = torch.cat([
                    model(chunk) for chunk in sample(
                        data['eval_train']['edge'], args.batch_size)])
                n_train = torch.cat([
                    model(chunk) for chunk in sample(
                        data['valid']['edge_neg'], args.batch_size)])
                p_valid = torch.cat([
                    model(chunk) for chunk in sample(
                        data['valid']['edge'], args.batch_size)])
                n_valid = n_train
                p_test = torch.cat([
                    model(chunk) for chunk in sample(
                        data['test']['edge'], args.batch_size)])
                n_test = torch.cat([
                    model(chunk) for chunk in sample(
                        data['test']['edge_neg'], args.batch_size)])
                for K in [10, 20, 30]:
                    evaluator.K = K
                    key = f'Hits@{K}'
                    h_train = evaluator.eval({
                        'y_pred_pos': p_train,
                        'y_pred_neg': n_train,
                    })[f'hits@{K}']
                    h_valid = evaluator.eval({
                        'y_pred_pos': p_valid,
                        'y_pred_neg': n_valid,
                    })[f'hits@{K}']
                    h_test = evaluator.eval({
                        'y_pred_pos': p_test,
                        'y_pred_neg': n_test,
                    })[f'hits@{K}']
                    loggers[key].add_result(run, (h_train, h_valid, h_test))
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * h_train:.2f}%, '
                          f'Valid: {100 * h_valid:.2f}%, '
                          f'Test: {100 * h_test:.2f}%')
                print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == '__main__':
    main()
