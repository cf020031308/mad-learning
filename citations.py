import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import dgl.nn
from sklearn import metrics


g_data_name = 'pubmed'  # cora | citeseer | pubmed
g_toy = False
g_dim = 32
n_samples = 8
total_epoch = 200
lr = 0.005
if g_toy:
    g_dim = g_dim // 2
elif g_data_name == 'pubmed':
    g_dim, n_samples, total_epoch, lr = 64, 64, 2000, 0.001


def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def cpu(x):
    return x.cpu() if torch.cuda.is_available() else x


def ip(x, y):
    return (x.unsqueeze(-2) @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class MAD(nn.Module):
    def __init__(
            self, in_feats, n_nodes, node_feats,
            n_samples, mem, feats, gather2neighbor=False,
    ):
        super(self.__class__, self).__init__()
        self.n_nodes = n_nodes
        self.node_feats = node_feats
        self.n_samples = n_samples
        self.mem = mem
        self.feats = feats
        self.gather2neighbor = gather2neighbor
        self.f = gpu(nn.Linear(in_feats, node_feats))
        self.g = (
            None if gather2neighbor else gpu(nn.Linear(in_feats, node_feats)))
        self.adapt = gpu(nn.Linear(1, 1))
        self.nn = None

    def nns(self, src, dst):
        if self.nn is None:
            n = self.n_samples
            self.nn = gpu(torch.empty((self.n_nodes, n), dtype=int))
            for perm in DataLoader(
                    range(self.n_nodes), 64, shuffle=False):
                self.nn[perm] = (
                    self.feats[perm].unsqueeze(1) - self.feats.unsqueeze(0)
                ).norm(dim=-1).topk(1 + n, largest=False).indices[..., 1:]
        return self.nn[src], self.nn[dst]

    def recall(self, src, dst):
        if self.mem is None:
            return 0
        return self.adapt(
            (0.0 + self.mem[src, dst]).unsqueeze(-1)).squeeze(-1)

    def forward(self, src, dst):
        n = src.shape[0]
        feats = self.feats
        g = self.f if self.gather2neighbor else self.g
        mid0 = torch.randint(0, self.n_nodes, (n, self.n_samples))
        mid1 = torch.randint(0, self.n_nodes, (n, self.n_samples))
        # mid0, mid1 = self.nns(src, dst)
        srcdiff = self.f(feats[src]).unsqueeze(1) - self.f(feats[mid0])
        logits1 = (
            ip(srcdiff, g(feats[dst]).unsqueeze(1))
            + self.recall(mid0, dst.unsqueeze(1))
        )
        dstdiff = self.f(feats[dst]).unsqueeze(1) - self.f(feats[mid1])
        logits2 = (
            ip(dstdiff, g(feats[src]).unsqueeze(1))
            + self.recall(src.unsqueeze(1), mid1)
        )
        logits = torch.cat((logits1, logits2), dim=1)
        dist = torch.cat((srcdiff, dstdiff), dim=1).norm(dim=2)
        logits = torch.cat((
            logits, gpu(torch.zeros(n, self.n_samples))), dim=1)
        dist = torch.cat((
            dist, gpu(torch.ones(n, self.n_samples))), dim=1)
        return torch.sigmoid(ip(logits, torch.softmax(-dist, dim=1)))


dataset = (
    dgl.data.CoraGraphDataset() if g_data_name == 'cora'
    else dgl.data.CiteseerGraphDataset() if g_data_name == 'citeseer'
    else dgl.data.PubmedGraphDataset())
graph = dataset[0]
src, dst = graph.edges()
node_features = gpu(graph.ndata['feat'])
node_labels = gpu(graph.ndata['label'])
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_nodes = graph.num_nodes()
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)
flt = src <= dst
src = src[flt]
dst = dst[flt]

adj = gpu(torch.zeros((n_nodes, n_nodes), dtype=bool))
adj[src, dst] = 1
adj[dst, src] = 1
if g_toy:
    mem = None
    train_src = gpu(src)
    train_dst = gpu(dst)
    mlp = gpu(nn.Linear(g_dim, n_labels))
    params = list(mlp.parameters())
    print('mlp params:', sum(p.numel() for p in params))
    mlp_opt = optim.Adam(params, lr=lr)
else:
    n = src.shape[0]
    perm = torch.randperm(n)
    val_num = int(0.05 * n)
    test_num = int(0.1 * n)
    train_src = gpu(src[perm[val_num + test_num:]])
    train_dst = gpu(dst[perm[val_num + test_num:]])
    val_src = gpu(src[perm[:val_num]])
    val_dst = gpu(dst[perm[:val_num]])
    test_src = gpu(src[perm[val_num:val_num + test_num]])
    test_dst = gpu(dst[perm[val_num:val_num + test_num]])
    train_src, train_dst = (
        torch.cat((train_src, train_dst)),
        torch.cat((train_dst, train_src)))
    val_src, val_dst = (
        torch.cat((val_src, val_dst)),
        torch.cat((val_dst, val_src)))
    test_src, test_dst = (
        torch.cat((test_src, test_dst)),
        torch.cat((test_dst, test_src)))
    mem = gpu(torch.zeros((n_nodes, n_nodes), dtype=bool))
    mem[train_src, train_dst] = 1

total_aucs = []
total_aps = []
for run in range(10):
    torch.manual_seed(run)
    mad = MAD(
        in_feats=n_features,
        n_nodes=n_nodes,
        node_feats=g_dim,
        n_samples=n_samples,
        mem=mem,
        feats=node_features,
        gather2neighbor=g_toy,
    )
    params = list(mad.parameters())
    print('params:', sum(p.numel() for p in params))
    opt = optim.Adam(params, lr=0.01)
    best_aucs = [0, 0]
    best_aps = [0, 0]
    best_accs = [0, 0]
    for epoch in range(1, total_epoch + 1):
        mad.train()
        for perm in DataLoader(
                range(train_src.shape[0]), batch_size=1024, shuffle=True):
            opt.zero_grad()
            p_pos = mad(train_src[perm], train_dst[perm])
            neg_src = gpu(torch.randint(0, n_nodes, (perm.shape[0], )))
            neg_dst = gpu(torch.randint(0, n_nodes, (perm.shape[0], )))
            idx = ~(mem[neg_src, neg_dst])
            p_neg = mad(neg_src[idx], neg_dst[idx])
            loss = (
                -torch.log(1e-5 + 1 - p_neg).mean()
                - torch.log(1e-5 + p_pos).mean()
            )
            loss.backward()
            opt.step()

        if epoch % 10:
            continue

        if g_toy:
            with torch.no_grad():
                embed = mad.f(node_features)
            for i in range(100):
                mlp.train()
                mlp_opt.zero_grad()
                logits = mlp(embed)
                loss = F.cross_entropy(
                    logits[train_mask], node_labels[train_mask])
                loss.backward()
                mlp_opt.step()
            with torch.no_grad():
                logits = mlp(embed)
                _, indices = torch.max(logits[valid_mask], dim=1)
                labels = node_labels[valid_mask]
                v_acc = torch.sum(indices == labels).item() * 1.0 / len(labels)
                _, indices = torch.max(logits[test_mask], dim=1)
                labels = node_labels[test_mask]
                t_acc = torch.sum(indices == labels).item() * 1.0 / len(labels)
            if v_acc > best_accs[0]:
                best_accs = [v_acc, t_acc]
                print(epoch, 'acc:', v_acc, t_acc)
            continue

        with torch.no_grad():
            mad.eval()
            aucs = []
            aps = []
            for src, dst in ((val_src, val_dst), (test_src, test_dst)):
                p_pos = mad(src, dst)

                n = src.shape[0]
                perm = torch.randperm(n * 2)
                neg_src = torch.cat((
                    src, gpu(torch.randint(0, n_nodes, (n, )))
                ))[perm]
                neg_dst = torch.cat((
                    gpu(torch.randint(0, n_nodes, (n, ))), dst
                ))[perm]
                idx = ~(adj[neg_src, neg_dst])
                neg_src = neg_src[idx][:n]
                neg_dst = neg_dst[idx][:n]
                p_neg = mad(neg_src, neg_dst)

                y_true = cpu(torch.cat((p_pos * 0 + 1, p_neg * 0)))
                y_score = cpu(torch.cat((p_pos, p_neg)))
                fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                ap = metrics.average_precision_score(y_true, y_score)
                aucs.append(auc)
                aps.append(ap)
            if aucs[0] > best_aucs[0]:
                best_aucs = aucs
                print(epoch, 'auc:', aucs)
            if aps[0] > best_aps[0]:
                best_aps = aps
                print(epoch, 'ap:', aps)
    print(run, 'best auc:', best_aucs)
    print(run, 'best ap:', best_aucs)
    print(run, 'best acc (toy):', best_accs)
    total_aucs.append(best_aucs[1])
    total_aps.append(best_aps[1])
total_aucs = torch.tensor(total_aucs)
total_aps = torch.tensor(total_aps)
print('auc mean:', total_aucs.mean().item(), 'std:', total_aucs.std().item())
print('ap mean:', total_aps.mean().item(), 'std:', total_aps.std().item())
