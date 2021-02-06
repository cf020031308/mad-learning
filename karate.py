import dgl
import torch
import torch.optim as optim

from mad import MADGraph


dataset = dgl.data.KarateClubDataset()
g = dataset[0]
n_nodes = g.num_nodes()
src, dst = g.edges()
labels = g.ndata['label'].tolist()
self_connected = False

mad = MADGraph(n_nodes, 2, src, dst, n_samples=2, directed=False, sentinel=1)
params = list(mad.parameters())
print('params:', sum(p.numel() for p in params))
opt = optim.Adam(params, lr=0.005)
for ep in range(1000):
    mad.train()
    opt.zero_grad()
    p_pos = torch.sigmoid(mad(src, dst))
    if self_connected:
        p_pos = torch.cat((
            p_pos,
            p_pos,
            torch.sigmoid(mad(src, src)),
            torch.sigmoid(mad(dst, dst)),
        ), dim=-1)
    p_neg = torch.cat((
        torch.sigmoid(mad(src, torch.randint(0, n_nodes, (src.shape[0], )))),
        torch.sigmoid(mad(torch.randint(0, n_nodes, (dst.shape[0], )), dst))
    ), dim=-1)
    loss = (
        -torch.log(1e-5 + 1 - p_neg).mean()
        - torch.log(1e-5 + p_pos).mean())
    loss.backward()
    opt.step()
    r = torch.rand(n_nodes, 2)
    # pos = mad.src_field.data
    # pos[:, 0] = 0
    # mad.position.data = pos
    print(ep, p_pos.mean().item(), p_neg.mean().item())
pos = mad.position.data.tolist()
field = mad.src_field.data.tolist()
with open('karate-encoding.csv', 'w') as file:
    for i in range(n_nodes):
        file.write(','.join(
            str(c) for c in [
                i, labels[i], pos[i][0], pos[i][1], field[i][0], field[i][1]]))
        file.write('\n')
with open('karate-edge.csv', 'w') as file:
    for u, v in zip(src, dst):
        file.write(','.join(
            str(c) for c in [
                pos[u][0], pos[u][1], pos[v][0], pos[v][1]]))
        file.write('\n')
