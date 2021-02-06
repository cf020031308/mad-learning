import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


n_nn = 8
batch_size = 1024


def gpu(x):
    return x
    return x.cuda() if torch.cuda.is_available() else x


def gen_data():
    # 19000101 ~ 20191231
    start = datetime.date(1900, 1, 1)
    delta = datetime.timedelta(days=1)
    fmt = '%Y%m%d'
    inputs = []
    outputs = []
    for i in range(43829):
        outputs.append(start.weekday())
        inputs.append([int(c) for c in start.strftime(fmt)])
        start += delta
    perm = torch.randperm(len(inputs))
    return (
        0.0 + torch.tensor(inputs)[perm],
        torch.tensor(outputs, dtype=int)[perm]
    )


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.pred = gpu(nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, out_feats),
        ))

    def forward(self, x):
        return self.pred(x)


class MAD(nn.Module):
    def __init__(
            self,
            mem,
            test_nns,
            train_nns,
            n_data,
            in_feats,
            hid_feats,
            n_classes,
            n_samples=8,
    ):
        super(self.__class__, self).__init__()
        self.mem = mem
        self.train_nns = train_nns
        self.test_nns = test_nns
        self.n_data = n_data
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.pos = gpu(nn.Sequential(
            nn.Linear(in_feats, hid_feats),
        ))
        self.field = gpu(nn.Sequential(
            nn.Linear(in_feats, hid_feats * n_classes),
        ))
        self.adapt = gpu(nn.Linear(n_classes, n_classes))

    def forward(self, idx, date):
        refs = (self.train_nns if self.training else self.test_nns)[idx]
        diff = self.pos(date).unsqueeze(1) - self.pos(train_dates[refs])
        grad = self.field(date).view(-1, 1, self.hid_feats, self.n_classes)
        logits = (
            self.adapt(self.mem[refs])
            + (diff.unsqueeze(-2) @ grad).squeeze(-2))
        dist = torch.softmax(-diff.norm(dim=-1), dim=-1)
        logits = torch.cat((
            logits, gpu(torch.zeros(logits.shape[0], 1, logits.shape[2]))
        ), dim=1)
        dist = torch.cat((dist, gpu(torch.zeros(dist.shape[0], 1))), dim=1)
        return (logits.transpose(1, 2) @ dist.unsqueeze(-1)).squeeze(-1)


torch.manual_seed(0)
dates, labels = gen_data()
val_num = int(dates.shape[0] * 0.1)
train_dates = gpu(dates[val_num:])
train_labels = gpu(labels[val_num:])
test_dates = gpu(dates[:val_num])
test_labels = gpu(labels[:val_num])
n_data = train_dates.shape[0]
mem = gpu(torch.zeros(n_data, 1 + train_labels.max()))
mem[torch.arange(n_data), train_labels] = 1
test_nns = gpu(torch.empty((val_num, n_nn), dtype=int))
for perm in DataLoader(range(val_num), batch_size=batch_size, shuffle=False):
    test_nns[perm] = (
        test_dates[perm].unsqueeze(1) - train_dates.unsqueeze(0)
    ).norm(dim=-1).topk(n_nn, largest=False).indices
train_nns = gpu(torch.empty((n_data, n_nn), dtype=int))
for perm in DataLoader(range(n_data), batch_size=batch_size, shuffle=False):
    train_nns[perm] = (
        train_dates[perm].unsqueeze(1) - train_dates.unsqueeze(0)
    ).norm(dim=-1).topk(1 + n_nn, largest=False).indices[..., 1:]

pred = MAD(
    mem=mem,
    test_nns=test_nns,
    train_nns=train_nns,
    n_data=n_data,
    in_feats=8,
    hid_feats=1,
    n_classes=7,
    n_samples=n_nn,
)
# pred = MLP(in_feats=8, hid_feats=32, out_feats=7)
params = list(pred.parameters())
print('params:', sum(p.numel() for p in params))
opt = optim.Adam(params)
criterion = nn.CrossEntropyLoss()
batch_size = 1024

for epoch in range(1, 1001):
    pred.train()
    for perm in DataLoader(
            range(train_dates.shape[0]),
            batch_size=batch_size,
            shuffle=True):
        opt.zero_grad()
        # outputs = pred(train_dates[perm])
        outputs = pred(perm, train_dates[perm])
        loss = criterion(outputs, train_labels[perm])
        loss.backward()
        opt.step()

    if epoch % 5:
        continue
    correct, total = 0, 0
    with torch.no_grad():
        pred.eval()
        for perm in DataLoader(
                range(test_dates.shape[0]),
                batch_size=batch_size,
                shuffle=False):
            # outputs = pred(test_dates[perm])
            outputs = pred(perm, test_dates[perm])
            _, predicted = torch.max(outputs.data, 1)
            total += perm.shape[0]
            correct += (predicted == test_labels[perm]).sum().item()
    print('Epoch: %s, Accuracy: %d%%, Loss: %f'
          % (epoch, 100 * correct / total, loss.item()))
