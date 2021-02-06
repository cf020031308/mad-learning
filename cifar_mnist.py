import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


n_nn = 32
batch_size = 256
hid_feats = 12
dataset, chans, stride, normalizer = {
    'cifar10': (
        torchvision.datasets.CIFAR10,
        3,
        5,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ),
    'cifar100': (
        torchvision.datasets.CIFAR100,
        3,
        5,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ),
    'mnist': (
        torchvision.datasets.MNIST,
        1,
        3,
        transforms.Normalize((0.1307, ), (0.3081, )),
    ),
    'kmnist': (
        torchvision.datasets.KMNIST,
        1,
        3,
        transforms.Normalize((0.1307, ), (0.3081, )),
    ),
}[sys.argv[1]]
method = sys.argv[2]
assert method in 'conv resnet mad-conv mad-resnet'.split()


def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

# dev = torch.device('cuda:6')
# gpu = lambda x: x.to(dev)


def batch(data, batch_size, shuffle=True):
    for perm in DataLoader(
            range(data.shape[0]), batch_size=batch_size, shuffle=shuffle):
        yield perm


class MAD(nn.Module):
    def __init__(
        self,
        mem,
        conv,
        in_feats,
        n_data,
        n_classes,
        hid_feats,
        n_samples=8,
    ):
        super(self.__class__, self).__init__()
        self.mem = mem
        self.n_data = n_data
        self.n_classes = n_classes
        self.conv = conv
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.n_samples = n_samples
        self.f = gpu(nn.Linear(in_feats, hid_feats))
        self.pos = lambda x: self.f(self.conv(x))
        self.g = gpu(nn.Linear(in_feats, hid_feats * n_classes))
        self.field = lambda x: self.g(self.conv(x))
        self.m = gpu(nn.Linear(n_classes, n_classes))

    def forward(self, idx, img):
        n = idx.shape[0]
        refs = gpu(torch.randint(0, self.n_data, (n, self.n_samples)))
        # n x a x d
        x = self.pos(img)
        diff = (
            x.unsqueeze(1)
            - self.pos(train[refs.flatten()]).view(n, -1, self.hid_feats))
        # n x 1 x d x c
        grad = self.field(img).view(-1, 1, self.hid_feats, self.n_classes)
        # n x a x c
        logits = (
            self.m(self.mem[refs])
            + (diff.unsqueeze(-2) @ grad).squeeze(-2))
        # n x a
        dist = torch.softmax(-diff.norm(dim=-1), dim=-1)
        logits = torch.cat((
            logits, gpu(torch.zeros(logits.shape[0], 1, logits.shape[2]))
        ), dim=1)
        dist = torch.cat((dist, gpu(torch.zeros(dist.shape[0], 1))), dim=1)
        ret = (logits.transpose(1, 2) @ dist.unsqueeze(-1)).squeeze(-1)
        return ret


datadir = '/kaggle/working'
if not os.path.exists(datadir):
    dirname = os.path.dirname(os.path.abspath(__file__))
    datadir = '%s/dataset' % dirname
transform = transforms.Compose([transforms.ToTensor(), normalizer])
trainset = dataset(
    root=datadir, train=True, download=True, transform=transform)
train = gpu(torch.cat([x.unsqueeze(0) for x, _ in trainset], dim=0))
train_labels = gpu(torch.tensor(trainset.targets, dtype=int))
testset = dataset(
    root=datadir, train=False, download=True, transform=transform)
test = gpu(torch.cat([x.unsqueeze(0) for x, _ in testset], dim=0))
test_labels = gpu(torch.tensor(testset.targets, dtype=int))
n_data = len(trainset)
n_classes = len(trainset.classes)
mem = gpu(torch.zeros(n_data, n_classes))
mem[torch.arange(n_data), train_labels] = 1
conv = gpu(nn.Sequential(
    nn.Conv2d(chans, 6, stride),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, stride),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(-3),
))
in_feats = 16 * 5 * 5

if method == 'conv':
    model = gpu(nn.Sequential(
        conv,
        nn.Linear(in_feats, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, n_classes),
    ))
    forward = model.forward
    model.forward = lambda idx, img: forward(img)
elif method == 'mad-conv':
    model = MAD(
        mem=mem,
        n_data=n_data,
        hid_feats=hid_feats,
        n_classes=n_classes,
        n_samples=n_nn,
        conv=conv,
        in_feats=in_feats,
    )
else:
    conv = gpu(torchvision.models.resnet18(pretrained=True))
    conv.fc = gpu(nn.Identity())
    in_feats = 512
    if method == 'resnet':
        conv.fc = gpu(nn.Linear(in_feats, n_classes))
        model = conv
        forward = model.forward
        model.forward = lambda idx, img: forward(img)
    elif method == 'mad-resnet':
        model = MAD(
            mem=mem,
            n_data=n_data,
            hid_feats=hid_feats,
            n_classes=n_classes,
            n_samples=n_nn,
            conv=conv,
            in_feats=in_feats,
        )
    else:
        pass
params = list(model.parameters())
print('params:', sum(p.numel() for p in params))
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(params)

for epoch in range(1, 51):
    model.train()
    for perm in batch(train, batch_size):
        opt.zero_grad()
        outputs = model(perm, train[perm])
        loss = criterion(outputs, train_labels[perm])
        loss.backward()
        opt.step()

    if epoch % 5:
        continue
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for perm in batch(test, batch_size):
            outputs = model(perm, test[perm])
            _, predicted = torch.max(outputs.data, 1)
            total += perm.shape[0]
            correct += (predicted == test_labels[perm]).sum().item()
    print('Epoch: %s, Accuracy: %d%%' % (epoch, 100 * correct / total))
