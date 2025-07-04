import argparse
import os
import sys

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import wandb

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.dataloader import TenhouDataset, process_data
from model.models import DiscardModel


@torch.no_grad()
def model_test(model, dataset: TenhouDataset):
    acc = 0
    total = 0
    length = len(dataset)
    while len(dataset) > 0:
        data = dataset()
        if len(data) == 0:
            break
        features, labels = process_data(data, label_trans=lambda x: x // 4)
        labels = labels.type(torch.LongTensor)
        features, labels = features.to(device), labels.to(device)
        output = model(features).softmax(1)
        available = features[:, :4].sum(1) != 0
        # available = (features[:, :16] * features[:, 86: 90].repeat_interleave(4, 1)).sum(1) != 0
        pred = (output * available).argmax(1)
        correct = (pred == labels).sum()
        acc += correct
        total += len(labels)
        print(f"Testing {length - len(dataset)} / {length} acc: {correct.item() / len(labels):.3f}".center(50, '-'), end='\r')
    dataset.reset()
    return acc / total


mode = 'discard'
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', '-n', default=50, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
args = parser.parse_args()

experiment = wandb.init(project='Mahjong', resume='allow', name=f'train-{mode}-sl')
train_set = TenhouDataset(data_dir='train_data', batch_size=128, mode=mode, target_length=2)
test_set = TenhouDataset(data_dir='train_data', batch_size=128, mode=mode, target_length=2)
length = len(train_set)
len_train = int(0.8 * length)
train_set.data_files, test_set.data_files = train_set.data_files[:len_train], train_set.data_files[len_train:]

num_layers = args.num_layers
in_channels = 291
model = DiscardModel(num_layers=num_layers, in_channels=in_channels)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
optim = Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
loss_fcn = CrossEntropyLoss()
epochs = args.epochs

os.makedirs(f'output/{mode}-model/checkpoints', exist_ok=True)
max_acc = 0
global_step = 0
for epoch in range(epochs):
    while len(train_set) > 0:
        data = train_set()
        if len(data) == 0:
            break
        features, labels = process_data(data, label_trans=lambda x: x // 4)
        labels = labels.type(torch.LongTensor)
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        loss = loss_fcn(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        global_step += 1
        print(f"Epoch-{epoch + 1}: {len_train - len(train_set)} / {len_train} loss={loss.item():.3f}".center(50, '-'), end='\r')
        experiment.log({
            'train loss': loss.item(),
            'epoch': epoch + 1
        })

    train_set.reset()

    torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/epoch_{epoch + 1}.pt')
    model.eval()
    acc = model_test(model, test_set)
    if acc > max_acc:
        max_acc = acc
        torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/best.pt')
    model.train()

    experiment.log({
        'epoch': epoch + 1,
        'test_acc': acc,
        'lr': optim.param_groups[0]['lr']
    })
    scheduler.step(acc)

