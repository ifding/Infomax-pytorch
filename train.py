import torch
from models import Encoder, DeepInfoMaxLoss
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    cifar_10_train_dt = CIFAR10(r'~/.torch',  download=True, transform=ToTensor())
    cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    encoder = Encoder().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam(encoder.parameters(), lr=1e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=1e-4)

    epoch_restart = 20
    root = Path(f"./Models/run1")

    if epoch_restart is not None and root is not None:
        enc_file = root / Path('encoder' + str(epoch_restart) + '.wgt')
        loss_file = root / Path('loss' + str(epoch_restart) + '.wgt')
        encoder.load_state_dict(torch.load(str(enc_file)))
        loss_fn.load_state_dict(torch.load(str(loss_file)))

    for epoch in range(epoch_restart + 1, 30):
        batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
        train_loss = []
        for x, target in batch:
            x = x.to(device)

            optim.zero_grad()
            loss_optim.zero_grad()
            y, M = encoder(x)
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0) # put the first one to the last
            loss = loss_fn(y, M, M_prime)
            train_loss.append(loss.item())
            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:])))
            loss.backward()
            optim.step()
            loss_optim.step()
            
        if epoch % 10 == 0:
            root = Path(f"./Models/cifar10")
            enc_file = root / Path('encoder' + str(epoch) + '.wgt')
            loss_file = root / Path('loss' + str(epoch) + '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))
