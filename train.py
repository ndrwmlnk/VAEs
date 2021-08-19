import argparse
import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np

import _pickle as cPickle
import bz2

from model_ae import ConvAE
from model_vae import ConvVAE
from model_sigma_vae import ConvSigmaVAE

""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

## Arguments
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='mse', metavar='N',
                    help='which model to use: mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='test', metavar='N')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--dim', type=int, default=2)
args = parser.parse_args()

## Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #array to tensor and rearrrange for pytorch (3,64,64)
        data = torch.Tensor((np.stack(data).reshape((args.batch_size, args.image_size, args.image_size, 3))) / 255).transpose(1, 2).transpose(1, 3)
        data = data.to(device)
        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(data)
        # Compute loss
        rec, kl = model.loss_function(recon_batch, data, mu, logvar)

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

        if (args.model == "ae"):
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           rec.item() / len(data)))

        else:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tlog_sigma: {:f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           rec.item() / len(data),
                           kl.item() / len(data),
                    model.log_sigma))
            
    train_loss /=  len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    if (args.model != "ae"):
        summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)
        summary_writer.add_scalar('train/log_sigma', model.log_sigma, epoch)

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    print("Train")
    for model_idx, model in enumerate(["optimal_sigma_vae", "mse_vae", "ae"]):
        args.model = model
        for size_idx, size in enumerate([64, 16, 8, 16]):
            args.image_size = size
            for dim_idx, dim in enumerate([256, 128, 64, 32, 16, 8, 4, 2, 1]):
                print(model, size, dim)
                args.dim = dim
                #load cropped dataset
                if(size_idx == 3):
                    args.log_dir = "cropped_{}_{}x{}_D{}".format(args.model, args.image_size, args.image_size, args.dim)
                    train_data = bz2.BZ2File('dataset/train_crop_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
                    test_data = bz2.BZ2File('dataset/test_crop_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
                #load scaled dataset
                else:
                    args.log_dir = "{}_{}x{}_D{}".format(args.model, args.image_size, args.image_size, args.dim)
                    train_data = bz2.BZ2File('dataset/train_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
                    test_data = bz2.BZ2File('dataset/test_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
                train_x = cPickle.load(train_data)
                test_x = cPickle.load(test_data)
                train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x))
                test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_x))
                kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           drop_last=True, **kwargs)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                          drop_last=True, **kwargs)
                ## Logging
                os.makedirs('vae_logs/{}'.format(args.log_dir), exist_ok=True)
                summary_writer = SummaryWriter(log_dir='vae_logs/' + args.log_dir, purge_step=0)

                ## Build Model
                if (args.model == "ae"):
                    model = ConvAE(device, 3, args).to(device)
                if (args.model == "mse_vae"):
                    model = ConvVAE(device, 3, args).to(device)
                if (args.model == "optimal_sigma_vae"):
                    model = ConvSigmaVAE(device, 3, args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                #training
                for epoch in range(1, args.epochs + 1):
                    train(epoch)
                    with torch.no_grad():
                        sample = model.sample(64).cpu()
                        save_image(sample.view(64, -1, args.image_size, args.image_size),
                                   'vae_logs/{}/sample_{}.png'.format(args.log_dir, str(epoch)))
                    summary_writer.file_writer.flush()

                torch.save(model.state_dict(), 'checkpoints/checkpoint_{}.pt'.format(args.log_dir))