import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import ModelMain  # Ensure your model.py is compatible with the dataset
from utils.atm import VRDataset  # This assumes the dataset class is in my_dataset.py


config = {
    'train':
        {
            'epochs': 100,
            'batch_size': 32,
            'batch_size_test': 32,
            'lr': 1.0e-3
        },
    'diffusion':
        {
            'layers': 12,
            'channels': 64,
            'nheads': 8,
            'diffusion_embedding_dim': 128,
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'num_steps': 50,
            'schedule': "cosine"
        },
    'model':
        {
            'is_unconditional': 0,
            'timeemb': 128,
            'featureemb': 16
        },
    'input_dim': 22  # Ensure this is set correctly in your config
}


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)  # Example loss function, adjust as needed
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # Adjust logging frequency as needed
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    parser = argparse.ArgumentParser(description='Training script for VR dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the dataset CSV files')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VRDataset(root_path=args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ModelMain(config, device).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

if __name__ == '__main__':
    main()
