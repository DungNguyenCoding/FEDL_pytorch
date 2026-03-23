import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloader(dataset='mnist', K=30):
    """
    Implements Pathological Non-IID partitioning (Section VI-A).
    """
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # Sort by labels to create non-IID shards [cite: 478]
        indices = np.argsort(train_data.targets.numpy())
        shards = [indices[i:i + 1000] for i in range(0, 60000, 1000)]
        loaders = []
        for k in range(K):
            # Assign 2 shards per device (2 digits only) [cite: 479]
            client_idx = np.concatenate([shards[k*2], shards[k*2 + 1]])
            loaders.append(DataLoader(Subset(train_data, client_idx), batch_size=32, shuffle=True))
        return loaders

    elif dataset == 'cifar10_svm':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        targets = np.array(train_data.targets)
        # Filter for Airplane (0) and Automobile (1) [cite: 476]
        idx_0 = np.where(targets == 0)[0]
        idx_1 = np.where(targets == 1)[0]
        loaders = []
        for k in range(K):
            # 330 samples of exactly ONE class per device [cite: 476]
            if k < 15:
                c_idx = idx_0[k*330 : (k+1)*330]
            else:
                c_idx = idx_1[(k-15)*330 : (k-14)*330]
            loaders.append(DataLoader(Subset(train_data, c_idx), batch_size=32, shuffle=True))
        return loaders

def get_test_loader(dataset='mnist'):
    """
    Standard IID test loader for global model evaluation[cite: 14, 600].
    """
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        # For CIFAR-10 SVM binary classification
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        targets = np.array(test_data.targets)
        mask = (targets == 0) | (targets == 1)
        test_data = Subset(test_data, np.where(mask)[0])

    return DataLoader(test_data, batch_size=128, shuffle=False)