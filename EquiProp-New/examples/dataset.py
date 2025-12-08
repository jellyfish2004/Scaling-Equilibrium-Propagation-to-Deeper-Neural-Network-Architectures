import torch
try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None

def _build_cifar10_loaders(batch_size: int = 128, num_workers: int = 4, persistent_workers: bool = True, normalize: bool = True):
    # CHANGE: CIFAR-10 dataloaders mirroring original setup
    assert torchvision is not None and T is not None, "torchvision is required to run CIFAR-10 training"
    # CHANGE: align preprocessing with original (flip, RandAugment, crop, normalize with scaled std)
    mean=(0.4914, 0.4822, 0.4465)
    std=(0.2023*3, 0.1994*3, 0.2010*3)
    final_norm = T.Normalize(mean, std) if normalize else T.Lambda(lambda x: x)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandAugment(num_ops=3, magnitude=9),
        T.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
        T.ToTensor(),
        final_norm,
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        final_norm,
    ])
    train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    test_loader  = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    return train_loader, test_loader


def _build_mnist_loaders(batch_size: int = 128, num_workers: int = 4, persistent_workers: bool = True):
    assert torchvision is not None and T is not None, "torchvision is required for MNIST"
    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    return train_loader, test_loader