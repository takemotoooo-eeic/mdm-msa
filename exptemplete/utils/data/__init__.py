from torchvision import datasets, transforms

__all__ = ["MNISTDataset"]


class MNISTDataset(datasets.MNIST):
    """MNIST dataset with a default classification transform."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        super().__init__(root=root, train=train, download=download, transform=transform)
