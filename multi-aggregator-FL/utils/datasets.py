from torchvision import transforms
from torchvision import datasets


def get_dataset(path, name):
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ])
        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(path, train=False, transform=transform)

        return train_dataset, test_dataset
    elif name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(path, train=False, transform=transform_test)
        return train_dataset, test_dataset
