import torchvision
from torchvision import transforms 
from torch.utils import data


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=2),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=2))

def load_data_cifar10(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    train_trans = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip()]
    test_trans = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if resize:
        # 测试集中心裁剪
        test_trans.insert(0, transforms.Resize(resize))
        # 训练集随机裁剪
        train_trans.insert(0, transforms.RandomResizedCrop(resize))
    train_trans = transforms.Compose(train_trans)
    test_trans = transforms.Compose(test_trans)
    cifar_train = torchvision.datasets.CIFAR10(
        root="../../../data", train=True, transform=train_trans, download=True)
    cifar_test = torchvision.datasets.CIFAR10(
        root="../../../data", train=False, transform=test_trans, download=True)
    return (data.DataLoader(cifar_train, batch_size, shuffle=True,
                            num_workers=2),
            data.DataLoader(cifar_test, batch_size, shuffle=False,
                            num_workers=2))