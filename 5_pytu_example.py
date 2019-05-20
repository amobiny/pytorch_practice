import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pytu.networks import DenseNet
from pytu.data import SimpleDataset
from pytu.advanced.objectives import CrossEntropyLossND
from pytu.strategies import SimpleStrategy
from pytu.iterators import Trainer
from pytu.advanced.metrics import accuracy
from pytu.iterators import Tester

from utils.loader_utils import get_test_loader, get_train_valid_loader


def one_hot_encode(y, num_cls):
    """
    one hot encoding
    :param y: must be of shape [num_samples]
    :param num_cls: number of classes
    :return: one hot encoded labels of shape [num-samples, num-cls]
    """
    return (y.unsqueeze(1) == torch.arange(num_cls).reshape(1, num_cls)).float()


def run():
    root = r'C:\Users\z0042n0w\Desktop\pytorch_practice\data'
    if not os.path.exists(root):
        os.mkdir(root)
    batch_size = 128
    # trainloader, validloader = get_train_valid_loader(root, batch_size=batch_size, random_seed=0)
    # testloader = get_test_loader(root, batch_size=batch_size)

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=None)

    dataset_train = SimpleDataset(train_dataset.data[:55000].unsqueeze(1).float(),
                                  one_hot_encode(train_dataset.targets[:55000], 10))
    dataset_valid = SimpleDataset(train_dataset.data[55000:].unsqueeze(1).float(),
                                  one_hot_encode(train_dataset.targets[55000:], 10))
    dataset_test = SimpleDataset(test_dataset.data.unsqueeze(1).float(),
                                 one_hot_encode(test_dataset.targets, 10))

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = DenseNet(input_nc=1, output_nc=10, n_pool=3, unary_output=True)
    optimizer = torch.optim.Adadelta(model.parameters())
    loss = CrossEntropyLossND()

    strategy = SimpleStrategy(model, optimizer, loss)

    trainer = Trainer(trainloader, validloader, strategy, save_criterions=['accuracy'], tensorboard=True)
    trainer.add_metric(accuracy, comp='higher')

    trainer.train(50)

    tester = Tester(testloader, strategy)
    tester.add_metric(accuracy)
    tester.test()


if __name__ == '__main__':
    run()






