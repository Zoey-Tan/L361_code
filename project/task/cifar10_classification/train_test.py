"""MNIST training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.types.common import IsolatedRNG


from sklearn.metrics import confusion_matrix, f1_score


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    _config["device"] = torch.device(type="mps")
    # print(_config)

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    # print("DEBUGGGGGG________", config.epochs, config.learning_rate, config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    local_distribution = [0 for _ in range(10)]
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            for t in target:
                local_distribution[t] += 1
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    testloader = DataLoader(
        torch.load("./data/cifar10/partition_0.1/test.pt"),
        batch_size=20,
        shuffle=False,
        generator=_rng_tuple[3],
    )

    net.eval()

    activations_all = None
    # first = True
    with torch.no_grad():
        for images, labels in testloader:
            # if first:
            #    print("Train", labels)
            #    first = False
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )

            outputs = net(images)
            softmax = nn.Softmax(dim=-1)
            if activations_all is None:
                activations_all = deepcopy(softmax(outputs))
            else:
                activations_all = torch.cat(
                    (activations_all, deepcopy(softmax(outputs))), dim=0
                )

    net.train()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
        "test_activation": torch.flatten(activations_all).tolist(),
        "test_activation_mean": torch.mean(activations_all, axis=0).tolist(),
        "test_activation_var": torch.var(activations_all, axis=0).tolist(),
        "local_distribution": local_distribution,
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior


    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    _config["device"] = torch.device(type="mps")
    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    predictions_all = []
    labels_all = []
    activations_all = None
    # first = True
    with torch.no_grad():
        for images, labels in testloader:
            # if first:
            #    print("Test", labels)
            #    first = False
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )

            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            softmax = nn.Softmax(dim=-1)
            if activations_all is None:
                activations_all = deepcopy(softmax(outputs))
            else:
                activations_all = torch.cat(
                    (activations_all, deepcopy(softmax(outputs))), dim=0
                )

            predictions_all = predictions_all + predicted.tolist()
            labels_all = labels_all + labels.tolist()

        torch.manual_seed(_rng_tuple[0])
        random_image = torch.rand(1, 3, 32, 32).to(config.device)
        softmax = nn.Softmax()
        random_activation = softmax(net(random_image))

        # _ = input()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
            "confusion_matrix": confusion_matrix(labels_all, predictions_all).tolist(),
            "f1_score": f1_score(labels_all, predictions_all, average=None).tolist(),
            "random_activation": random_activation.tolist(),
            "test_activation": torch.flatten(activations_all).tolist(),
            "test_activation_mean": torch.mean(activations_all, axis=0).tolist(),
            "test_activation_var": torch.var(activations_all, axis=0).tolist(),
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
