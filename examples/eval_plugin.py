#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the Evaluation Plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_scenario
from avalanche.evaluation.metrics import EpochAccuracy, TaskForgetting, \
    EpochLoss, EpochTime, AverageEpochTime, TaskAccuracy, MinibatchAccuracy
from avalanche.evaluation.metrics.confusion_matrix import TaskConfusionMatrix
from avalanche.extras.logging import Logger
from avalanche.extras.models import SimpleMLP
from avalanche.extras.strategy_trace import DotTrace
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive


def main():
    # --- CONFIG
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        RandomCrop(28, padding=4),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    mnist_train = MNIST('./data/mnist', train=True,
                        download=True, transform=train_transform)
    mnist_test = MNIST('./data/mnist', train=False,
                       download=True, transform=test_transform)
    scenario = nc_scenario(
        mnist_train, mnist_test, 5, task_labels=False, seed=1234)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    # The evaluation plugin can be used to compute many different metrics
    # which can be saved and visualized by using a Logger.
    # The evaluation plugin will also use a "tracer" which will print
    # (and log to a file if you want) a bunch of useful information in a
    # nice and easy to read format.
    # Here we define a custom
    my_logger = Logger()
    trace = DotTrace(stdout=True, trace_file='./logs/my_log.txt')
    evaluation_plugin = EvaluationPlugin(
        MinibatchAccuracy(), EpochAccuracy(), TaskAccuracy(), TaskForgetting(),
        EpochLoss(), EpochTime(), AverageEpochTime(),
        TaskConfusionMatrix(num_classes=scenario.n_classes),
        loggers=my_logger, tracers=trace)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
        device=device, plugins=[evaluation_plugin])

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for step in scenario.train_stream:
        print("Start of step: ", step.current_step)
        print("Current Classes: ", step.classes_in_this_step)

        cl_strategy.train(step, num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.test(scenario.test_stream, num_workers=4))


if __name__ == '__main__':
    main()
