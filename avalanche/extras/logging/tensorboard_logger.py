#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-11-2020                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module handles all the functionalities related to the logging of
Avalanche experiments. """

from pathlib import Path

from PIL.Image import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from avalanche.evaluation.metric_results import AlternativeValues, MetricValue
from avalanche.extras.logging import StrategyLogger
from avalanche.extras.logging.strategy_logger import TraceList


class TensorboardLogger(StrategyLogger):
    """
    TensorboardLogger is a simple class to handle the interface with the
    TensorBoard API offered by Pytorch.
    """
    def __init__(self, tb_log_dir=".", tb_log_exp_name="tb_data",
                 text_trace: TraceList = None):
        super().__init__(text_trace=text_trace)
        tb_log_dir = Path(tb_log_dir)
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(tb_log_dir / tb_log_exp_name)

    def log_metric(self, metric_value: MetricValue):
        super().log_metric(metric_value)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, float, int)

        if not isinstance(value, (Image, float, int)):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.writer.add_image(name, to_tensor(value),
                                  global_step=metric_value.x_plot)
        elif isinstance(value, (float, int)):
            self.writer.add_scalar(name, value, global_step=metric_value.x_plot)


__all__ = [
    'TensorboardLogger'
]
