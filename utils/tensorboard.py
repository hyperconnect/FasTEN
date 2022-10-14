import logging
from typing import Any
from io import BytesIO
from pathlib import Path

import tensorflow as tf
import torch
import numpy as np
from PIL import Image
from tensorflow.compat.v1.summary import Summary


class TensorboardWriter(object):
    def __init__(self, logdir: Path=None):
        self.logger = logging.getLogger(__name__)
        self.writer = tf.compat.v1.summary.FileWriter(logdir)

    def add_summary(self, value: float, tag: str, step: int):
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
