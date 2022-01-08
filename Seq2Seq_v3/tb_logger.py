#import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import scipy.misc
import os
import torch
from os import path

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir='./', network='BERT', name=None):
        """Create a summary writer logging to log_dir."""
        self.name = name
        self.network = network
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            print('Tensorboard Log is logged : ', os.path.join(log_dir, name))
            self.writer = SummaryWriter(logdir="{}".format(os.path.join(log_dir, name)))
        else:
            print('Tensorboard Log is logged : ', log_dir)
            self.writer = SummaryWriter(logdir="{}".format(log_dir))

    def scalar_summary(self, tags, values, step):
        """Log a scalar variable.
        """
        self.writer.add_scalar(tag=tags, scalar_value=values, global_step=step)
        self.writer.flush()
