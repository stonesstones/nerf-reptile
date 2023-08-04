import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, log_dir, model_log_dir=None, global_step=0):
        self._log_dir = log_dir
        self._model_log_dir = model_log_dir
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self.global_step = global_step
        self.outer_epoch = 0
        print('########################')
        print('logging outputs to ', log_dir)
        print('global step start at: ', self.global_step)
        print('########################')

    def add_scalars(self, scalar_dict, global_step=None):
        if global_step is None:
            global_step = self.global_step
        for key, value in scalar_dict.items():
            print('{}: {}'.format(key, value))
            self._writer.add_scalar(f"{key}", value, global_step=global_step)

    def add_histograms(self, scalar_dict, global_step=None):
        if global_step is None:
            global_step = self.global_step
        for key, value in scalar_dict.items():
            print('{}: {}'.format(key, value))
            self._writer.add_histogram(f"{key}", value, global_step=global_step, bins='auto')

    def add_images(self, image_dict, global_step=None):
        if global_step is None:
            global_step = self.global_step
        for key, img in image_dict.items():
            print('image: {}, shape: {}'.format(key, img.shape))
            grid = make_grid(img)
            self._writer.add_images(f"{key}", grid, global_step=global_step, dataformats='HWC')

    def add_global_step(self):
        self.global_step += 1

    def add_outer_epoch(self):
        self.outer_epoch += 1

    def flush(self):
        self._writer.flush()

    def should_record(self, step) -> bool:
        return (self.global_step % step) == 0

    def save_weights(self, models, i):
        for key, model in models.items():
            path = os.path.join(self._model_log_dir, '{}_{:06d}.pth'.format(key, i))
            torch.save(model.state_dict(), path)
            print('saved weights at', path)
