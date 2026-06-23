import datetime
import sys
import argparse
import torch

from torch.optim import SGD, Adam

from torch import nn

from .model import LunaModel
from .dataset import Lunadataset

from torch.utils.data import DataLoader

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def run(app, *argv):
    args = list(argv)
    args.insert(0, "--num-workers=2")



class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )


        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.total_training_sample_count = 0


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model


    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.0001, momentum = 99)


    def train_data_loader(self):
        train_dataset = Lunadataset(val_stride=10, isValSet_bool = False)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size = batch_size*torch.cuda.device_count()

        train_dataloader = DataLoader(
            dataset= train_dataset,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dataloader


    def val_data_loader(self):
        val_dataset = Lunadataset(val_stride=10, isValSet_bool = True)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size = batch_size*torch.cuda.device_count()

        val_dataloader = DataLoader(
            dataset= val_dataset,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dataloader
    
    def main(self):
        #log
        train_dl = self.train_data_loader
        val_dl = self.val_data_loader

        for epoch_ndx in range(1, self.cli_args.epochs +1):
            






if __name__ == '__main__':
    LunaTrainingApp().main()