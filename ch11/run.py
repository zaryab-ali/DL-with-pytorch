import datetime
import sys
import argparse
import torch

from util.util import enumerateWithEstimate

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

METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

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
            # log
            train_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "trn", train_metrics_t)

            val_metrics_t = self.do_validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, "val", val_metrics_t)

        if hasattr(self, "trn_writer"):
            self.trn_writer.close()
            self.val_writer.close()


    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx = train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.conpute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()
        self.total_training_sample_count += len(train_dl.dataset)

        return trnMetrics_g.to("cpu")

        

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device = self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.conpute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')


    def conpute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t , label_t, series_list, center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_g = loss_func(logits_g, label_g[:,1])
        start_ndx = batch_ndx *batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()
    





    def log_metrics(self, *args):






if __name__ == '__main__':
    LunaTrainingApp().main()