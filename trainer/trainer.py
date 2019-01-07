import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.logger.info("data loader size: {}×{}".format(self.data_loader.batch_size, len(self.data_loader)))
        if self.do_validation:
            self.logger.info("valid data loader size: {}×{}".format(self.valid_data_loader.batch_size,
                                                                    len(self.valid_data_loader)))

    def _eval_metrics(self, output, target):
        names, values = [], []
        for i, metric in enumerate(self.metrics):
            result = metric(output, target)
            if isinstance(result, tuple) or isinstance(result, list):
                names.append(result[0])
                values.append(result[1])
            else:
                names.append(metric.__name__)
                values.append(result)
        flat_names = [item for sublist in names for item in sublist]
        flat_values = np.array([item for sublist in values for item in sublist])
        for i, name in enumerate(flat_names):
            self.writer.add_scalar(name, flat_values[i])
        return flat_names, flat_values

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = 0
        metric_names = None
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)
            if isinstance(target, list):
                target = [x.to(self.device) for x in target]
            else:
                target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            metric_names, metrics = self._eval_metrics(output, target)
            total_metrics += metrics

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (metric_names, (total_metrics / len(self.data_loader)).tolist())
        }

        if self.do_validation and epoch % self.validate_freq == 0:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = 0
        metric_names = None
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                if isinstance(target, list):
                    target = [x.to(self.device) for x in target]
                else:
                    target = target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                metric_names, metrics = self._eval_metrics(output, target)
                total_val_metrics += metrics
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (metric_names, (total_val_metrics / len(self.valid_data_loader)).tolist())
        }
