import torch


class EarlyStopping(object):
    def __init__(self, metric='loss', mode='min', min_delta=0, patience=10, checkpoint_file='/tmp/dnn_checkpoint.pt', required_min_iteration=0):
        """

        :param metric: loss, accuracy, f1, sensitivity, specificity, precision
        :param mode:
        :param min_delta:
        :param patience: how many bad epochs is required to stop early
        """
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.bad_epoch_count = 0
        self.required_min_iteration = required_min_iteration
        self.best_index = None
        self.best_metrics = None
        self.best_model = None

        self.checkpoint_file = checkpoint_file

    def step(self, epoch, results, model):
        # From Goodfellow's Deep Learning book
        if self.best_index is None:
            self.best_index = epoch
            self.best_metrics = results
            self.save_checkpoint(epoch, results, model)
        else:
            if self.mode == 'min':
                if self.best_metrics[self.metric] - results[self.metric] > self.min_delta:
                    # Update best metrics and save checkpoint
                    self.save_checkpoint(epoch, results, model)
                else:
                    self.bad_epoch_count += 1
            else:
                if self.best_metrics[self.metric] - results[self.metric] < self.min_delta:
                    # Update best metrics and save checkpoint
                    self.save_checkpoint(epoch, results, model)
                else:
                    self.bad_epoch_count += 1
            # Prevent early stopping before min iteration would be done in training
            if self.bad_epoch_count > self.patience and epoch > self.required_min_iteration:
                return True
            else:
                return False

    def save_checkpoint(self, epoch, results, model):
        self.best_index = epoch
        self.best_metrics = results
        self.bad_epoch_count = 0
        torch.save(model.state_dict(), self.checkpoint_file)


