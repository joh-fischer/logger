import yaml
from datetime import datetime
import csv
import os


class Logger:
    def __init__(self, log_dir: str, name: str = None, include_time: bool = False):
        self.name = name if name else 'log'
        self.time = datetime.now().strftime('%y-%m-%d_%H%M%S') if include_time else ''

        self.metrics = []
        self.hparams = {}

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def log_hparams(self, params: dict):
        """
        Log hyper-parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the hyper-parameters as key-value pairs. For example {'optimizer': 'Adam', 'lr': 1e-02}.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: dict, step: int = None):
        """
        Log metrics.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary containing the metrics as key-value pairs. For example {'acc': 0.9, 'loss': 0.2}.
        step : int, optional
            Step number where metrics are to be recorded.
        """
        step = step if step else len(self.metrics)

        metrics = {k: self._handle_value(v) for k, v in metrics_dict.items()}
        metrics['step'] = step
        self.metrics.append(metrics)

    def save(self):
        """
        Save the hyper-parameters and metrics to a file.
        """
        if self.metrics:
            metrics_file_path = os.path.join(self.log_dir, self.name + self.time + '.csv')
            last_m = {}
            for m in self.metrics:
                last_m.update(m)
            metrics_keys = list(last_m.keys())

            with open(metrics_file_path, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=metrics_keys)
                writer.writeheader()
                writer.writerows(self.metrics)

        if self.hparams:
            hparams_file_path = os.path.join(self.log_dir, self.name + self.time + '.yaml')

            output = yaml.dump(self.hparams, Dumper=yaml.CDumper)
            with open(hparams_file_path, 'w') as file:
                file.write(output)


    @staticmethod
    def _handle_value(value):
        #if isinstance(value, torch.Tensor):
        #    return value.item()
        return value
