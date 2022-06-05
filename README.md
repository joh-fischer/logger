# Custom CSV Logger

## Usage

```python
from logger import Logger

logger = Logger('./logs')

logger.log_metrics({'acc': 0.9, 'loss': 0.2}, step=1)
logger.log_hparams({'lr': 1e-4})

logger.save()
```

## Todo
- include .epoch property
  - dictionary with aggregator
  - AverageMeter
  - log metrics with aggregate=True
- phase keyword ('train', 'val', ...)
- integrate tensorboard
- load history from csv
- https://github.com/fyu/pytorch_examples/blob/master/imagenet/main.py AverageMeter class
- copyright issues

## Ideas

Make a `AverageMeter()` per logged value per epoch and make it accessible via dot indexing. E.g.
`logger.epoch_summary['acc'].avg`.