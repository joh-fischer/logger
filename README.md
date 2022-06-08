# Custom CSV Logger

## Usage

```python
from logger.logger import Logger

logger = Logger('./logs')

logger.log_metrics({'acc': 0.9, 'loss': 0.2}, step=1)
logger.log_hparams({'lr': 1e-4})

logger.save()
```

## Todo
- integrate tensorboard
- https://github.com/fyu/pytorch_examples/blob/master/imagenet/main.py AverageMeter class
- copyright issues
- `setup.py` and packaging
- set phase with logger.set_phase() ?

## Ideas

Make a `AverageMeter()` per logged value per epoch and make it accessible via dot indexing. E.g.
`logger.epoch_summary['acc'].avg`.
