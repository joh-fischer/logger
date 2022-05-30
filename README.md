# Custom Multipurpose CSV Logger

## Usage

```python
from logger import Logger

logger = Logger('./logs')

logger.log_metrics({'acc': 0.9, 'loss': 0.2}, step=1)
logger.log_hparams({'lr': 1e-4})

logger.save()
```

## Todo

- log metrics with dict and arrays (without step), just appending
- maybe logger.newEpoch() function to start new epoch
- always use epoch and step, s.t. it is stored per epoch, per step
- save function with csv writer
- load history from csv
