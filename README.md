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
- save function with csv writer
