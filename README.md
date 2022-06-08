# Custom CSV Logger

## Usage

```python
from logger import Logger
import numpy as np

logger = Logger('./logs')

logger.log_hparams({'lr': 1e-4, 'optimizer': 'Adam'})

for epoch in range(4):
    logger.init_epoch()
    
    # training simulation
    for step in range(10):
        logger.log_metrics({'loss': np.random.rand(), 'acc': np.random.rand()},
                           phase='train', aggregate=True)
        
    print("Average:", logger.epoch['loss'].avg)
    print("Sum:", logger.epoch['loss'].sum)
    
    # validation simulation
    for step in range(2):
        logger.log_metrics({'val_loss': np.random.rand()},
                           phase='val', aggregate=True)
        
        print('Running average:', logger.epoch['val_loss'].avg)

logger.save()
```

The output of the `logs/metrics.csv` file will look like this
```python
    epoch  phase      loss       acc  step  val_loss
0       0  train  0.957911  0.145924     0       NaN
1       0  train  0.953460  0.490201     1       NaN
2       0  train  0.691565  0.109499     2       NaN
3       0  train  0.423360  0.822187     3       NaN
4       0    val       NaN       NaN     4  0.828831
5       0    val       NaN       NaN     5  0.736984
6       1  train  0.849483  0.228059     6       NaN
7       1  train  0.971056  0.489887     7       NaN
8       1  train  0.761992  0.052321     8       NaN
9       1  train  0.968839  0.998951     9       NaN
10      1    val       NaN       NaN    10  0.089479
11      1    val       NaN       NaN    11  0.840629
```

## Todo
- integrate tensorboard
- copyright issues
- `setup.py` and packaging
- set phase with logger.set_phase() ?
