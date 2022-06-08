from logger import Logger
import numpy as np
import pandas as pd


logger = Logger('./logs')

logger.log_hparams({'lr': 1e-4, 'optimizer': 'Adam'})

for epoch in range(2):
    logger.init_epoch()

    # training simulation
    for step in range(4):
        logger.log_metrics({'loss': np.random.rand(), 'acc': np.random.rand()}, phase='train', aggregate=True)

    print("Average:", logger.epoch['loss'].avg)
    print("Sum:", logger.epoch['loss'].sum)

    # validation simulation
    for step in range(2):
        logger.log_metrics({'val_loss': np.random.rand()}, phase='val', aggregate=True)

        print('Running average:', logger.epoch['val_loss'].avg)

logger.save()

print(pd.read_csv('logs/metrics.csv').head(20))