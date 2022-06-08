from logger import Logger
import pandas as pd
import numpy as np
import time

logger = Logger('./logs')
logger.log_hparams({'lr': 1e-4, 'optim': 'Adam'})

epochs = 2

t0 = time.time()

for epoch in range(epochs):
    print("-- Epoch {}".format(epoch+1))
    logger.init_epoch()

    # training simulation
    for step in range(3):
        logger.log_metrics({'loss': np.random.rand(), 'acc': np.random.rand()}, phase='train', aggregate=True, n=1)

    print("Average:", logger.epoch['loss'].avg)
    print("Sum:", logger.epoch['loss'].sum)

    # validation simulation
    for step in range(2):
        logger.log_metrics({'val_acc': np.random.rand()}, phase='val')

print("\nTime:", time.time() - t0)

print("Length of metrics:", len(logger.metrics))
print(logger.metrics)
print(logger.hparams)
logger.save()
print(pd.read_csv('logs/metrics.csv').head(20))
