from logger import Logger
import pandas as pd
import time

logger = Logger('./logs')
logger.log_hparams({'lr': 1e-4, 'optim': 'Adam'})

epochs = 2

t0 = time.time()

for epoch in range(epochs):
    logger.init_epoch()
    for step in range(3):
        logger.log_metrics({'loss': epoch*step*2, 'acc': epoch*step*2}, step)

    for step in range(1):
        logger.log_metrics({'val_acc': 2 * epoch}, step)

print("Time:", time.time() - t0)

print("Length of metrics:", len(logger.metrics))
print(logger.metrics)
print(logger.hparams)
logger.save()
print(pd.read_csv('./logs/metrics.csv').head(20))
