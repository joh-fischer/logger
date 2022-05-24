from logger import Logger
import pandas as pd

logger = Logger('./logs')
logger.log_hparams({'lr': 1e-4, 'optim': 'Adam'})
for i in range(10):
    logger.log_metrics({'acc': i}, i)
    if i % 2 == 0:
        logger.log_metrics({'val_acc': i*10}, i)

print(logger.metrics)
print(logger.hparams)
print(pd.read_csv('./logs/log.csv').head(5)
logger.save()
