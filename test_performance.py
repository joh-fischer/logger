from logger import Logger
import numpy as np
import time

logger = Logger()

print("With aggregation")
times = []
for step in range(20000):
    t0 = time.time()
    logger.log_metrics({'loss': np.random.rand(), 'acc': np.random.rand()}, phase='train', aggregate=True)
    times.append( time.time() - t0)

print("Total time:", np.sum(times))
print("Average time:", np.mean(times))
print("Max time:", np.max(times))
print("Min time:", np.min(times))

print("Without aggregation")
times = []
for step in range(20000):
    t0 = time.time()
    logger.log_metrics({'loss': np.random.rand(), 'acc': np.random.rand()}, phase='train', aggregate=False)
    times.append( time.time() - t0)

print("Total time:", np.sum(times))
print("Average time:", np.mean(times))
print("Max time:", np.max(times))
print("Min time:", np.min(times))