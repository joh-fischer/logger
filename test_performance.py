from logger import Logger
import torch
import time

logger = Logger()

print("With aggregation")
times = []
for step in range(20000):
    t0 = time.time()
    logger.log_metrics({'loss': torch.rand(1), 'acc': torch.rand(1)}, phase='train', aggregate=True)
    times.append( time.time() - t0)

print("Total time:", torch.sum(torch.tensor(times)).item())
print("Min time:", torch.min(torch.tensor(times)).item())
print("Average time:", torch.mean(torch.tensor(times)).item())
print("Max time:", torch.max(torch.tensor(times)).item())

print("Without aggregation")
times = []
for step in range(20000):
    t0 = time.time()
    logger.log_metrics({'loss': torch.rand(1), 'acc': torch.rand(1)}, phase='train', aggregate=False)
    times.append( time.time() - t0)

print("Total time:", torch.sum(torch.tensor(times)).item())
print("Min time:", torch.min(torch.tensor(times)).item())
print("Average time:", torch.mean(torch.tensor(times)).item())
print("Max time:", torch.max(torch.tensor(times)).item())