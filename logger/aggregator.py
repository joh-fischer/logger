class Aggregator:
    def __init__(self):
        """
        Aggregate sum and average.
        """
        self.sum = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
