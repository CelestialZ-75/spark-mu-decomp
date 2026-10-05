"""Three time-block operations used by the existing CKC loop."""

import operator
import socket
from itertools import pairwise

import numpy as np


class SparkOps:
    def __init__(self, session, matrix, partitions: int = 2):
        if matrix.device.type != "cpu":
            raise ValueError("Spark uses CPU blocks")
        self.samples = matrix.shape[1]
        if not 1 <= partitions <= self.samples:
            raise ValueError("partitions must be between 1 and the sample count")
        self.context = session.sparkContext
        array = matrix.detach().numpy()
        bounds = np.linspace(0, self.samples, partitions + 1, dtype=int)
        blocks = [(int(a), array[:, a:b].copy()) for a, b in pairwise(bounds)]
        self.blocks = self.context.parallelize(blocks, partitions).cache()
        # Collecting host names also materializes the blocks before iteration timing.
        self.hosts = self.blocks.map(lambda block: socket.gethostname()).collect()

    def correlation(self):
        import torch

        total = self.blocks.map(lambda block: block[1] @ block[1].T).treeReduce(operator.add)
        return torch.from_numpy(total / self.samples)

    def project(self, vector):
        import torch

        broadcast = self.context.broadcast(vector.detach().numpy())
        try:
            parts = self.blocks.map(lambda block: (block[0], broadcast.value @ block[1])).collect()
            parts.sort(key=lambda part: part[0])
            return torch.from_numpy(np.concatenate([part[1] for part in parts]))
        finally:
            broadcast.destroy()

    def weighted_sum(self, weights):
        import torch

        broadcast = self.context.broadcast(weights.detach().numpy())
        try:
            result = self.blocks.map(
                lambda block: block[1] @ broadcast.value[block[0] : block[0] + block[1].shape[1]]
            ).treeReduce(operator.add)
            return torch.from_numpy(result)
        finally:
            broadcast.destroy()

    def close(self):
        self.blocks.unpersist(blocking=True)
