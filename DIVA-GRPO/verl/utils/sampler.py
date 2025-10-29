from torch.utils.data import RandomSampler
import torch
import math

class ChunkedRandomSampler(RandomSampler):
    def __init__(self, data_source, samples_per_call=4, total_batch_size=512, generator=None):
        super().__init__(data_source, generator=generator)
        self.samples_per_call = samples_per_call
        self.num_calls_per_batch = math.ceil(total_batch_size / samples_per_call)
        self.total_batches = math.ceil(len(data_source) / self.num_calls_per_batch)
        self.used_indices = set()
        self.all_indices = None

    def __iter__(self):
        print("total_batches", self.total_batches)
        if self.all_indices is None or len(self.used_indices) >= len(self.all_indices):
            self.all_indices = list(super().__iter__())
            self.used_indices.clear()

        remaining_indices = [i for i in self.all_indices if i not in self.used_indices]

        for idx in remaining_indices[:self.num_calls_per_batch]:
            self.used_indices.add(idx)
            yield idx

    def __len__(self):
        return len(data_source)
