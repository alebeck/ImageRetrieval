import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DataSplitter(DataLoader):

    def __init__(self, dataset, batch_size, val_size, num_workers=4, shuffle=True):
        super().__init__(dataset, batch_size)

        # setup samplers
        idx = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(idx)

        split = int(np.floor(val_size * len(dataset)))
        train_idx, val_idx = idx[split:], idx[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # initialize data loaders
        self.train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
