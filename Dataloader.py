from torch.utils import data
import os
from multiprocessing import Lock
import random
from PIL import Image
import numpy as np
import torch


class FolderDataset(data.Dataset, ):

    def __init__(self, folder_loc, files=None):
        if folder_loc[-1] != "/":
            self.loc = folder_loc + "/"
        else:
            self.loc = folder_loc
        if files is None:
            self.files = [str(x) for x in os.listdir(folder_loc) if x.endswith('.jpg')]
        else:
            self.files = files
        self.locks = [Lock() for _ in self.files]

        random.Random().shuffle(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        age = int(file.split("_")[0])
        with self.locks[index]:
            im = Image.open(self.loc + file)
        im = np.asarray(im)
        return (im, age)

    def __len__(self):
        return len(self.files)

        # Train Test Val


class FolderDataloader(data.DataLoader):

    def __init__(self, dataset, collate_fn=None, num_workers=2, batch_size=1):
        super().__init__(dataset, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size, shuffle=True)


def collate(batch):
    images = np.asarray([x[0] for x in batch])
    labels = [x[1] for x in batch]
    images = np.einsum("bxyc->bcxy", images)

    return torch.tensor(images, dtype=torch.float), torch.tensor(labels, dtype=torch.uint8)


def generate_dataloaders(folder_loc, split=(0.6, 0.3, 0.1), collate_fxn=collate, num_workers=2, batch_size=1):
    files = [str(x) for x in os.listdir(folder_loc) if x.endswith('.jpg')]
    random.Random(0).shuffle(files)
    split_a = int(len(files) * split[0])
    split_b = split_a + int(len(files) * split[1])
    split_c = split_b + int(len(files) * split[2])

    train = files[0:split_a]
    test = files[split_a:split_b]
    val = files[split_b:]

    train_ds = FolderDataset(folder_loc, files=train)
    train_dl = FolderDataloader(train_ds, collate_fn=collate_fxn, num_workers=num_workers, batch_size=batch_size)
    test_ds = FolderDataset(folder_loc, files=test)
    test_dl = FolderDataloader(test_ds, collate_fn=collate_fxn, num_workers=num_workers)
    val_ds = FolderDataset(folder_loc, files=val)
    val_dl = FolderDataloader(val_ds, collate_fn=collate_fxn, num_workers=num_workers)

    return train_dl, test_dl, val_dl
