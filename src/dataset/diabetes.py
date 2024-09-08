import numpy as np
import os
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class Diabetes(Dataset):
    data_name = 'Diabetes'

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, 'data'))
        self.target_size = load(os.path.join(self.processed_folder, 'meta'))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        data_set, meta = self.make_data()
        save(data_set, os.path.join(self.processed_folder, 'data'))
        save(meta, os.path.join(self.processed_folder, 'meta'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}'.format(self.__class__.__name__, self.__len__(), self.root)
        return fmt_str

    def make_data(self):
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm].reshape(-1, 1)
        data = X.astype(np.float32)
        target = y.astype(np.float32)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        target_size = 1
        return data_set, target_size
