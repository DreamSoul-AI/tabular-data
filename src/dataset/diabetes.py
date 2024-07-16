import anytree
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class Diabetes(Dataset):
    data_name = 'Diabetes'

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'))

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

    '''
        def process(self):
            makedir_exist_ok(self.raw_folder)

            csv_path = os.path.join(self.raw_folder, 'diabetes.csv')
            data_df = pd.read_csv(csv_path)

            X = data_df.drop('Outcome', axis=1).values.astype(np.float32)
            y = data_df['Outcome'].values.astype(np.float32).reshape(-1, 1)

            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]
            split_idx = int(len(X) * 0.8)
            train_data, test_data = X[:split_idx], X[split_idx:]
            train_target, test_target = y[:split_idx], y[split_idx:]
            train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)

            save((train_id, train_data, train_target), os.path.join(self.processed_folder, 'train.pt'))
            save((test_id, test_data, test_target), os.path.join(self.processed_folder, 'test.pt'))
            save(1, os.path.join(self.processed_folder, 'meta.pt'))
            return
    '''

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(), self.root,
                                                                     self.split)
        return fmt_str

    def make_data(self):
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm].reshape(-1, 1)
        split_idx = int(X.shape[0] * 0.8)
        train_data, test_data = X[:split_idx].astype(np.float32), X[split_idx:].astype(np.float32)
        train_target, test_target = y[:split_idx].astype(np.float32), y[split_idx:].astype(np.float32)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        target_size = 1
        return (train_id, train_data, train_target), (test_id, test_data, test_target), target_size