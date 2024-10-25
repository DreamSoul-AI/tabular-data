import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from module import check_exists, makedir_exist_ok, save, load


class TabLLM(Dataset):
    data_name = 'TabLLM'

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        # if not check_exists(self.processed_folder):
        self.process()
        self.data, self.meta = load(os.path.join(self.processed_folder, 'data'))

    def __getitem__(self, index):
        # Retrieve the row data for the given index
        row_data = {col: torch.tensor(value) for col, value in self.data[self.split][index].items()}
        if self.transform:
            row_data = self.transform(row_data)
        return row_data

    def __len__(self):
        return len(self.data[self.split])

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def dataset_path(self):
        return os.path.join(self.raw_folder, f"datasets_serialized/{self.data_name}")

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        dataset, meta = self.make_data()
        save((dataset, meta), os.path.join(self.processed_folder, 'data'))

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def make_data(self):
        dataset = load_from_disk(self.dataset_path)
        print(dataset[0])
        exit()
        return dataset, meta

    def __repr__(self):
        return f"Dataset {self.__class__.__name__}\nSize: {self.__len__()}\nRoot: {self.root}\nSplit: {self.split}"


class Bank(TabLLM):
    data_name = 'bank'


class Blood(TabLLM):
    data_name = 'blood'


class CalHousing(TabLLM):
    data_name = 'calhousing'


class Car(TabLLM):
    data_name = 'car'


class CreditG(TabLLM):
    data_name = 'creditg'


class Diabetes(TabLLM):
    data_name = 'diabetes'


class Heart(TabLLM):
    data_name = 'heart'


class Income(TabLLM):
    data_name = 'income'


class Jungle(TabLLM):
    data_name = 'jungle'
