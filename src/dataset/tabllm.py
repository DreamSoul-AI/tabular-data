import os
import torch
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import make_classes_counts


class TabLLM(Dataset):
    data_name = 'TabLLM'
    raw_data_name = 'tabllm'

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        # if not check_exists(self.processed_folder):
        self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, 'data'))
        self.classes_counts = make_classes_counts(self.target)
        self.meta = load(os.path.join(self.processed_folder, 'meta'))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed', self.data_name)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw', 'datasets', self.raw_data_name)

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
        return f"Dataset {self.__class__.__name__}\nSize: {self.__len__()}\nRoot: {self.root}\nSplit: {self.split}"

    def make_data(self):
        raise NotImplementedError


class Bank(TabLLM):
    # https://www.kaggle.com/datasets/sonujha090/bank-marketing
    data_name = 'Bank'
    raw_data_name = 'bank'
    feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                     'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    def make_data(self):
        columns = {'V' + str(i + 1): v for i, v in enumerate(self.feature_names)}
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'phpkIxskf.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset.rename(columns={'Class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        dataset['label'] = dataset['label'].astype(int)
        for col in dataset.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
            dataset[col] = dataset[col].astype(float)
        dataset = dataset.to_numpy()
        X, y = dataset[:, :-1], dataset[:, -1]
        data = X.astype(np.float32)
        target = y.astype(np.int64)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        classes = ['False', 'True']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        input_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'input_size': input_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Blood(TabLLM):
    # https://www.kaggle.com/datasets/mmmarchetti/transfusion-dataset
    data_name = 'Blood'
    raw_data_name = 'blood'
    feature_names = ['recency', 'frequency', 'monetary', 'time']

    def make_data(self):
        columns = {'V' + str(i + 1): v for i, v in enumerate(self.feature_names)}
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'php0iVrYT.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset.rename(columns={'Class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        dataset['label'] = dataset['label'].astype(int)
        dataset = dataset.to_numpy()
        X, y = dataset[:, :-1], dataset[:, -1]
        data = X.astype(np.float32)
        target = y.astype(np.int64)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        classes = ['False', 'True']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        input_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'input_size': input_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class CalHousingC(TabLLM):
    # https://www.kaggle.com/datasets/camnugent/california-housing-prices
    data_name = 'CalHousingC'
    raw_data_name = 'calhousing'
    feature_names = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
                     'latitude', 'longitude']

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'houses.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        label = dataset.pop('label')
        dataset['label'] = label
        median_price = dataset['label'].median()
        dataset['label'] = dataset['label'] > median_price
        dataset['label'] = dataset['label'].astype(int)
        dataset = dataset.to_numpy()
        X, y = dataset[:, :-1], dataset[:, -1]
        data = X.astype(np.float32)
        target = y.astype(np.int64)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        classes = ['False', 'True']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        input_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'input_size': input_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class CalHousingR(TabLLM):
    # https://www.kaggle.com/datasets/camnugent/california-housing-prices
    data_name = 'CalHousingR'
    raw_data_name = 'calhousing'
    feature_names = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
                     'latitude', 'longitude']

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'houses.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        label = dataset.pop('label')
        dataset['label'] = label
        dataset = dataset.to_numpy()
        X, y = dataset[:, :-1], dataset[:, -1]
        data = X.astype(np.float32)
        target = y.astype(np.float32)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        input_size = list(X.shape[1:])
        target_size = 1
        meta = {'input_size': input_size, 'target_size': target_size}
        return data_set, meta


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


def byte_to_string_columns(data):
    for col, dtype in data.dtypes.items():
        if dtype == object:  # Only process byte object columns.
            data[col] = data[col].apply(lambda x: x.decode("utf-8"))
    return data
