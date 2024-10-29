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
        return f'Dataset {self.__class__.__name__}\nSize: {self.__len__()}\nRoot: {self.root}\nSplit: {self.split}'

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
        classes = ['low', 'high']
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
    # https://archive.ics.uci.edu/dataset/19/car+evaluation
    data_name = 'Car'
    raw_data_name = 'car'
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']

    def make_data(self):
        dataset = pd.read_csv(os.path.join(self.raw_folder, 'car.data'), names=self.feature_names)
        label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        with pd.option_context('future.no_silent_downcasting', True):
            dataset['label'] = dataset['label'].replace(label_dict)
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
        classes = ['unacc', 'acc', 'good', 'vgood']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        input_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'input_size': input_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class CreditG(TabLLM):
    # https://www.kaggle.com/datasets/ppb00x/credit-risk-customers
    data_name = 'CreditG'
    raw_data_name = 'creditg'
    feature_names = [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment', 'installment_commitment',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits',
        'job', 'num_dependents', 'own_telephone', 'foreign_worker'
    ]

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'dataset_31_credit-g.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'good'
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
        classes = ['bad', 'good']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        input_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'input_size': input_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Diabetes(TabLLM):
    # https://www.kaggle.com/datasets/mathchi/diabetes-data-set
    data_name = 'Diabetes'
    raw_data_name = 'diabetes'
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']

    def make_data(self):
        dataset = pd.read_csv(os.path.join(self.raw_folder, 'diabetes.csv'))
        dataset = dataset.rename(columns={'Outcome': 'label'})
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


class Heart(TabLLM):
    # https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    data_name = 'Heart'
    raw_data_name = 'heart'
    feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                     'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    def make_data(self):
        dataset = pd.read_csv(os.path.join(self.raw_folder, 'heart.csv'))
        dataset = dataset.rename(columns={'HeartDisease': 'label'})
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


class Income(TabLLM):
    # https://www.kaggle.com/datasets/uciml/adult-census-income
    data_name = 'Income'
    raw_data_name = 'income'
    feature_names = ['age', 'workclass', 'education', 'marital_status', 'occupation',
                     'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                     'native_country', 'label']

    def make_data(self):
        names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'label']

        def strip_string_columns(df):
            df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

        dataset_train = pd.read_csv(os.path.join(self.raw_folder, 'adult.data'), names=names, na_values=['?', ' ?'])
        strip_string_columns(dataset_train)
        dataset_train['label'] = dataset_train['label'] == '>50K'
        dataset_test = pd.read_csv(os.path.join(self.raw_folder, 'adult.test'), names=names, na_values=['?', ' ?'])
        strip_string_columns(dataset_test)
        dataset_test['label'] = dataset_test['label'] == '>50K.'
        dataset = pd.concat([dataset_train, dataset_test], axis=0)
        dataset = dataset.drop(columns=['fnlwgt', 'education_num'])
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


class Jungle(TabLLM):
    # https://k8sapi.openml.org/d/41027
    data_name = 'Jungle'
    raw_data_name = 'jungle'
    feature_names = ['white_piece0_strength', 'white_piece0_file', 'white_piece0_rank', 'black_piece0_strength',
                     'black_piece0_file', 'black_piece0_rank']

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder,
                                                          'jungle_chess_2pcs_raw_endgame_complete.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'w'  # Does white win?
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


def byte_to_string_columns(data):
    for col, dtype in data.dtypes.items():
        if dtype == object:  # Only process byte object columns.
            data[col] = data[col].apply(lambda x: x.decode('utf-8'))
    return data
