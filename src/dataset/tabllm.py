import os
import torch
import pandas as pd
import numpy as np
from scipy.io import arff
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from config import cfg


class TabLLM(Dataset):
    data_name = 'TabLLM'
    raw_data_name = 'tabllm'
    feature_names = []
    target_names = []

    def __init__(self, root, process=False, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        if not check_exists(self.processed_folder) or process:
            self.process(process)
        self.meta = {}
        self.id, self.numeric_data, self.target = load(os.path.join(self.processed_folder, 'numeric', 'data'))
        _, self.semantic_data, _ = load(os.path.join(self.processed_folder, 'semantic', 'data'))
        self.meta['numeric'] = load(os.path.join(self.processed_folder, 'numeric', 'meta'))
        self.meta['semantic'] = load(os.path.join(self.processed_folder, 'semantic', 'meta'))
        if os.path.exists(os.path.join(self.processed_folder, '{}.txt'.format(self.data_name))):
            with open(os.path.join(self.processed_folder, '{}.txt'.format(self.data_name)), 'r') as file:
                self.description = file.read()
        else:
            self.description = None
        self.encoder_model_name = 'intfloat/multilingual-e5-large-instruct'
        cache_dir = os.path.join('output', 'cache')
        cache_tokenizer_path = os.path.join(cache_dir, self.encoder_model_name, 'tokenizer')
        if not os.path.exists(os.path.join(cache_dir, self.encoder_model_name)):
            local_files_only = False
        else:
            local_files_only = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name, trust_remote_code=True,
                                                       cache_dir=cache_tokenizer_path,
                                                       local_files_only=local_files_only)

    def __getitem__(self, index):
        semantic_data = self.transform_semantic_data(self.semantic_data[index])
        input = {'id': torch.tensor(self.id[index]),
                 'numeric_data': torch.tensor(self.numeric_data[index]),
                 'semantic_data': semantic_data,
                 'target': self.target[index]}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.id)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed', self.data_name)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw', 'datasets', self.raw_data_name)

    def process(self, process):
        if not check_exists(self.raw_folder):
            self.download()
        if not check_exists(os.path.join(self.processed_folder, 'numeric')) or process:
            data_set, meta = self.make_numeric_data()
            save(data_set, os.path.join(self.processed_folder, 'numeric', 'data'))
            save(meta, os.path.join(self.processed_folder, 'numeric', 'meta'))
        if not check_exists(os.path.join(self.processed_folder, 'semantic')) or process:
            data_set, meta = self.make_semantic_data()
            save(data_set, os.path.join(self.processed_folder, 'semantic', 'data'))
            save(meta, os.path.join(self.processed_folder, 'semantic', 'meta'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        return f'Dataset {self.__class__.__name__}\nSize: {self.__len__()}\nRoot: {self.root}'

    def make_numeric_data(self):
        raise NotImplementedError

    def make_semantic_data(self):
        raise NotImplementedError

    def transform_semantic_data(self, data):
        max_length = cfg['model']['max_length']
        seq_length = 2 * (len(self.feature_names) + len(self.target_names))

        data = [element for tup in data for element in tup]
        data = self.tokenizer(data, return_tensors="pt", padding='max_length',
                              max_length=max_length, truncation=True)
        data['input_ids'] = data['input_ids'].view(-1, seq_length, max_length)
        data['attention_mask'] = data['attention_mask'].view(-1, seq_length, max_length)
        return data


class Bank(TabLLM):
    # https://archive.ics.uci.edu/dataset/222/bank+marketing
    # https://www.kaggle.com/datasets/sonujha090/bank-marketing

    data_name = 'Bank'
    raw_data_name = 'bank'
    feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                     'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    target_names = ['deposit']

    def make_data(self):
        columns = {'V' + str(i + 1): v for i, v in enumerate(self.feature_names)}
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'phpkIxskf.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset.rename(columns={'Class': 'deposit'}, inplace=True)
        dataset['deposit'] = dataset['deposit'] == '2'
        return dataset

    def make_numeric_data(self):
        dataset = self.make_data()

        dataset['deposit'] = dataset['deposit'].astype(int)
        for col in dataset.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
            dataset[col] = dataset[col].astype(float)
        dataset = dataset.to_numpy()
        data, target = dataset[:, :-1], dataset[:, -1]
        data = data.astype(np.float32)
        target = target.astype(np.int64)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)

        classes = ['False', 'True']
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        data_size = list(data.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta

    def make_semantic_data(self):
        dataset = self.make_data()
        dataset = dataset.astype(str)
        data = dataset.apply(convert_to_semantic, axis=1)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, None)
        meta = {}
        return data_set, meta


class Blood(TabLLM):
    # https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center
    # https://www.kaggle.com/datasets/ninalabiba/blood-transfusion-dataset

    data_name = 'Blood'
    raw_data_name = 'blood'
    feature_names = ['recency', 'frequency', 'monetary', 'time']
    target_names = ['donate']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class CalHousingC(TabLLM):
    # https://www.kaggle.com/datasets/camnugent/california-housing-prices
    data_name = 'CalHousingC'
    raw_data_name = 'calhousing'
    feature_names = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
                     'latitude', 'longitude']
    target_names = ['median_house_value']

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'houses.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        median_house_value = dataset.pop('median_house_value')
        dataset['median_house_value'] = median_house_value
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class CalHousingR(TabLLM):
    # https://www.kaggle.com/datasets/camnugent/california-housing-prices
    data_name = 'CalHousingR'
    raw_data_name = 'calhousing'
    feature_names = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
                     'latitude', 'longitude']
    target_names = ['median_house_value']

    def make_data(self):
        dataset = pd.DataFrame(arff.loadarff(os.path.join(self.raw_folder, 'houses.arff'))[0])
        dataset = byte_to_string_columns(dataset)
        median_house_value = dataset.pop('median_house_value')
        dataset['median_house_value'] = median_house_value
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        dataset = dataset.to_numpy()
        X, y = dataset[:, :-1], dataset[:, [-1]]
        data = X.astype(np.float32)
        target = y.astype(np.float32)
        id = np.arange(len(data)).astype(np.int64)
        data_set = (id, data, target)
        data_size = list(X.shape[1:])
        target_size = 1
        meta = {'data_size': data_size, 'target_size': target_size}
        return data_set, meta


class Car(TabLLM):
    # https://archive.ics.uci.edu/dataset/19/car+evaluation
    data_name = 'Car'
    raw_data_name = 'car'
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
    target_names = ['label']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
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
    target_names = ['class']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Diabetes(TabLLM):
    # https://www.kaggle.com/datasets/mathchi/diabetes-data-set
    data_name = 'Diabetes'
    raw_data_name = 'diabetes'
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']
    target_names = ['Outcome']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Heart(TabLLM):
    # https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    data_name = 'Heart'
    raw_data_name = 'heart'
    feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                     'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    target_names = ['HeartDisease']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Income(TabLLM):
    # https://archive.ics.uci.edu/dataset/2/adult
    # https://www.kaggle.com/datasets/uciml/adult-census-income
    data_name = 'Income'
    raw_data_name = 'income'
    feature_names = ['age', 'workclass', 'education', 'marital_status', 'occupation',
                     'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                     'label', 'label']
    target_names = ['label']

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
        dataset = dataset.drop(columns=['fnlwgt', 'education_num'])  # drop based on TabLLM
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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


class Jungle(TabLLM):
    # https://k8sapi.openml.org/d/41027
    data_name = 'Jungle'
    raw_data_name = 'jungle'
    feature_names = ['white_piece0_strength', 'white_piece0_file', 'white_piece0_rank', 'black_piece0_strength',
                     'black_piece0_file', 'black_piece0_rank']
    target_names = ['class']

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
        data_size = list(X.shape[1:])
        target_size = len(classes)
        meta = {'data_size': data_size, 'target_size': target_size, 'classes_to_labels': classes_to_labels}
        return data_set, meta


def byte_to_string_columns(data):
    for col, dtype in data.dtypes.items():
        if dtype == object:  # Only process byte object columns.
            data[col] = data[col].apply(lambda x: x.decode('utf-8'))
    return data


def convert_to_semantic(row):
    return [(col, value) for col, value in row.items()]
