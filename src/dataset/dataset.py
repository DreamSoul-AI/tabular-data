import copy
import dataset
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from module import apply_recursively
from config import cfg


def make_dataset(data_name, eval_mode=None, process=False, verbose=True):
    if verbose:
        print('fetching data {}...'.format(data_name))
    if data_name in ['Bank', 'Blood', 'CalHousingC', 'CalHousingR', 'Car', 'CreditG', 'Diabetes', 'Heart', 'Income',
                     'Jungle']:
        root = os.path.join('data', 'TabLLM')
        raw_dataset = eval('dataset.{}(root=root, process=process)'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    eval_mode = '0.9-holdout' if eval_mode is None else eval_mode
    if 'holdout' in eval_mode:
        test_size = 1 - float(eval_mode.split('-')[0])
        train_idx, test_idx = train_test_split(range(len(raw_dataset)), test_size=test_size, random_state=cfg['seed'])
        train_idx, test_idx = [train_idx], [test_idx]
    elif 'fold' in eval_mode:
        k = int(eval_mode.split('-')[0])
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg['seed'])
        train_idx, test_idx = [], []
        for train_idx_i, test_idx_i in kf.split(range(len(raw_dataset))):
            train_idx.append(train_idx_i)
            test_idx.append(test_idx_i)
    elif 'loo' in eval_mode:
        loo = LeaveOneOut()
        train_idx, test_idx = [], []
        for train_idx_i, test_idx_i in loo.split(range(len(raw_dataset))):
            train_idx.append(train_idx_i)
            test_idx.append(test_idx_i)
    elif 'full' in eval_mode:
        train_idx, test_idx = [list(range(len(raw_dataset)))], [list(range(len(raw_dataset)))]
    else:
        raise ValueError('Not valid eval mode')
    dataset_ = []
    for i in range(len(train_idx)):
        dataset_i = {'train': split_dataset(raw_dataset, train_idx[i]), 'test': split_dataset(raw_dataset, test_idx[i])}
        dataset_.append(dataset_i)
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    def add_(input_, key=None):
        split_names = key.split('.')
        current = batch
        for split_name in split_names[:-1]:
            if split_name not in current:
                current[split_name] = {}
            current = current[split_name]
        if split_names[-1] not in current:
            current[split_names[-1]] = input_.unsqueeze(0)
        else:
            current[split_names[-1]] = torch.cat([current[split_names[-1]], input_.unsqueeze(0)], dim=0)
        return

    batch = {}
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    for i in range(len(input)):
        input_i = input[i]
        apply_recursively(add_, input_i, apply_condition=apply_condition, identity_condition=identity_condition)
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset, model=None, tokenizer=None):
    processed_dataset = dataset
    if cfg['num_shots'] != -1:
        split_index = torch.randperm(len(dataset['train']))[:cfg['num_shots']]
        dataset['train'] = split_dataset(dataset['train'], split_index)

    cfg['sample_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    if cfg['data_mode'] == 'numeric':
        cfg['model']['data_size'] = processed_dataset['train'].meta['numeric']['data_size']
        cfg['model']['target_size'] = processed_dataset['train'].meta['numeric']['target_size']
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    if cfg['model_name'] in ['ridge', 'ann', 'svm', 'rf', 'gb', 'gp', 'dt']:
        cfg['num_steps'] = None
    return processed_dataset


def split_dataset(dataset, idx):
    dataset_ = copy.deepcopy(dataset)
    dataset_.id = [dataset.id[s] for s in idx]
    dataset_.data = [dataset.numeric[s] for s in idx]
    dataset_.data = [dataset.semantic[s] for s in idx]
    dataset_.target = [dataset.target[s] for s in idx]
    return dataset_
