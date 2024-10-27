import os
import torch
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    # data_names = ['Diabetes', 'Iris']
    data_names = ['Bank', 'Blood', 'CalHousingC', 'CalHousingR']
    # data_names = ['Bank']
    # eval_modes = ['0.9-holdout', '3-fold', '10-fold']
    eval_modes = ['3-fold']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    cfg['make_stats'] = True
    with torch.no_grad():
        for data_name in data_names:
            for eval_mode in eval_modes:
                cfg['control']['data_name'] = '-'.join([data_name])
                cfg['control']['eval_mode'] = eval_mode
                process_control()
                dataset = make_dataset(cfg['data_name'], cfg['eval_mode'])
                stats = []
                for i in range(len(dataset)):
                    dataset_i = process_dataset(dataset[i])
                    cfg['step'] = 0
                    data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'],
                                                   shuffle=False)
                    stats_i = {'data': Stats(dim=dim)}
                    if cfg['model']['task_mode'] == 'regression':
                        stats_i['target'] = Stats(dim=1)
                    for i, input in enumerate(data_loader['train']):
                        stats_i['data'].update(input['data'])
                        if cfg['model']['task_mode'] == 'regression':
                            stats_i['target'].update(input['target'])
                    stats.append(stats_i)
                    if cfg['model']['task_mode'] == 'regression':
                        print('Name: {}\nData:\n{}\nTarget:\n{}'.format(cfg['control']['data_name'],
                                                                        stats_i['data'], stats_i['target']))
                    else:
                        print('Name: {}\nData:\n{}'.format(cfg['control']['data_name'],
                                                           stats_i['data']))
                makedir_exist_ok(stats_path)
                save(stats, os.path.join(stats_path, '{}_{}'.format(data_name, eval_mode)), 'torch')
