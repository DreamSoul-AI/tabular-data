import os
import torch
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['Diabetes']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    cfg['make_stats'] = True
    with torch.no_grad():
        for data_name in data_names:
            cfg['control']['data_name'] = '-'.join([data_name])
            process_control()
            dataset = make_dataset(cfg['data_name'])
            cfg['step'] = 0
            data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'],
                                           shuffle=False)
            stats = {'data': Stats(dim=dim), 'target': Stats(dim=1)}
            for i, input in enumerate(data_loader['train']):
                stats['data'].update(input['data'])
                stats['target'].update(input['target'])
            print('Name: {}\nData:\n{}\nTarget:\n{}'.format(cfg['control']['data_name'],
                                                            stats['data'], stats['target']))
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, data_name), 'torch')
