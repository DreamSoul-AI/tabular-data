import os
import shutil
import torch
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from module import save, Stats, makedir_exist_ok, process_control


def copy_dataset_description():
    # Define source and target directories
    source_dir = './dataset/description'
    target_base_dir = './data/TabLLM/processed'

    # Check if source and target directories exist
    if os.path.exists(source_dir) and os.path.exists(target_base_dir):
        # Loop through files in the source directory
        for file_name in os.listdir(source_dir):
            # Check if the current item is a file and ends with .txt
            if file_name.endswith('.txt'):
                # Derive the dataset name and target folder
                data_name = os.path.splitext(file_name)[0]
                target_dir = os.path.join(target_base_dir, data_name)

                # Check if the target subdirectory exists
                if os.path.exists(target_dir):
                    # Copy the file
                    source_file = os.path.join(source_dir, file_name)
                    target_file = os.path.join(target_dir, file_name)
                    shutil.copy(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")
                else:
                    print(f"Target directory '{target_dir}' does not exist. Skipping {file_name}.")
    return


if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    # data_names = ['Bank', 'Blood', 'CalHousingC', 'CalHousingR', 'Car', 'CreditG', 'Diabetes', 'Heart', 'Income',
    #               'Jungle']
    data_names = ['Bank']
    data_modes = ['numeric', 'semantic']
    # data_modes = ['semantic']
    # eval_modes = ['0.9-holdout', '3-fold', '10-fold']
    eval_modes = ['0.9-holdout']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    cfg['make_stats'] = True
    process = False
    with torch.no_grad():
        for data_name in data_names:
            for data_mode in data_modes:
                for eval_mode in eval_modes:
                    cfg['control']['data_name'] = '-'.join([data_name])
                    cfg['control']['data_mode'] = data_mode
                    cfg['control']['eval_mode'] = eval_mode
                    process_control()
                    dataset = make_dataset(cfg['data_name'], cfg['data_mode'], cfg['eval_mode'], process=process)
                    stats = []
                    for i in range(len(dataset)):
                        dataset_i = process_dataset(dataset[i])
                        cfg['step'] = 0
                        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'],
                                                       shuffle=False)
                        if data_mode == 'numeric':
                            stats_i = {'data': Stats(dim=dim)}
                            if cfg['model']['task_mode'] == 'regression':
                                stats_i['target'] = Stats(dim=1)
                            for i, input in enumerate(data_loader['train']):
                                stats_i['data'].update(input['data'])
                                if cfg['model']['task_mode'] == 'regression':
                                    stats_i['target'].update(input['target'])
                            stats.append(stats_i)
                            if cfg['model']['task_mode'] == 'regression':
                                print('Name: {}({})\nData:\n{}\nTarget:\n{}'.format(cfg['control']['data_name'],
                                                                                    cfg['control']['data_mode'],
                                                                                    stats_i['data'], stats_i['target']))
                            else:
                                print('Name: {}({})\nData:\n{}'.format(cfg['control']['data_name'],
                                                                       cfg['control']['data_mode'],
                                                                       stats_i['data']))
                        else:
                            print('Name: {} ({})'.format(cfg['control']['data_name'], cfg['control']['data_mode']))
                    makedir_exist_ok(stats_path)
                    save(stats, os.path.join(stats_path, '{}_{}'.format(data_name, eval_mode)), 'torch')
    copy_dataset_description()
