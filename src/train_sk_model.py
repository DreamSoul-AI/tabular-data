import argparse
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import check, resume, process_control, gather_input

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['run_mode'] = 'train'
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join('output', 'logger', 'train', 'runs', cfg['tag'])
    dataset = make_dataset(cfg['data_name'], cfg['eval_mode'])
    result = resume(cfg['checkpoint_path'], resume_mode=cfg['resume_mode'])
    model = []
    if result is None:
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
    else:
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    for i in range(len(dataset)):
        dataset_i = process_dataset(dataset[i])
        model_i = make_model(cfg['model'], i)
        if result is None:
            cfg['step'] = 0
        else:
            cfg['step'] = result['cfg']['step']
            model_i.load_state_dict(result['model'][i])
        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'], cfg['num_steps'],
                                       cfg['step'], cfg['step_period'], cfg['pin_memory'], cfg['num_workers'],
                                       cfg['collate_mode'], cfg['seed'])
        train(data_loader['train'], model_i, logger, i)
        test(data_loader['test'], model_i, logger)
        model.append(model_i)
    evaluation = logger.evaluate('test', 'full')
    logger.append(evaluation, 'test')
    info = {'info': ['Model: {} (full)'.format(cfg['tag']), 'Test Epoch: 1(100%)']}
    logger.append(info, 'test')
    print(logger.write('test'))
    logger.save(True)
    result = {'cfg': cfg, 'model': [model_i.state_dict() for model_i in model], 'logger': logger.state_dict()}
    check(result, cfg['checkpoint_path'])
    if logger.compare('test'):
        shutil.copytree(cfg['checkpoint_path'], cfg['best_path'], dirs_exist_ok=True)
    logger.reset()
    return


def train(data_loader, model, logger, index):
    input = {}
    for i, input_i in enumerate(data_loader):
        for key, value in input_i.items():
            if key not in input:
                input[key] = []
            input[key].append(value)
    for key in input:
        input[key] = torch.cat(input[key], dim=0)
    input_size = input['data'].size(0)
    output = model.fit(input)
    evaluation = logger.evaluate('train', 'batch', input, output)
    logger.append(evaluation, 'train', n=input_size)
    info = {'info': ['Model: {} ({})'.format(cfg['tag'], index),
                     'Train Epoch: 1(100%)']}
    logger.append(info, 'train')
    print(logger.write('train'))
    return


def test(data_loader, model, logger):
    input = gather_input(data_loader)
    input_size = input['data'].size(0)
    output = model.predict(input)
    evaluation = logger.evaluate('test', 'batch', input, output)
    logger.append(evaluation, 'test', input_size)
    logger.add('test', input, output)
    return


if __name__ == "__main__":
    main()
