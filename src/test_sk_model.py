import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import save, resume, process_control

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
    cfg['run_mode'] = 'test'
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join('output', 'logger', 'test', 'runs', cfg['tag'])
    cfg['result_path'] = os.path.join('output', 'result', cfg['tag'])
    cfg['viz_path'] = os.path.join('output', 'viz', cfg['tag'])
    dataset = make_dataset(cfg['data_name'], cfg['eval_mode'])
    result = resume(cfg['best_path'])
    cfg['step'] = result['cfg']['step']
    test_logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
    result_list = []
    for i in range(len(dataset)):
        dataset_i = process_dataset(dataset[i])
        model_i = make_model(cfg['model'], i)
        model_i.load_state_dict(result['model'][i])
        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'])
        result_i = test(data_loader['test'], model_i, test_logger, i, None, True)
        result_list.append(result_i)
    evaluation = test_logger.evaluate('test', 'full')
    test_logger.append(evaluation, 'test')
    info = {'info': ['Model: {} (full)'.format(cfg['tag']), 'Test Epoch: 1(100%)']}
    test_logger.append(info, 'test')
    print(test_logger.write('test'))
    test_logger.save(True)
    result = resume(cfg['checkpoint_path'])
    result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                     'test': test_logger.state_dict()}, 'result': result_list}
    save(result, cfg['result_path'])
    return


def test(data_loader, model, logger, index, mode=None, verbose=False):
    input = {'numeric': {}}
    for i, input_i in enumerate(data_loader):
        for key, value in input_i['numeric'].items():
            if key not in input['numeric']:
                input['numeric'][key] = []
            input['numeric'][key].append(value)
    for key in input['numeric']:
        input['numeric'][key] = torch.cat(input['numeric'][key], dim=0)
    input_size = len(input['numeric']['data'])
    output = model.predict(input)
    if mode is None or mode == 'batch':
        evaluation = logger.evaluate('test', 'batch', input, output)
        logger.append(evaluation, 'test', input_size)
    if mode is None or mode == 'full':
        logger.add('test', input, output)
    info = {'info': ['Model: {} ({})'.format(cfg['tag'], index),
                     'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
    logger.append(info, 'test')
    if verbose:
        print(logger.write('test'))
    logger.save(True)
    return


if __name__ == "__main__":
    main()
