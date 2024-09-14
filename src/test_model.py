import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import save, resume, to_device, process_control

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
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    dataset = make_dataset(cfg['data_name'], cfg['eval_mode'])
    for i in range(len(dataset)):
        cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint', str(i))
        cfg['best_path'] = os.path.join(cfg['tag_path'], 'best', str(i))
        cfg['logger_path'] = os.path.join('output', 'logger', 'test', 'runs', cfg['tag'], str(i))
        cfg['result_path'] = os.path.join('output', 'result', cfg['tag'], str(i))
        dataset_i = process_dataset(dataset[i])
        model = make_model(cfg['model'], i)
        result = resume(cfg['best_path'])
        cfg['step'] = result['cfg']['step']
        model = model.to(cfg['device'])
        model.load_state_dict(result['model'])
        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'])
        test_logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
        result_i = test(data_loader['test'], model, test_logger, i)
        result = resume(cfg['checkpoint_path'])
        result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                         'test': test_logger.state_dict()}, 'result': result_i}
        save(result, cfg['result_path'])
    return


def test(data_loader, model, logger, index):
    result = {'output': [], 'target': []}

    def gather_result(input, output):
        result['output'].append(output['target'].detach().cpu())
        result['target'].append(input['target'].detach().cpu())
        result['output'] = [torch.cat(result['output'], dim=0)]
        result['target'] = [torch.cat(result['target'], dim=0)]
        return

    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            gather_result(input, output)
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
            logger.add('test', input, output)
        evaluation = logger.evaluate('test', 'full')
        logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {} ({})'.format(cfg['tag'], index),
                         'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test'))
        logger.save(True)
    return result


if __name__ == "__main__":
    main()
