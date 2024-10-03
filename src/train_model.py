import argparse
import copy
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control

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
    result = resume(cfg['checkpoint_path'], resume_mode=cfg['resume_mode'])
    if result is None:
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
    else:
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    dataset = make_dataset(cfg['data_name'], cfg['eval_mode'])
    model = []
    optimizer = []
    scheduler = []
    for i in range(len(dataset)):
        dataset_i = process_dataset(dataset[i])
        model_i = make_model(cfg['model'], i)
        if result is None:
            cfg['step'] = 0
            model_i = model_i.to(cfg['device'])
            optimizer_i = make_optimizer(model_i.parameters(), cfg[cfg['tag']]['optimizer'])
            scheduler_i = make_scheduler(optimizer_i, cfg[cfg['tag']]['optimizer'])
        else:
            cfg['step'] = result['cfg']['step']
            model_i = model_i.to(cfg['device'])
            optimizer_i = make_optimizer(model_i.parameters(), cfg[cfg['tag']]['optimizer'])
            scheduler_i = make_scheduler(optimizer_i, cfg[cfg['tag']]['optimizer'])
            model_i.load_state_dict(result['model'][i])
            optimizer_i.load_state_dict(result['optimizer'][i])
            scheduler_i.load_state_dict(result['scheduler'][i])
        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'], None,
                                       cfg['step'], cfg['step_period'], cfg['pin_memory'], cfg['num_workers'],
                                       cfg['collate_mode'], cfg['seed'])
        test_logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
        best_model = copy.deepcopy(model_i)
        while cfg['step'] < cfg['num_steps']:
            train(data_loader['train'], model_i, optimizer_i, scheduler_i, test_logger, i)
            test(data_loader['test'], model_i, test_logger, i, 'batch')
            if test_logger.compare('train'):
                best_model.load_state_dict(copy.deepcopy(model_i.state_dict()))
            test_logger.reset()
        test_logger.metric.reset_best()
        test(data_loader['test'], best_model, logger, i, None, True)
        model.append(best_model)
        optimizer.append(optimizer_i)
        scheduler.append(scheduler_i)
    evaluation = logger.evaluate('test', 'full')
    logger.append(evaluation, 'test')
    info = {'info': ['Model: {} ({})'.format(cfg['tag'], 'full', True),
                     'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
    logger.append(info, 'test')
    print(logger.write('test'))
    logger.save(True)
    result = {'cfg': cfg, 'model': [model_i.state_dict() for model_i in model],
              'optimizer': [optimizer_i.state_dict() for optimizer_i in optimizer],
              'scheduler': [scheduler_i.state_dict() for scheduler_i in scheduler],
              'logger': logger.state_dict()}
    check(result, cfg['checkpoint_path'])
    if logger.compare('test'):
        shutil.copytree(cfg['checkpoint_path'], cfg['best_path'], dirs_exist_ok=True)
    return


def train(data_loader, model, optimizer, scheduler, logger, index, verbose=False):
    model.train(True)
    start_time = time.time()
    with logger.profiler:
        for i, input in enumerate(data_loader):
            if i % cfg['step_period'] == 0 and cfg['profile']:
                logger.profiler.step()
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            loss = 1 / cfg['step_period'] * output['loss']
            loss.backward()
            if (i + 1) % cfg['step_period'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            evaluation = logger.evaluate('train', 'batch', input, output)
            logger.append(evaluation, 'train', n=input_size)
            idx = cfg['step'] % cfg['eval_period']
            if idx % max(int(cfg['eval_period'] * cfg['log_interval']), 1) == 0 and (i + 1) % cfg['step_period'] == 0:
                step_time = (time.time() - start_time) / (idx + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['eval_period'] - (idx + 1)) * step_time))
                exp_finished_time = datetime.timedelta(
                    seconds=round((cfg['num_steps'] - (cfg['step'] + 1)) * step_time))
                info = {'info': ['Model: {} ({})'.format(cfg['tag'], index),
                                 'Train Epoch: {}({:.0f}%)'.format((cfg['step'] // cfg['eval_period']) + 1,
                                                                   100. * idx / cfg['eval_period']),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                logger.append(info, 'train')
                if verbose:
                    print(logger.write('train'))
            if (i + 1) % cfg['step_period'] == 0:
                cfg['step'] += 1
            if (idx + 1) % cfg['eval_period'] == 0 and (i + 1) % cfg['step_period'] == 0:
                break
    return


def test(data_loader, model, logger, index, mode=None, verbose=False):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
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
