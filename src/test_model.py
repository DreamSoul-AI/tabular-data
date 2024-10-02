import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import save, resume, to_device, process_control, make_shap, viz_shap

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
    shap_list = []
    for i in range(len(dataset)):
        dataset_i = process_dataset(dataset[i])
        model_i = make_model(cfg['model'], i)
        model_i = model_i.to(cfg['device'])
        model_i.load_state_dict(result['model'][i])
        data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'])
        result_i = test(data_loader['test'], model_i, test_logger)
        shap_i = make_shap(data_loader, model_i, cfg['model']['model_name'], cfg['device'])
        result_list.append(result_i)
        shap_list.append(shap_i)
    evaluation = test_logger.evaluate('test', 'full')
    test_logger.append(evaluation, 'test')
    info = {'info': ['Model: {} (full)'.format(cfg['tag']),
                     'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
    test_logger.append(info, 'test')
    print(test_logger.write('test'))
    test_logger.save(True)
    result = resume(cfg['checkpoint_path'])
    result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                     'test': test_logger.state_dict()}, 'result': result_list, 'shap': shap_list}
    save(result, cfg['result_path'])
    if cfg['viz_condition']:
        viz_shap(shap_list, cfg['viz_path'])
    return


def test(data_loader, model, logger):
    def gather_result(input, output):
        result['id'].append(input['id'])
        result['pred'].append(output['pred'])
        result['target'].append(input['target'])
        return

    with torch.no_grad():
        result = {'id': [], 'pred': [], 'target': []}

        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
            logger.add('test', input, output)
            gather_result(input, output)
        for k in result:
            result[k] = torch.cat(result[k], dim=0).detach().cpu().numpy()
    return result


if __name__ == "__main__":
    main()
