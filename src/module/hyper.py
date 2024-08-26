from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    cfg['batch_size'] = 256
    cfg['step_period'] = 1
    cfg['num_steps'] = 80000
    cfg['eval_period'] = 200
    cfg['num_epochs'] = 20
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    model_name_list = cfg['model_name'].split('-')
    cfg['model']['model_name'] = model_name_list[0]
    data_shape = {'Diabetes': [10]}
    target_size = {'Diabetes': 1}
    cfg['model']['data_shape'] = data_shape[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats(cfg['control']['data_name'])

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
    cfg[tag]['optimizer']['lr'] = 1e-1
    cfg[tag]['optimizer']['momentum'] = 0
    cfg[tag]['optimizer']['weight_decay'] = 0
    cfg[tag]['optimizer']['nesterov'] = True if cfg[tag]['optimizer']['momentum'] != 0 else False
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
