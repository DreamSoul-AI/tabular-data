from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['data_mode'] = cfg['control']['data_mode']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['eval_mode'] = cfg['control']['eval_mode']
    cfg['num_shots'] = cfg['control']['num_shots']

    if cfg['data_mode'] == 'numeric':
        cfg['batch_size'] = 256
    elif cfg['data_mode'] == 'semantic':
        cfg['batch_size'] = 2
    else:
        raise ValueError('Not valid data mode')
    cfg['step_period'] = 1
    cfg['num_steps'] = 30
    cfg['eval_period'] = 30
    cfg['eval'] = {}
    cfg['eval']['num_steps'] = 30
    # cfg['num_epochs'] = 30
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    model_name_list = cfg['model_name'].split('-')
    cfg['model']['model_name'] = model_name_list[0]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 0.5, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['kan'] = {'hidden_size': [128, 64]}
    cfg['model']['ridge'] = {'regularization': 1}
    cfg['model']['ann'] = {'hidden_size': (128, 64), 'solver': 'adam'}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats('{}_{}'.format(cfg['control']['data_name'], cfg['eval_mode']))
    cfg['model']['data_mode'] = cfg['data_mode']
    if cfg['data_name'] in ['CalHousingR']:
        cfg['model']['task_mode'] = 'regression'
    elif cfg['data_name'] in ['Bank', 'Blood', 'CalHousingC', 'Car', 'CreditG', 'Diabetes',
                              'Heart', 'Income', 'Jungle']:
        cfg['model']['task_mode'] = 'classification'
    else:
        raise ValueError('Not valid dataset name')
    cfg['model']['bert'] = {'hidden_size': 1024}
    cfg['model']['max_length'] = 8
    cfg['model']['mask_mode'] = 'target'

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 1e-4
    cfg[tag]['optimizer']['nesterov'] = True if cfg[tag]['optimizer']['momentum'] != 0 else False
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    cfg[tag]['optimizer']['warmup_ratio'] = 0
    return
