from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['eval_mode'] = cfg['control']['eval_mode']

    cfg['batch_size'] = 256
    cfg['step_period'] = 1
    cfg['num_steps'] = None
    cfg['eval_period'] = 200
    cfg['num_epochs'] = 200
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    model_name_list = cfg['model_name'].split('-')
    cfg['model']['model_name'] = model_name_list[0]
    data_shape = {'Bank': [16], 'Blood': [4], 'CalHousingC': [4], 'CalHousingR': [4],
                  'Car': [6], 'CreditG': [20], 'Diabetes': [8], 'Heart': [11], 'Income': [12], 'Jungle': [6]}
    target_size = {'Bank': 2, 'Blood': 2, 'CalHousingC': 2, 'CalHousingR': 1, 'Car': 4,
                   'CreditG': 2, 'Diabetes': 2, 'Heart': 2, 'Income': 2, 'Jungle': 2}
    cfg['model']['data_shape'] = data_shape[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 0.5, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['kan'] = {'hidden_size': [128, 64]}
    cfg['model']['ridge'] = {'regularization': 1}
    cfg['model']['ann'] = {'hidden_size': (128, 64), 'solver': 'adam'}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats('{}_{}'.format(cfg['control']['data_name'], cfg['eval_mode']))

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
    cfg[tag]['optimizer']['scheduler_name'] = 'None'
    cfg[tag]['optimizer']['warmup_ratio'] = 0
    return
