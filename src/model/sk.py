from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from .model import normalize
from .loss import make_loss


class SK:
    def __init__(self, model_name, stats, task_mode, index, **kwargs):
        super().__init__()
        self.task_mode = task_mode
        self.index = index
        self.data_mean = stats[index]['data'].mean
        self.data_std = stats[index]['data'].std
        if task_mode == 'regression':
            self.target_mean = stats[index]['target'].mean
            self.target_std = stats[index]['target'].mean
        if task_mode == 'regression':
            if model_name == 'ridge':
                self.core = Ridge(alpha=kwargs['regularization'])
            elif model_name == 'ann':
                self.core = MLPRegressor(hidden_layer_sizes=kwargs['hidden_size'], solver=kwargs['solver'])
            elif model_name == 'svm':
                self.core = SVR()
            elif model_name == 'rf':
                self.core = RandomForestRegressor()
            elif model_name == 'gb':
                self.core = GradientBoostingRegressor()
            elif model_name == 'gp':
                self.core = GaussianProcessRegressor()
            else:
                raise ValueError('Not valid model name')
        elif task_mode == 'classification':
            if model_name == 'ridge':
                self.core = LogisticRegression(C=1 / kwargs['regularization'])
            elif model_name == 'ann':
                self.core = MLPClassifier(hidden_layer_sizes=kwargs['hidden_size'], solver=kwargs['solver'])
            elif model_name == 'svm':
                self.core = SVC()
            elif model_name == 'rf':
                self.core = RandomForestClassifier()
            elif model_name == 'gb':
                self.core = GradientBoostingClassifier()
            elif model_name == 'gp':
                self.core = GaussianProcessClassifier()
        else:
            raise ValueError('Not valid task mode')

    def fit(self, input):
        output = {}
        x = self.normalize_input(input)
        self.core.fit(x.numpy(), input['target'].numpy())
        output['pred'] = input['target'].new_tensor(self.core.predict(x.numpy()))
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        self.normalize_output(input, output)
        return output

    def predict(self, input):
        output = {}
        x = self.normalize_input(input)
        output['pred'] = input['target'].new_tensor(self.core.predict(x.numpy()))
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        self.normalize_output(input, output)
        return output

    def normalize_input(self, input):
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        input['target'] = input['target'].view(-1)
        if self.task_mode == 'regression':
            input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        return x

    def normalize_output(self, input, output):
        if self.task_mode == 'regression':
            output['pred'] = normalize(output['pred'], self.target_std, self.target_mean)
            input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return

    def state_dict(self):
        return self.core

    def load_state_dict(self, model):
        self.core = model
        return


def sk(cfg, index):
    model_name = cfg['model_name']
    stats = cfg['stats']
    task_mode = cfg['task_mode']
    if model_name == 'ridge':
        kwargs = {'regularization': cfg[model_name]['regularization']}
    elif model_name == 'ann':
        kwargs = {'hidden_size': cfg['ann']['hidden_size'], 'solver': cfg['ann']['solver']}
    else:
        kwargs = {}
    model = SK(model_name, stats, task_mode, index, **kwargs)
    return model
