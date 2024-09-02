from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from .model import make_loss, normalize


class SK:
    def __init__(self, model_name, stats, task_mode):
        super().__init__()
        self.data_mean = stats['data'].mean
        self.data_std = stats['data'].std
        self.task_mode = task_mode
        if task_mode == 'regression':
            self.target_mean = stats['target'].mean
            self.target_std = stats['target'].mean
        if task_mode == 'regression':
            if model_name == 'svm':
                self.model = SVR()
            elif model_name == 'rf':
                self.model = RandomForestRegressor()
            elif model_name == 'gb':
                self.model = GradientBoostingRegressor()
            elif model_name == 'gp':
                self.model = GaussianProcessRegressor()
            else:
                raise ValueError('Not valid model name')
        elif task_mode == 'classification':
            if model_name == 'svm':
                self.model = SVC()
            elif model_name == 'rf':
                self.model = RandomForestClassifier()
            elif model_name == 'gb':
                self.model = GradientBoostingClassifier()
            elif model_name == 'gp':
                self.model = GaussianProcessClassifier()
        else:
            raise ValueError('Not valid task mode')

    def fit(self, input):
        output = {}
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        input['target'] = input['target'].view(-1)
        if self.task_mode == 'regression':
            input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        self.model.fit(x.numpy(), input['target'].numpy())
        output['target'] = input['target'].new_tensor(self.model.predict(x.numpy()))
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        if self.task_mode == 'regression':
            output['target'] = normalize(output['target'], self.target_std, self.target_mean)
            input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return output

    def predict(self, input):
        output = {}
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        input['target'] = input['target'].view(-1)
        if self.task_mode == 'regression':
            input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        output['target'] = input['target'].new_tensor(self.model.predict(x.numpy()))
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        if self.task_mode == 'regression':
            output['target'] = normalize(output['target'], self.target_std, self.target_mean)
            input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return output

    def state_dict(self):
        return self.model

    def load_state_dict(self, model):
        self.model = model
        return


def sk(model_name, stats, task_mode):
    model = SK(model_name, stats, task_mode)
    return model
