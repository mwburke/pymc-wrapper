from copy import deepcopy
from functools import partial

from pymc import Model, Normal, HalfNormal
from pymc import sample, set_data, sample_posterior_predictive, MutableData
from numpy import quantile, vstack, zeros, vectorize
from pickle import dump, load

from pymc_wrapper.utils import save_config_to_file, update_params_from_trace


class PymcModel:
    """
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = Model()
        self.trace = None
        self.var_dict = None
        self.data_dict = None

        self._generate_model()

    def check_is_fitted(self):
        return self.trace is not None

    def fit(self, X, Y):

        with self.model:
            if self.var_dict is None:
                self._generate_model(X, Y)

            else:
                for c in self.model_config['independent_vars']:
                    set_data({
                        c: X[c]
                    })
                set_data({'y': Y})

            if self.model_config['sample_params']:
                trace = sample(**self.model_config['sample_params'])
            else:
                trace = sample()

            self.trace = trace

            self._create_prediction_func()

    def predict(self, X, alpha=None):
        # For each independent variable, do set_data, then trace sampling and output results
        if not self.check_is_fitted():
            raise Exception('This model is not yet fitted')

        # If alpha is none, we can use our saved means for the parameters to
        # run the function quickly and save time for sampling the posterior
        if alpha is None:

            return self.quick_predict_func(**{c: X[c] for c in self.model_config['independent_vars']})

        else:
            with self.model:
                if self.var_dict is None:
                    self._generate_model(X, Y)

                for c in self.model_config['independent_vars']:
                    set_data({
                        c: X[c]
                    })

                # In order to avoid shape errors, we have to update our Y to be
                # the same length as our dependent variables
                # It doesn't matter the value we use, so we set it to zero
                set_data({'y': zeros(X.shape[0])})

                posterior_predictive = sample_posterior_predictive(
                    self.trace,
                    predictions = True
                )

            model_preds = posterior_predictive.predictions['Y_obs']

            mean_preds = model_preds.mean(dim=['chain', 'draw'])

            q = (1 - alpha) / 2
            lower_bounds = model_preds.quantile(q=q, dim=['chain', 'draw'])
            upper_bounds = model_preds.quantile(q=1 - q, dim=['chain', 'draw'])

            predictions = vstack((lower_bounds, mean_preds, upper_bounds))

            return predictions

    def _create_prediction_func(self):
        if not self.check_is_fitted():
            raise Exception('This model is not yet fitted')

        mean_variable_vals = dict()

        for var_name in self.model_config['variable_params'].keys():
            mean_variable_vals[var_name] = float(self.trace.posterior[var_name].mean())

        self.quick_predict_func = vectorize(partial(self.model_config['function_params']['function'], **mean_variable_vals))

    def _generate_model(self):
        with self.model:
            self.var_dict = dict()
            self.data_dict = dict()

            for variable, var_config in self.model_config['variable_params'].items():
                self.var_dict[variable] = var_config['dist'](**var_config['params'])

            for c in self.model_config['independent_vars']:
                self.data_dict[c] = MutableData(c, [0.])

            # Expected value of outcome
            mu = self.model_config['function_params']['function'](
                **self.data_dict,
                **self.var_dict
            )

            y = MutableData('y', [0.])

            sigma = HalfNormal("sigma", sigma=1)
                # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=sigma, observed=y)

    def export_trained_config(self, file_path):
        if not self.check_is_fitted():
            raise Exception('This model is not yet fitted')

        config = deepcopy(self.model_config)

        for name, params in config['variable_params'].items():
            config['variable_params'][name] = update_params_from_trace(params, self.model, self.trace)

        del config['function_params']

        save_config_to_file(config, file_path)

    def save_trace(self, file_path):
        with open(file_path, 'wb') as f:
            dump({'trace': self.trace}, f)

    def load_trace(self, file_path):
        with open(file_path, 'rb') as f:
            model_load = load(f)

        self.trace = model_load['trace']

        self._create_prediction_func()
