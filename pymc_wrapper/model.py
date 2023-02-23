from copy import deepcopy
from functools import partial

from pymc import Model, Normal, HalfNormal
from pymc import sample, set_data, sample_posterior_predictive, MutableData
from numpy import quantile, vstack, zeros, vectorize
from pickle import dump, load

from pymc_wrapper.utils import save_config_to_file, update_params_from_trace


class PymcModel:
    """
    PyMC model wrapper object that mimics the scikit-learn fit/predict paradigm for ease of use.
    """

    def __init__(self, model_config):
        """
        Create model object from configuration dictionary and generate the PyMC model for reuse.

        Parameters
        ----------
        model_config : dict
            Dictionary of the model configuration used to generate the PyMC model including the
            variable names, types, parameters, independent variable names and dependent variable
            function definition.

            Here is a description of each component in the config:

            - independent_vars: a list of the names of each independent variable
            - sample_params: a dictionary of parameters to be passed into the pm.sample function
            - variable_params: a dictionary of PyMC variables to be used in the model
                - variable_dict: a dictionary defining a specific variable definition with
                                 its name as the variable_params key
                    - dist: str of the name of the PyMC distribution (case sensitive) to be used for the variable
                    - params: a dictionary of the parameters for the variable and the values to be used as
                              the model's priors
            - function_params: a dictionary of function parameters
                - function: a python function defining the relationship of independent variables
                               to the dependent variable

        Please see example/example_config.yaml for an example definition without the python function
        """
        self.model_config = model_config
        self.model = Model()
        self.trace = None
        self.var_dict = None
        self.data_dict = None

        self._generate_model()

    def check_is_fitted(self):
        """
        Check if the model object has already been fitted and can access a sample trace.
        """
        return self.trace is not None

    def fit(self, X, Y):
        """
        For a new set of observations, perform sampling to infer the variables' values.
        If the model has not been created yet, it will initialize it.

        After sampling, the model will be ready to use the predict function.

        This function can be called on multiple sets of observations, meaning that the user
        can pass in differebt sets of observations and perform fitting on each set.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas dataframe that contains columns for each of the independent variables
            specified in the model config
        Y : pandas.Series or numpy.array
            A one-dimensional array containing the values of the dependent variable values
            for each of the observations specified in X
        """
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
        """
        Used the learned variable distributions and input independent variables to predict outcomes
        for the dependent variable, and optional credible interval.

        If alpha is None, then a quick prediction function will be used to generate median
        outcome predictions without performing any sampling.

        If alpha is not None, then sampling will be used to generate median outcomes along with
        the credible interval defined by alpha.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas dataframe that contains columns for each of the independent variables
            specified in the model config
        alpha: float
            Alpha is an optional float value between 0 and 1 that defines the credible interval
            returned in the output function. The quantiles used for the CI will be
            (1 - alpha / 2) and 1 - (1 - alpha / 2).

        Returns
        -------
        np.array
            If alpha is None, then it wil be a 1xN dimensional array where N is the number of
            observations in X.
            If alpha is not None, then it will be a 3xN dimensional array where N is the number
            of observations in X, and the 1st, 2nd and 3rd columns of data are the lower bound
            of the credible interval, median outcome and upper bound of the credible interval
            respectively.
        """
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
        """
        Sample the posterior distribution, calculate the mean values for each variable
        and prepopulate the dependent variable function with these values. This function
        will return outcomes when passed in independent variable observations either in single
        or vectorized format.
        """
        if not self.check_is_fitted():
            raise Exception('This model is not yet fitted')

        mean_variable_vals = dict()

        for var_name in self.model_config['variable_params'].keys():
            mean_variable_vals[var_name] = float(self.trace.posterior[var_name].mean())

        self.quick_predict_func = vectorize(partial(self.model_config['function_params']['function'], **mean_variable_vals))

    def _generate_model(self):
        """
        Initialize PyMC model definition, variables, and mutable data definitions.
        """
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
        """
        Update the existing model configuration and override the original priors with the
        posterior values of the variables.

        Same format as the original model_config dict used to initialize the model object
        except with function_params removed.

        Parameters
        ----------
        file_path : str
            String location of the yaml file path to export the trained configuration
        """
        if not self.check_is_fitted():
            raise Exception('This model is not yet fitted')

        config = deepcopy(self.model_config)

        for name, params in config['variable_params'].items():
            config['variable_params'][name] = update_params_from_trace(params, self.model, self.trace)

        del config['function_params']

        save_config_to_file(config, file_path)

    def save_trace(self, file_path):
        """
        Pickles the model's trace object to the specified file path.

        Parameters
        ----------
        file_path : str
            String location of the yaml file path to export the trained configuration
        """
        with open(file_path, 'wb') as f:
            dump({'trace': self.trace}, f)

    def load_trace(self, file_path):
        """
        Loads a trace into the model object to leverage saved model sampling results
        and be ready for prediction without re-training.

        Parameters
        ----------
        file_path : str
            String location of the yaml file path to export the trained configuration
        """
        with open(file_path, 'rb') as f:
            model_load = load(f)

        self.trace = model_load['trace']

        self._create_prediction_func()
