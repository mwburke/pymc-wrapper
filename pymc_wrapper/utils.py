from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from pymc import distributions


def pymc_dist_from_name(dist_name):
    return getattr(distributions, dist_name)


def update_config_with_dist_funcs(config):
    for param, param_config in config['variable_params'].items():
        param_config['dist'] = pymc_dist_from_name(param_config['dist'])

    return config


def update_config_with_dist_names(config):
    for param, param_config in config['variable_params'].items():
        param_config['dist'] = param_config['dist'].__name__

    return config


def load_config_from_file(file_path):
    with open(file_path, 'r') as f:
        config = load(f, Loader=Loader)

    config = update_config_with_dist_funcs(config)

    return config


def save_config_to_file(config, file_path):
    config = update_config_with_dist_names(config)

    with open(file_path, 'w') as f:
        dump(config, f, Dumper=Dumper)


def update_params_from_trace(param_config, model, trace):
    """
    NOTE: this is hard coded to support some very basic distributions.
    Each distribution contains its own parameters, so this would need to be overhauled if
    you want to extend it to more complex functions rather than just Normal and HalfNormal values.
    """
    with model:
        if 'mu' in param_config['params']:
            param_config['params']['mu'] = float(trace.posterior[param_config['params']['name']].mean())
            param_config['params']['sigma'] = float(trace.posterior[param_config['params']['name']].std())
        else:
            param_config['params']['sigma'] = float(trace.posterior[param_config['params']['name']].mean())

    return param_config
