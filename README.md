# PyMC Wrapper

This is a POC library to explore the possibility of making the PyMC model interface more simple, reusable and closer to the scikit-learn paradigm. The main concept is that it relies on defining a model config to automatically generate the PyMC model and allowing the trace to be saved and loaded to avoid re-training. We imagine its use case to be deploying Bayesian models on a regular basis once exploratory work has established the model relationship.

Please check out the `example` folder and notebook for sample configuration files and code samples for saving/loading as well as training/prediction.

This has been tested with PyMC >=5.0.0 and pyyaml 6.0.
