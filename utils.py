import numpy as np

def gen_parameters_from_log_space(low_value=0.0001, high_value=0.001, n_samples=5):
    """
    Generate a list of parameters by sampling uniformly from a logarithmic space
    
    E.g.
           [ x   x  x | x  x x   |  x  x x  | x  x   x ]
        0.0001      0.001       0.01       0.1         1
    
    Which will draw much more small numbers than larger ones.
    """
    a = np.log10(low_value)
    b = np.log10(high_value)
    r = np.sort(np.random.uniform(a, b, n_samples))
    return 10 ** r