import numpy as np
import random
from .tools import check_bool_individual


def weighted_bool_individual(icls, weight, size):
    """
    Parameters
    ----------
    weight: float
        Weight of choosing a chromosome
    size:
        Number of elements create

    Returns
    -------
        List random (not uniform) bool values
    """
    if weight:
        choice = random.choices([0, 1], [1 - weight / size, weight / size], k=size)
    else:
        choice = random.choices([0, 1], k=size)

    # Make sure there is at least one value on true
    choice = check_bool_individual(choice)

    return icls(choice)


def np_weighted_bool_individual(icls, weight, size):
    """
    Parameters
    ----------
    weight: float
        Weight of choosing a chromosome
    size:
        Number of elements create

    Returns
    -------
        np.array random (not uniform) index values
    """

    if weight:
        choice = np.random.choice(size, size=weight, replace=False)
    else:
        choice = np.random.choice(size, size=size // 2, replace=False)

    return icls(choice)
