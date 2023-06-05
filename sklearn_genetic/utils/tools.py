import random
import numpy as np

def mutFlipBit(individual, indpb):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.

    Parameters
    ----------
    individual: Individual to be mutated.
    indpb: Independent probability for each attribute to be flipped.

    Returns
    -------
        A tuple of one individual.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    individual = check_bool_individual(individual)

    return (individual,)


def cxUniform(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.

    Parameters
    ----------
    ind1: The first individual participating in the crossover.
    ind2: The second individual participating in the crossover.
          Independent probability for each attribute to be exchanged.

    Returns
    -------
    returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    ind1 = check_bool_individual(ind1)
    ind2 = check_bool_individual(ind2)

    return ind1, ind2


def check_bool_individual(individual):
    """
    Makes sures there is no individual with all its values at zero
    """
    if sum(individual) == 0:
        index = random.choice(range(len(individual)))
        individual[index] = 1

    return individual


def np_mutFlipBit(individual, indpb, n_features):
    """Emulate the ``mutFlipBit`` function with index array instead of boolean list.
    This is done by selecting randomly features to be mutated (with probability of *indbp*)
    and re-selecting random new features to replaces the selected features.
    
    Parameters
    ----------
    individual: Individual to be mutated.
    indpb: Independent probability for each attribute to be flipped.
    n_features: Total number of features in the problem.

    Returns
    -------
        A tuple of one individual.

    This function uses the :func:`~numpy.random.random` `~numpy.random.choice` and `~numpy.unique` function from numpy
    :mod:`numpy` module.
    """
    mut_idx = np.random.random(individual.shape[0]) < indpb
    individual[mut_idx] = np.random.choice(n_features, np.sum(mut_idx))
    
    fit1 = individual.fitness
    individual = np.unique(individual)
    individual.fitness = fit1

    return (individual,)


def np_cxUniform(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.

    Parameters
    ----------
    ind1: The first individual participating in the crossover.
    ind2: The second individual participating in the crossover.
          Independent probability for each attribute to be exchanged.

    Returns
    -------
    returns: A tuple of two individuals.

    This function uses the :func:`~numpy.random.random` `~numpy.where` and `~numpy.unique` function from numpy
    :mod:`numpy` module.
    """
    
    min_size = min(ind1.shape[0], ind2.shape[0])
    cx_idx = np.where(np.random.random(min_size) < indpb)
    
    tmp = ind1[cx_idx]
    ind1[cx_idx] = ind2[cx_idx]
    ind2[cx_idx] = tmp
    
    fit1 = ind1.fitness
    ind1 = np.unique(ind1)
    ind1.fitness = fit1
    
    ind2 = np.unique(ind2)
    
    return ind1, ind2
