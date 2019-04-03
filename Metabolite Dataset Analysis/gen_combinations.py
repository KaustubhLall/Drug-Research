### File - gen_combinations.py
from datacontainer import *


def generateSequences(n, k):
    '''
    Generates all possible combinations of integer sequences of length k using the first 0, ..., n integers.
    Returns a list of sequences. Each sequence is a tuple with k entries in it. The tuples are ordered as :
    [ (1, 2, 3, ...), (1, 3, 4, ...), ..., (2, 3, 4, ...), (2, 4, 5, ... ), ... ]
    :param n: size of chosing pool.
    :param k: size of each pool
    :return: list of all possible sequences.
    '''
    return list(choose_iter(list(range(n)), k))

def choose_iter(elements, length):
    """
    Recursive helper method to generate sequences. This is a generator function that can be called as an iterable.
    :param elements: Collection of elements to combine.
    :param length: Length of each combination.
    :return: generator
    """
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i + 1:len(elements)], length - 1):
                yield (elements[i],) + next
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next


