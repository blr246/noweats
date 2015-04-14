"""
Some utility methods.
"""
from collections import defaultdict


def counter(iterable):
    """ Return a dict of counts for items in iterable. """
    counts = defaultdict(int)
    for item in iterable:
        counts[item] += 1
    return counts
