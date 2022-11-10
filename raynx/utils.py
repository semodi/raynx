from itertools import product
from typing import Iterable


def batch(iterable: Iterable, n: int = 1):
    """Iterate in batches of size n"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def inner_iter(iterables: list[Iterable]):
    """Iterate over inner product of iterables, checks that all iterables
    have the same length"""
    if not all(len(iterables[0]) == len(iter) for iter in iterables):
        raise RuntimeError("Iterable length mismatch")
    return tuple(zip(*iterables))


def outer_iter(iterables: list[Iterable]):
    """Iterate over outer product of iterables, checks that all iterables
    have the same length"""
    prod = list(product(*iterables))
    return prod


iter_modes = {"inner": inner_iter, "outer": outer_iter}
