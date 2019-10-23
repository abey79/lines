import numpy as np


def segment_list_equal(s1: np.ndarray, s2: np.ndarray) -> bool:
    if len(s1) != len(s2):
        return False

    set1 = {frozenset(tuple(c) for c in s) for s in s1}
    set2 = {frozenset(tuple(c) for c in s) for s in s2}
    return set1 == set2
