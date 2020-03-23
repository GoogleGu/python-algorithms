import math


def calc_entropy(datasets):
    entropy = 0
    N = sum(len(dataset) for dataset in datasets)
    for dataset in datasets:
        p = len(dataset) / N
        entropy += - p * math.log(p, 2)
    return entropy


