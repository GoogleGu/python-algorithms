import random


def quick_sort(array):
    array = random.shuffle(array)
    sort_subarray(array, 0, len(array))


def partition(array, start, end):


    return 0


def sort_subarray(array, start, end):
    if start >= end:
        return
    k = partition(array, start, end)
    sort_subarray(array, start, k-1)
    sort_subarray(array, k+1, end)
