def selection_sort(array):

    for i in range(len(array)):
        minimum_index = i
        for j in range(i+1, len(array)):
            if array[j] < array[minimum_index]:
                minimum_index = j
        array.swap(i, minimum_index)


def insertion_sort(array):

    for i in range(len(array)):
        for j in range(i, 0, -1):
            if array[j-1] > array[j]:
                array.swap(j, j-1)
            else:
                break


def shell_sort(array):

    n = len(array)
    h = 1
    while h < n/3:
        h = h*3 + 1
    while h >= 1:
        for i in range(0, n, h):
            for j in range(i, 0, -h):
                if array[j-h] > array[j]:
                    array.swap(j, j-h)
                else:
                    break
        h = h // 3


if __name__ == '__main__':
    numbers = [1, 3, 2, 5, 0, 12349, 122]
    shell_sort(numbers)
    print(numbers)
