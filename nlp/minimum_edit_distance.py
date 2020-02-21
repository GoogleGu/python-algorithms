# Dynamic programming dynamic programming is the name for a class of algorithms, first introduced by Bellman (1957),
# that apply a table-driven method to solve problems by combining solutions to sub-problems.
# References:
#   1.Speech and Language Processing pp26
#   2.https://algorithms.tutorialhorizon.com/dynamic-programming-edit-distance-problem/


EDIT_COST = {
    'delete': 1,
    'insert': 1,
    'substitute': 2,
}


def minimum_edit_distance(source, target):

    n, m = len(source), len(target)
    distances = [[0 for _ in range(m+1)] for _ in range(n+1)]

    distances[0][0] = 0
    for i in range(1, n+1):
        distances[i][0] = distances[i-1][0] + EDIT_COST['delete']
    for j in range(1, m+1):
        distances[0][j] = distances[0][j-1] + EDIT_COST['insert']

    for i in range(1, n+1):
        for j in range(1, m+1):
            del_cost = distances[i-1][j] + EDIT_COST['delete']
            ins_cost = distances[i][j-1] + EDIT_COST['insert']
            if source[i-1] != target[j-1]:
                sub_cost = distances[i-1][j-1] + EDIT_COST['substitute']
            else:
                sub_cost = distances[i-1][j-1]
            distances[i][j] = min(del_cost, ins_cost, sub_cost)
            # print(source[:i-1], target[:j-1])
            # print_table(source, target, distances)
    return distances[n][m]


def print_table(source, target, table):
    source = '#' + source
    target = ' #' + target
    print('  '.join(target))

    for i in range(len(table)):
        print(source[i], table[i])
    print('-' * len(target) * 4)


if __name__ == '__main__':
    print(minimum_edit_distance('abcde', 'cdef'))