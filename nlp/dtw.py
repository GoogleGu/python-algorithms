import pprint


def dtw(s1, s2):

    distances = [[abs(e1-e2) for e2 in s2] for e1 in s1]
    pprint.pprint(distances)

    min_distance = 0
    path = []
    i = j = 0
    greedy_min_dis = distances[0][0]
    while i < len(s1) and j < len(s2):
        min_distance += greedy_min_dis
        path.append((i, j))
        left_dis = distances[i][j+1] if j+1 < len(s2) else 100000000
        down_dis = distances[i+1][j] if i+1 < len(s1) else 100000000
        diagnol_dis = distances[i+1][j+1] if j+1 < len(s2) and i+1 < len(s1) else 100000000
        greedy_min_dis = min(left_dis, down_dis, diagnol_dis)
        if greedy_min_dis == left_dis:
            j += 1
        elif greedy_min_dis == down_dis:
            i += 1
        else:
            i += 1
            j += 1
    return min_distance, path


if __name__ == '__main__':
    print(dtw([1, 1, 3, 3, 2, 4], [1, 3, 2, 2, 4, 4]))
