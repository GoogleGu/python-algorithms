
def recursive_merge_sort(data):

    def divide(start, end):
        length = end - start
        if length == 1:
            pass
        else:
            mid = (start + end) // 2
            left_start, left_end = divide(start, mid)
            right_start, right_end = divide(mid, end)
            merge(data, left_start, right_start, right_end)
            # print(data, start, end)

        return start, end

    divide(0, len(data))
    return data


def loop_merge_sort(data):

    size = 1
    while size < len(data):
        start_index = 0
        while start_index < len(data):
            right_end = start_index + size + size
            mid_index = min(start_index + size, len(data))
            merge(data, start_index, mid_index, min(right_end, len(data)))
            start_index = right_end
        size += size
    return data


def merge(data, left_start, mid, right_end):
    temp = []
    left_index, right_index, left_end = left_start, mid, mid
    for pos in range(left_start, right_end):
        if left_index == left_end:
            temp.append(data[right_index])
            right_index += 1
        elif right_index == right_end:
            temp.append(data[left_index])
            left_index += 1
        else:
            if data[left_index] <= data[right_index]:
                temp.append(data[left_index])
                left_index += 1
            else:
                temp.append(data[right_index])
                right_index += 1

    for i in range(len(temp)):
        data[left_start+i] = temp[i]


if __name__ == '__main__':

    sorted_data = loop_merge_sort([7, 6, 5, 4, 3, 2, 1, 0])
    print(sorted_data)



