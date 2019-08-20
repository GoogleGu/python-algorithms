

def merge_sort(data):

    def divide(start, end):
        length = end - start
        if length == 1:
            pass
        else:
            mid = (start + end) // 2
            left_start, left_end = divide(start, mid)
            right_start, right_end = divide(mid, end)
            merge(data, left_start, left_end, right_start, right_end)
            # print(data, start, end)

        return start, end

    divide(0, len(data))
    return data


def merge(data, left_start, left_end, right_start, right_end):
    temp = []
    left_index, right_index = left_start, right_start
    # print(left_start, left_end, right_start, right_end)
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

    sorted_data = merge_sort([1,2,3,4,5,6,7,0])
    print(sorted_data)



