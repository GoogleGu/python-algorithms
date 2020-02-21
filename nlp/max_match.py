# a naive implementation of MaxMatch in book <speech and language processing > page 20 chapter 1


def recusive_max_match(string, tokens) -> list:

    if not string:
        return []

    for i in range(len(string), 0, -1):
        target = string[:i]
        remainder = string[i:]
        if target in tokens:
            sub_matches = recusive_max_match(remainder, tokens)
            return [target] + sub_matches
    return [string]


def iterative_max_match(string, tokens) -> list:
    result = []

    remainder = string
    while remainder:
        for i in range(len(string), 0, -1):
            target = string[:i]
            leftover = string[i:]
            if target in tokens:
                remainder = leftover
                result.append(target)
                break
        result.append(remainder)
        break

    return result



if __name__ == '__main__':
    string = 'intention'
    tokens = ["in", "tent","intent","##tent", "##tention", "##tion", "#ion"]
    result = iterative_max_match(string, tokens)
    print(result)
