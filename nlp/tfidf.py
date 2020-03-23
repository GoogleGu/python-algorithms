from collections import defaultdict
import math


def tfidf(documents):

    N = len(documents)
    term_count = defaultdict(lambda :[1 for _ in range(N)])
    document_count = defaultdict(lambda :0)

    for i in range(N):
        for word in documents[i]:
            term_count[word][i] += 1
        for word in set(documents[i]):
            document_count[word] += 1
    for word, counts in term_count.items():
        term_count[word] = [math.log(count, 10) * math.log(N/document_count[word], 10) for count in counts]
    return term_count


if __name__ == '__main__':
    docs = [
        ['greed', 'is', 'good'],
        ['who', 'is', 'your', 'daddy'],
        ['angry', 'is', 'good'],
        ['good', 'good', 'study', 'day', 'day', 'up'],
    ]

    print(tfidf(docs))