# a naive implementation of BPE in book <speech and language processing > page 19 chapter 1

from collections import Counter, defaultdict
from pprint import pprint


class BytePairEncoding:


    def __init__(self, words, k):
        words = [word + '_' for word in words]
        temp_dictionary = Counter(words)
        self.dictionary = []
        for key, val in temp_dictionary.items():
            self.dictionary.append([[c for c in key], val])
        self.vocabulary = set(c for word in words for c in word)


    def merge_token(self):
        new_tokens = defaultdict(lambda :0)
        for token, count in self.dictionary:
            for pre, post in zip(token, token[1:]):
                new_tokens[pre+post] += count
        new_voca = max(new_tokens.items(), key=lambda x: x[1])[0]
        self.vocabulary.add(new_voca)
        print(new_voca)
        for item in self.dictionary:
            item[0] = merge_token_in_word(item[0], new_voca)


def merge_token_in_word(word, seq):
    new_word = []
    index = 0
    while index < len(word)-1:
        token = word[index] + word[index+1]
        if token != seq:
            new_word.append(word[index])
            index += 1
        else:
            new_word.append(token)
            index += 2
    if index < len(word):
        new_word.append(word[index])
    return new_word









if __name__ == '__main__':
    words = [
        'low', 'low', 'low', 'low', 'low',
        'lowest', 'lowest',
        'newer', 'newer', 'newer', 'newer', 'newer', 'newer',
        'wider', 'wider', 'wider',
        'new', 'new',
    ]

    bpe = BytePairEncoding(words, 5)
    for i in range(10):
        bpe.merge_token()
    pprint(bpe.dictionary)
