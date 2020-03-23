import random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def load_text(filename):
    lemmatizer = WordNetLemmatizer()
    with open(filename, "rb") as file:
        text = file.read().lower().decode('utf-8').strip()
        stop_words = set(stopwords.words('english'))
        text_list = text.split(' ')

        text_list_without_stopwords = [lemmatizer.lemmatize(text_list[i], 'v') for i in range(len(text_list)) if text_list[i] not in stop_words]
        text_without_stopwords = ' '.join(text_list_without_stopwords)

        sent_list = sent_tokenize(text_without_stopwords)
        sent_list_tokenized = [word_tokenize(s) for s in sent_list]

        w2i = dict()
        i2w = dict()
        for
        return sent_list_tokenized

def generate_negative_samples(vocab, center_word, n):
    samples = []
    count = 0
    while count < n:
        rand_word = vocab[random.randint(len(vocab))]
        if rand_word == center_word:
            samples.append((center_word, rand_word))
            count += 1
    return samples


def gather_samples(split_text, vocab, window=2, negative_rate=2):
    positive_samples = []
    negative_samples = []
    N = len(split_text)
    windows = list(range(-window, window+1))
    windows.remove(0)
    for i in range(N):
        center_word = split_text[i]
        for w in windows:
            if i+w < 0 or i+w >= N:
                continue
            context_word = split_text[i+w]
            positive_samples.append((center_word, context_word))
        negative_samples.extend(generate_negative_samples(vocab, center_word, window * negative_rate))

    return positive_samples, negative_samples
