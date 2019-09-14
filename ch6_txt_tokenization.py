import numpy as np

def tokenize_word():
    samples = ['The cat sat on the mat.',
               'The dog ate my homework.']

    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples),
                                  max_length,
                                  max(token_index.values())+1))
    # index=0 is reserved for unknown


    for i, sample in enumerate(samples):
        tmp = list(enumerate(sample.split()))
        for j, word in tmp[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1

    print(results)
    print(token_index)


import string

def tokenize_chars():
    samples = ['The cat sat on the mat.']
    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters)+1)))

    max_length = 100

    results = np.zeros((len(samples),
                        max_length,
                        max(token_index.values())+1))

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1

    print(results)
    print(token_index)

from keras.preprocessing.text import Tokenizer

def keras_tokenizer():
    samples = ['The cat sat on the mat.',
               'The dog ate my homework.']

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index

    print(f'Found {len(word_index)} unique tokens.')


def hash_tokenizer():
    samples = ['The cat sat on the mat.',
               'The dog ate my homework.']
    dimensionality = 1000

    max_len = 10

    results = np.zeros((len(samples),
                        max_len,
                        dimensionality))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_len]:
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1


if __name__ == '__main__':
    # keras_tokenizer()
    hash_tokenizer()