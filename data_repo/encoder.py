# Taken from my own work at:
# https://github.com/harikuts/del-step-sim/blob/master/global_encoder.py.


import argparse
import io
import itertools
import numpy as np
import pickle

def load_corpus(filenames):
    corpus = []
    for filename in filenames:
        with io.open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                corpus.append(line.split())
    return corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", "-c", required=True, action="store", \
        help = "Corpus of tweets to load.", nargs='+')
    parser.add_argument("--out", "-o", required=False, action="store", default="encoder.pkl", \
        help = "Output filename to store encoder.")
    args = parser.parse_args()
    # Load corpus
    corpus = load_corpus(args.corpus)
    # Assemble tokenizer
    all_words = list(itertools.chain.from_iterable(corpus))
    unique_words = np.unique(all_words)
    unique_word_index = dict((c, i) for i, c in enumerate(list(unique_words)))
    with io.open(args.out, 'wb') as f:
        pickle.dump(unique_word_index, f)