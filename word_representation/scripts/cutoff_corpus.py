# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to replace rare words in a corpus with a special symbol.
"""
import argparse
from collections import Counter

def count_words(corpus_path, unknown_symbol):
    """Counts words in the corpus."""
    word_counts = Counter()
    num_words = 0
    with open(corpus_path, "r") as corpus_file:
        for line in corpus_file:
            tokens = line.split()
            for token in tokens:
                if token == unknown_symbol:
                    print("WARNING: unknown symbol {0} present in the corpus".
                          format(unknown_symbol))
                word_counts[token] += 1
                num_words += 1
    return (word_counts, num_words)

def cutoff_corpus(args):
    """Replaces rare words in a corpus with a special symbol."""
    (word_counts, num_words) = count_words(args.corpus_path, args.unknown)

    print("Corpus: {0} word types, {1} words".format(len(word_counts),
                                                     num_words))
    print("\tReplacing words that occur <= {0} times...".format(args.cutoff))
    cutoff_vocab = {}

    with open(args.corpus_path, "r") as corpus_file, \
            open(args.output_path, "w") as output_file:
        for line in corpus_file:
            tokens = line.split()
            for i in range(len(tokens)):
                token = tokens[i]
                new_token = token if word_counts[token] > args.cutoff \
                    else args.unknown
                output_file.write(new_token)
                if i < len(tokens) - 1:
                    output_file.write(" ")
                cutoff_vocab[new_token] = True
            output_file.write("\n")

    print("New corpus: {0} word types (including the unknown symbol)".format(
            len(cutoff_vocab)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus_path", type=str, help="path to input corpus")
    argparser.add_argument("cutoff", type=int, help="word types "
                           "occurring <= this number are replaced")
    argparser.add_argument("output_path", type=str, help="path to output")
    argparser.add_argument("--unknown", type=str, default="<?>",
                           help="unknown word symbol (default: %(default)s)")
    parsed_args = argparser.parse_args()
    cutoff_corpus(parsed_args)
