# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to compute mutual information of adjacent clusters.
"""
import argparse
from collections import Counter
from math import log

def read_clusters(cluster_path):
    """Reads word clusters from a file with lines: [cluster] [word] [count]."""
    word_to_cluster = {}
    cluster_to_words = {}
    with open(cluster_path, "r") as cluster_file:
        for line in cluster_file:
            tokens = line.split()
            if len(tokens) > 0:
                word_to_cluster[tokens[1]] = tokens[0]  # "The" -> "0001110"
                if not tokens[0] in cluster_to_words:
                    cluster_to_words[tokens[0]] = {}
                cluster_to_words[tokens[0]][tokens[1]] = True
    return (word_to_cluster, cluster_to_words)

def count_cluster_cooccurrences(corpus_path, word_to_cluster, unknown_symbol):
    """Counts cluster cooccurrences in the given corpus."""
    num_samples = 0
    c1c2_count = Counter()
    c1_count = Counter()
    c2_count = Counter()

    prev = ""
    with open(corpus_path, "r") as corpus_file:
        for line in corpus_file:
            tokens = line.split()
            for token in tokens:
                if prev:
                    c1 = word_to_cluster[prev] if prev in word_to_cluster \
                        else word_to_cluster[unknown_symbol]
                    c2 = word_to_cluster[token] if token in word_to_cluster \
                        else word_to_cluster[unknown_symbol]
                    assert(c1 and c2)  # Must map every word to cluster.
                    num_samples += 1
                    c1c2_count[(c1, c2)] += 1
                    c1_count[c1] += 1
                    c2_count[c2] += 1
                prev = token

    return (c1c2_count, c1_count, c2_count, num_samples)

def compute_mi(args):
    """
    Computes mutual information of adjacent clusters C, C':

                                                       p(c,c')
         \E [ PMI(C,C') ] =  SUM       p(c,c') * log ----------
                            {c,c'}                    p(c) p(c')

    From N samples of cluster pairs, this is estimated as:

                                      #(c,c')           #(c,c') N
     \hat\E [ PMI(C,C') ] =  SUM       ------ *  log  -------------
                            {c,c'}       N             #(c) #(c')
    """
    (word_to_cluster, cluster_to_words) = read_clusters(args.cluster_path)

    (c1c2_count, c1_count, c2_count, num_samples) = \
        count_cluster_cooccurrences(args.corpus_path, word_to_cluster,
                                    args.unknown)

    mi = 0.0
    for (c1, c2) in c1c2_count:
        prob_c1c2 = c1c2_count[(c1, c2)] / num_samples
        pmi = log(float(c1c2_count[(c1, c2)]) * num_samples /
                  c1_count[c1] / c2_count[c2], 2)  # Base 2 log
        mi += prob_c1c2 * pmi

    print("Clustering:  {0} word types in {1} clusters".format(
            len(word_to_cluster), len(cluster_to_words)))
    print("Corpus    :  {0} samples of cluster pairs".format(num_samples))
    print("Mutual information:\n\t{0:.3f}".format(mi))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("cluster_path", type=str, help="path to clusters: "
                           "[cluster] [word] [count]")
    argparser.add_argument("corpus_path", type=str, help="path to corpus")
    argparser.add_argument("--unknown", type=str, default="<?>",
                           help="unknown word symbol (default: %(default)s)")
    parsed_args = argparser.parse_args()
    compute_mi(parsed_args)
