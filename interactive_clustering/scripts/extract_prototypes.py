import argparse
import sys
from collections import Counter

def main(args):

    DELIM = "__<label>__"

    t2w = {}

    with open(args.data_path) as f:
        for line in f:
            toks = line.split()
            for tok in toks:
                word, tag = tok.split(args.delim)
                if not tag in t2w: t2w[tag] = Counter()
                t2w[tag][word] += 1

    prototypes = {}
    memory = {}

    for tag in t2w:
        prototypes[tag] = []
        v = sorted(t2w[tag].items(), key=lambda x: x[1], reverse=True)
        i = 0
        while i < len(v) and len(prototypes[tag]) < args.k:
            if not v[i][0] in memory:
                memory[v[i][0]] = True
                prototypes[tag].append(v[i][0])
            i += 1

    with open(args.out_path, 'w') as f:
        for tag in t2w:
            f.write(tag + " ")
            for i, word in enumerate(prototypes[tag]):
                f.write(word)
                if i < len(t2w[tag]) - 1: f.write(" ")
            f.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=str)
    argparser.add_argument("out_path", type=str)
    argparser.add_argument("--delim", type=str, default="__<label>__")
    argparser.add_argument("--k", type=int, default=3)
    parsed_args = argparser.parse_args()
    main(parsed_args)
