# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to extract n-best parses from trees.
"""
import argparse
from bllipparser import RerankingParser
from bllipparser.RerankingParser import Tree

def extract_nbest_from_trees(args):
    """Extracts n-best parses from each tree in a file."""
    parser = RerankingParser.fetch_and_load(args.parser, verbose=False)
    parser.set_parser_options(nbest=args.n)
    with open(args.output_path, "w") as outfile:
        with open(args.trees_path, "r") as infile:
            for line in infile:
                if not line.strip(): continue
                gold_tree = Tree.trees_from_string(line)[0]
                gold_tree.label = args.label
                gold_tree_string = str(gold_tree)
                outfile.write(str(gold_tree) + "\n");
                sentence = " ".join(gold_tree.tokens())
                nbest = parser.parse(sentence, rerank=True)
                num_trees = 1  # Already have the gold tree.
                for item in nbest:
                    if num_trees >= args.n: break
                    tree = item.ptb_parse
                    tree.label = args.label
                    tree_string = str(tree)

                    if tree_string != gold_tree_string:
                        num_trees += 1
                        outfile.write(tree_string + "\n")
                outfile.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("trees_path", type=str, help="path to file with "
                           "a tree per line")
    argparser.add_argument("output_path", type=str, help="path to output")
    argparser.add_argument("--parser", type=str, default="WSJ+Gigaword-v2",
                           help="parser model: WSJ-PTB3, WSJ+Gigaword-v2, "
                           "SANCL2012-Uniform, etc. (default: %(default)s)")
    argparser.add_argument("--n", type=int, default=50,
                           help="number of candidate trees "
                           "(default: %(default)d)")
    argparser.add_argument("--label", type=str, default="TOP",
                           help="label for the top node of a tree "
                           "(default: %(default)s)")
    parsed_args = argparser.parse_args()
    extract_nbest_from_trees(parsed_args)
