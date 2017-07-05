# Author: Karl Stratos (me@karlstratos.com)
"""
This module is used to extract n-best parses from trees. Note that we only use
(n-1) trees from the parser if the gold tree is already in the list.
"""
import argparse
from bllipparser import RerankingParser
from bllipparser.RerankingParser import Tree
from bllipparser import tokenize

def extract_nbest_from_trees(args):
    """Extracts n-best parses from each tree in a file."""
    assert(args.len <= 399)  # Hard-coded length restriction.
    parser = RerankingParser.fetch_and_load(args.parser, verbose=False)
    parser.set_parser_options(nbest=args.n)
    with open(args.output_path, "w") as outfile:
        with open(args.trees_path, "r") as infile:
            num_instances = 0
            num_instances_with_gold = 0
            avg_gold_rank = 0

            for line in infile:
                if not line.strip(): continue
                gold_tree = Tree.trees_from_string(line)[0]
                gold_tree.label = args.label
                gold_tree_string = str(gold_tree)
                if len(gold_tree.tokens()) > args.len:
                    # Somehow the gold tree sentence is longer than what we can
                    # handle: skip!
                    continue
                num_instances += 1
                outfile.write(gold_tree_string + "\n")
                sentence = " ".join(gold_tree.tokens())
                nbest = parser.parse(sentence, rerank=True)
                num_trees = 1  # Already have the gold tree.
                for i, item in enumerate(nbest):
                    if num_trees >= args.n: break
                    tree = item.ptb_parse
                    tree.label = args.label
                    tree_string = str(tree)

                    if tree_string != gold_tree_string:
                        num_trees += 1
                        outfile.write(tree_string + "\n")
                    else:
                        avg_gold_rank += i+1
                        num_instances_with_gold += 1

                outfile.write("\n")

            if (not args.quiet) and num_instances:
                print "{0} out of {1} ({2:.1f}%) had gold tree in " \
                    "{3}-best list".format(num_instances_with_gold,
                                           num_instances,
                                           float(num_instances_with_gold) *
                                           100.0 / num_instances, args.n)
                if num_instances_with_gold:
                    print "In those {0} instances, the average rank of " \
                        "the gold tree was {1:.1f}".format(
                        num_instances_with_gold, float(avg_gold_rank) /
                        num_instances_with_gold)

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
    argparser.add_argument("--len", type=int, default=399,
                           help="max sentence length to parse "
                           "(default: %(default)d)")
    argparser.add_argument("--quiet", action="store_true", help="no messages")
    parsed_args = argparser.parse_args()
    extract_nbest_from_trees(parsed_args)
