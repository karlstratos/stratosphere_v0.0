# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to extract n-best parses from sentences.
"""
import argparse
from bllipparser import RerankingParser

def extract_nbest_from_sentences(args):
    """Extracts n-best parses from each sentence in a file."""
    parser = RerankingParser.fetch_and_load(args.parser, verbose=False)
    parser.set_parser_options(nbest=args.n)
    with open(args.output_path, "w") as outfile:
        with open(args.sentences_path, "r") as infile:
            for line in infile:
                if not line.strip(): continue
                nbest = parser.parse(line, rerank=True)
                for item in nbest:
                    tree = item.ptb_parse
                    tree.label = args.label
                    reranker_score = item.reranker_score
                    outfile.write(str(tree) + "\n");
                outfile.write("\n");

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("sentences_path", type=str, help="path to file with "
                           "a sentence per line")
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
    extract_nbest_from_sentences(parsed_args)
