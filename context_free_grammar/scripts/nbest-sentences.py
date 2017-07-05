
# Author: Karl Stratos (me@karlstratos.com)
"""
This module is used to extract n-best parses from sentences.
"""
import argparse
from bllipparser import RerankingParser
from bllipparser import tokenize

def extract_nbest_from_sentences(args):
    """Extracts n-best parses from each sentence in a file."""
    assert(args.len <= 399)  # Hard-coded length restriction.
    parser = RerankingParser.fetch_and_load(args.parser, verbose=False)
    parser.set_parser_options(nbest=args.n)
    with open(args.output_path, "w") as outfile:
        with open(args.sentences_path, "r") as infile:
            num_sentences = 0
            num_sentences_skipped = 0

            for line in infile:
                tokens = tokenize(line)
                if not tokens: continue
                num_sentences += 1
                if len(tokens) > args.len:
                    num_sentences_skipped += 1
                    continue

                nbest = parser.parse(tokens, rerank=True)
                for item in nbest:
                    tree = item.ptb_parse
                    tree.label = args.label
                    outfile.write(str(tree) + "\n")
                outfile.write("\n")

            if (not args.quiet) and num_sentences_skipped:
                print "{0} out of {1} ({2:.1f}%) skipped for " \
                    "length > {3}".format(num_sentences_skipped, num_sentences,
                                          float(num_sentences_skipped) * 100.0 /
                                          num_sentences, args.len)

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
    argparser.add_argument("--len", type=int, default=399,
                           help="max sentence length to parse "
                           "(default: %(default)d)")
    argparser.add_argument("--quiet", action="store_true", help="no messages")
    parsed_args = argparser.parse_args()
    extract_nbest_from_sentences(parsed_args)
