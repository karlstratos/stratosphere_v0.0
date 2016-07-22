# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to display similar words (in consine similarity).
"""
import argparse
from numpy import array
from numpy import dot
from numpy import linalg

def read_embeddings(embedding_path, no_counts, ignore_line1, normalize):
    """Reads word embeddings from various file formats."""
    embedding = {}
    dim = 0

    with open(embedding_path, "r") as embedding_file:
        line_num = 0
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                line_num += 1

                # Ignore the first line of the file?
                if ignore_line1 and (line_num == 1): continue

                if no_counts:
                    # Embeddings have no count column.
                    word = tokens[0]
                    starting_index = 1
                else:
                    # Embeddings have a count column (as the first column).
                    word = tokens[1]
                    starting_index = 2
                values = []
                for i in range(starting_index, len(tokens)):
                    values.append(float(tokens[i]))

                # Ensure that the dimension matches.
                if dim:
                    assert(len(values) == dim)
                else:
                    dim = len(values)

                # Set the embedding, normalize the length if specified so.
                embedding[word] = array(values)
                if normalize:
                    embedding[word] /= linalg.norm(embedding[word])

    return embedding, dim

def display_nearest_neighbors(args):
    """Interactively displays similar words (in consine similarity)."""
    # Need to normalize the vector length for computing cosine similarity.
    normalize = True
    embedding, dim = read_embeddings(args.embedding_path, args.no_counts,
                                     args.ignore_line1, normalize)
    print("Read {0} embeddings of dimension {1}".format(len(embedding), dim))

    while True:
        try:
            word = input("Type a word (or just quit the program): ")
            if not word in embedding:
                print("There is no embedding for word \"{0}\"".format(word))
            else:
                neighbors = []
                for other_word in embedding:
                    if other_word == word:
                        continue
                    cosine = dot(embedding[word], embedding[other_word])
                    neighbors.append((cosine, other_word))
                neighbors.sort(reverse=True)
                for i in range(min(args.num_neighbors, len(neighbors))):
                    cosine, buddy = neighbors[i]
                    print("\t\t{0:.4f}\t\t{1}".format(cosine, buddy))
        except (KeyboardInterrupt, EOFError):
            print()
            exit(0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file")
    argparser.add_argument("--num_neighbors", type=int, default=30,
                           help="number of nearest neighbors to display")
    argparser.add_argument("--no_counts", action="store_true", help="embeddings"
                           " don't have counts for the first column?")
    argparser.add_argument("--ignore_line1", action="store_true", help="ignore "
                           "the first line in the embeddings file?")
    parsed_args = argparser.parse_args()
    display_nearest_neighbors(parsed_args)
