# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module contains various utility functions.
"""
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
