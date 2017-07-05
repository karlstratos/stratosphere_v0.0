# Author: Karl Stratos (me@karlstratos.com)
"""
This module is used to process the original Penn Treebank WSJ dataset into the
standard split of train/dev/test portions: the resulting files contain a single
tree per line. It does not perform any modification on the trees themselves.

Argument 1: [data path]
Argument 2: [output path]
"""
import os
import sys

def collapse_whitespaces(string):
    """
    Collapses consecutive whitespaces in the given string to a single space.
    """
    toks = string.split()
    new_string = " ".join(toks)
    return new_string

def gather_trees(dir_path):
    """
    Gathers trees in the given directory as a list of strings.
    """
    trees = []
    for treefile in os.listdir(dir_path):
        with open(os.path.join(dir_path, treefile), "r") as treefile_open:
            tree_string = ""
            for line in treefile_open.readlines():
                tree_string_segment = line[:-1]

                # Use the fact that there is a 1-1 mapping between lines
                # starting with "(" and trees.
                if line[0] == "(":
                    if tree_string.split():
                        tree = collapse_whitespaces(tree_string)
                        trees.append(tree)
                    tree_string = tree_string_segment
                else:
                    tree_string += tree_string_segment

            if tree_string.split():
                tree = collapse_whitespaces(tree_string)
                trees.append(tree)

    return trees

def write_split(trees_train, trees_dev, trees_test, output_path):
    """
    Writes the split data (train, dev, test) to the given location.
    """
    if not os.path.exists(output_path):
         os.makedirs(output_path)
    train_path = os.path.join(output_path, "penn-wsj-raw.train")
    sys.stderr.write("Writing {0} trees for training to {1}\n"
                     .format(len(trees_train), train_path))
    with open(train_path, "w") as train_path_open:
        for tree in trees_train:
            train_path_open.write(tree + "\n")

    dev_path = os.path.join(output_path, "penn-wsj-raw.dev")
    sys.stderr.write("Writing {0} trees for development to {1}\n"
                     .format(len(trees_dev), dev_path))
    with open(dev_path, "w") as dev_path_open:
        for tree in trees_dev:
            dev_path_open.write(tree + "\n")

    test_path = os.path.join(output_path, "penn-wsj-raw.test")
    sys.stderr.write("Writing {0} trees for final evaluation to {1}\n"
                     .format(len(trees_test), test_path))
    with open(test_path, "w") as test_path_open:
        for tree in trees_test:
            test_path_open.write(tree + "\n")

def create_standard_split(wsj_path, output_path):
    """
    Creates the standard split of training data (Sections 02-21), development
    data (Section 22), and test data (Section 23) given the path to the
    directory containing the original Penn WSJ dataset.
    """
    remaining_sections = [no for no in range(25)]
    trees_train = []
    trees_dev = []
    trees_test = []

    for subdir in os.listdir(wsj_path):
        subdir_num = int(subdir)
        remaining_sections.remove(subdir_num)

        # Ignore unused sections.
        if subdir_num < 2 or subdir_num > 23:
            sys.stderr.write("Skipping Section {0}\n".format(subdir_num))
            continue

        # If the section is used, read the trees in the directory.
        subdir_path = os.path.join(wsj_path, subdir)
        trees = gather_trees(subdir_path)

        sys.stderr.write("Read {0} trees from Section {1}\n"
                         .format(len(trees), subdir_num))

        if subdir_num >= 2 and subdir_num <= 21:
            trees_train.extend(trees)
        elif subdir_num == 22:
            trees_dev.extend(trees)
        else:
            trees_test.extend(trees)

    assert not remaining_sections  # checking all sections are addressed

    # Write the split data.
    write_split(trees_train, trees_dev, trees_test, output_path)

if __name__ == "__main__":
    # Path to input: original Penn WSJ sub-directories 00-24.
    WSJ_PATH = sys.argv[1]

    # Path to output: standard split of Penn WSJ.
    OUTPUT_PATH = sys.argv[2]

    create_standard_split(WSJ_PATH, OUTPUT_PATH)
