# Author: Karl Stratos (me@karlstratos.com)
"""
This module is used to split a text file into even lines and odd lines.
"""
import argparse

def main(args):
    """Main"""
    odd_lines = []
    even_lines = []
    with open(args.data_path, "r") as data_file, \
            open(args.even_line_path, "w") as even_file, \
            open(args.odd_line_path, "w") as odd_file:
        line_num = 1
        for line in data_file:
            if line.split():  # Keep only non-empty lines.
                if line_num % 2 == 0:  # Even
                    even_file.write(line)
                else:  # Odd
                    odd_file.write(line)
                line_num += 1

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=str, help="path to data")
    argparser.add_argument("even_line_path", type=str, help="path to the "
                           "even line portion")
    argparser.add_argument("odd_line_path", type=str, help="path to the odd "
                           "line portion")
    parsed_args = argparser.parse_args()
    main(parsed_args)
