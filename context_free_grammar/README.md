Use the Python script to split the original Penn WSJ data to standard
train-dev-test portions:

`python3 split_penn_treebank.py ../../data/parsing/penn-wsj/original /tmp/standard-split`

To convert a raw treebank to standard format, type variations of:

`./process_trees --raw ../data/parsing/penn-wsj/standard-split/penn-wsj-raw.train --trees /tmp/penn-wsj.train`

To train a model, type variations of:

`./grammar --trees ../data/parsing/penn-wsj/standard-split/penn-wsj.train --model /tmp/grammar --bin left --hor 0 --ver 0 --train`

To parse with a trained model, type variations of:

`./grammar --trees ../data/parsing/penn-wsj/standard-split/penn-wsj.dev10 --model /tmp/grammar --pred /tmp/pred --len 100 --decode viterbi`
