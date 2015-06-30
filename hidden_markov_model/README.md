// Supervised
./hmm --model /tmp/sup --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en45.train --rare 0 --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en45.dev

// Unsupervised Baum-Welch
./hmm --model /tmp/bw --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.dev --rare 5 --train --unsup bw --states 12 --emiter 500  --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en12.dev

// Unsupervised anchor
./hmm --model /tmp/anchor --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.dev --rare 5 --train --unsup anchor --states 12 --emiter 500 --fwiter 2000 --window 11 --context list --cand 100 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en12.dev --log /tmp/anchor_log
