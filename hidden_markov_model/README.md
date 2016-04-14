// Supervised
./hmm --output /tmp/sup_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.train --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --states 0

// Unsupervised cluster
./hmm --output /tmp/cluster_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train50 --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --unsup cluster --states 12 --cluster ../../scratch/induced_brown_clusters/en.train_c12_min1/paths

// Unsupervised Baum-Welch
./hmm --output /tmp/bw_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train50 --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --unsup bw --states 12 --lives 10

// Unsupervised anchor
./hmm --output /tmp/anchor_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train50 --train  --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev  --unsup anchor --states 12  --lives 3

// Unsupervised anchor with features
./hmm --output /tmp/anchorfeat_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train50 --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --unsup anchor --states 12 --extend cap,hyphen,digit,suff1,suff2,suff3 --extweight 0.1 --check 1 --lives 3
