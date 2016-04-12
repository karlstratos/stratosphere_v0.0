
./hmm --output /tmp/anchor_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train50 --train --unsup anchor --states 12 --window 3 --context list --hull brown --add 10 --power 0.5 --cand 200 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --check 1 --lives 1

// Supervised
./hmm --output /tmp/sup_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.train --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev

// Unsupervised cluster
./hmm --output /tmp/cluster_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --train --unsup cluster --states 12 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --cluster ../../scratch/induced_brown_clusters/en.train_c12_min1/paths

// Unsupervised Baum-Welch
./hmm --output /tmp/bw_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --train --unsup bw --states 12 --check 10 --lives 10 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev

// Unsupervised anchor
./hmm --output /tmp/anchor_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --train --unsup anchor --states 12 --window 3 --context list --hull brown --add 10 --power 0.5 --cand 200 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --check 1 --lives 1

// Unsupervised anchor with features
./hmm --output /tmp/anchorfeat_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --train --unsup anchor --states 12 --window 3 --hull brown --add 10 --power 0.5 --cand 200 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --check 1 --lives 1 --extend basic,pref1,pref2,pref3,suff1,suff2,suff3 --extweight 0.001
