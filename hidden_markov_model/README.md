// Quick
./hmm --model /tmp/anchor --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.dev --rare 5 --train --unsup anchor --states 12 --window 5 --context list --hull brown --add 10 --power 0.5 --cand 100 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --check 1 --lives 3

// Supervised
./hmm --model /tmp/sup_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.train --rare 5 --train --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev

// Unsupervised cluster
./hmm --model /tmp/cluster_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --rare 5 --train --unsup cluster --states 50 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --cluster ../../scratch/brown_clusters/en.train__c50__min6/paths

// Unsupervised Baum-Welch
./hmm --model /tmp/bw_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --rare 5 --train --unsup bw --states 50 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev

// Unsupervised anchor
./hmm --model /tmp/anchor_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --rare 5 --train --unsup anchor --states 50 --window 11 --context list --hull brown --add 10 --power 0.5 --cand 100 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --check 1 --lives 3

// Unsupervised anchor with features
./hmm --model /tmp/anchorfeat_hmm --data ../data/sequence_labeling/part_of_speech/universal_treebanks/en/unlabeled/en.train --rare 5 --train --unsup anchor --states 50 --check 1 --lives 3 --window 11 --hull brown --cand 100 --dev ../data/sequence_labeling/part_of_speech/universal_treebanks/en/labeled/en-universal-tag.dev --extend basic,pref1,pref2,pref3,suff1,suff2,suff3 --extweight 0.01
