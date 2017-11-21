./prune --tree data/trees/uni2en1000_all.brown --oracle data/en-original-tag.majority --num-proto 100 --out /tmp/out.txt
./prune --tree data/trees/uni2en1000_all.brown --proto data/en-original-tag-3majority.proto  --out propagation3.txt
python ~/work/data/universal_treebanks_v2.0/pos-version/en/labeled/eval.py ~/work/data/universal_treebanks_v2.0/pos-version/en/labeled/en-original-tag.all propagation3.txt

./icluster --proposed ../../brown-cluster-percy/universal2_en12/paths --desired ../../data/universal_treebanks_v2.0/pos-version/en/labeled/en-universal-tag.majority
