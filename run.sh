# from repo root
source .venv/bin/activate

# --- Steps 1-3: cwd must be src/ ---
cd src

python pipeline/1_node_extraction_bert.py 2>&1 | tee ../logs/1_node_extraction.log
python pipeline/2_graph_construction.py   2>&1 | tee ../logs/2_graph_construction.log
python pipeline/3_kg_embedding.py         2>&1 | tee ../logs/3_kg_embedding.log

# --- Step 4: cwd must be repo root ---
cd ..
python src/main.py --dataset AAAI 2>&1 | tee logs/4_main_train.log
