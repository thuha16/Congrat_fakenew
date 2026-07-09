# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Congrat: a Contrastive Multi-Knowledge-Graph Learning framework for fake news detection. News articles,
extracted entities, and LDA topics form a heterogeneous graph; a heterogeneous GNN (`Congrat` in
`src/model.py`) is trained with a contrastive loss between two knowledge-graph branches (Wikidata TransE
embeddings + a semantic/DBpedia branch) plus a supervised classification loss, to classify news as
real/fake. This working copy is being adapted for a masters thesis around the COVID-19 (AAAI-2021) dataset,
with an LLM/BERT (Sentence-Transformers) upgrade to feature extraction — see `readme.md` (Vietnamese) for
the thesis-specific run instructions and `readme_original.md` for the original paper's description.

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric pandas numpy spacy gensim scikit-learn networkx requests tqdm sentence-transformers
python -m spacy download en_core_web_sm
```

`requirements.txt` is a full conda-export lockfile from the original paper's env (Python 3.7, torch
1.11+cu102, torch-geometric 2.1.0) — treat it as a reference, not something to `pip install -r` directly.

## Pipeline

The COVID-19 dataset is built from raw CSVs into a `HeteroData` graph through four sequential scripts.
**Steps 1–3 must be run with cwd = `src/`** (they use `../data/...` relative paths); **step 4 must be run
with cwd = repo root** (`utils.py` uses `./data/...`). Mixing these up is the most common failure mode.

```bash
cd src
python pipeline/1_node_extraction_bert.py   # CSVs -> cleaned text, SpaCy entities, LDA topics, SBERT features
python pipeline/2_graph_construction.py     # builds the news/entity/topic NetworkX graph + edges
python pipeline/3_kg_embedding.py           # Wikidata entity linking + TransE/SBERT entity embeddings
cd ..
python src/main.py --dataset AAAI           # trains/evaluates the GNN (must use --dataset AAAI, not FakeNewsNet)
```

`pipeline/1_node_extraction_baseline.py` is a separate, non-BERT baseline pipeline (Doc2Vec entity/news
features, random entity init) for FakeNewsNet-style CSVs — not part of the COVID-19/BERT flow above.

(Previously `pipeline/2_graph_construction.py` hardcoded a `FakeNewsNet` data dir left over from an older
pipeline, and `utils.py:load_dataset("AAAI")` pointed at a directory name that didn't match what the
pipeline scripts actually wrote. Both have been fixed so all four steps agree on
`data/COVID19/completed_data/` with the `AAAI2021_COVID19_fake_news.*` filename prefix.)

Other bugs already fixed that a fresh checkout of an older commit would still hit:
- `model.py` imported `matplotlib.pyplot` and `torch_scatter`/several unused `torch_geometric` symbols that
  were never actually used in the file — removed rather than adding those packages as real dependencies.
- `pipeline/1_node_extraction_bert.py`'s `clean_text` didn't collapse internal whitespace, so a raw tweet
  with an embedded newline could produce a spaCy entity string containing a literal `\n`; writing that as a
  `mapindex.txt` key (`f"{old_idx}\t{new_idx}\n"`) split it across two lines and broke the tab-separated
  format on read. Fixed by collapsing all whitespace to single spaces in `clean_text`.
- Every node-type check in `utils.py`/the pipeline is a naive `node_id.isdigit()` (news) / `"topic_"` prefix
  (topic) / else (entity) heuristic — entity node IDs are the raw entity text, not a namespaced ID. A
  purely-numeric entity string (e.g. `"288"`, likely a mistagged case count) is therefore indistinguishable
  from a news node's digit-string ID and corrupts the graph parsing (`KeyError` in `load_dataset`). Fixed by
  filtering out purely-numeric entity strings during extraction in `1_node_extraction_bert.py`. If you add
  new entity sources, keep this filter or namespace entity IDs properly instead.
- Wikidata's search API (`search_wikidata` in `3_kg_embedding.py`) now rejects requests with no `User-Agent`
  header (HTTP 403, see phabricator T400119), and the function's bare `except: pass` silently swallowed
  this, so entity linking always returned 0 real matches and every `kg_entities` vector fell back to random
  init. Fixed by sending a `User-Agent` header.
- `3_kg_embedding.py` originally scanned a `../data/pretrained/wikidata5m_transe.txt` that never actually
  existed in this checkout (never downloaded, and not the actual filename OpenKE ships). It's been rewired
  to do real offline lookups against a locally downloaded copy of OpenKE's full Wikidata TransE release —
  `data/pretrained/Wikidata/knowledge graphs/entity2id.txt` (Q-node → row index) plus
  `data/pretrained/Wikidata/embeddings/dimension_100/transe/entity2vec.bin` (memmapped float32, 100-dim) —
  instead of a 512-dim text file. `utils.py`'s `kg_entities` tensor allocation was updated to 100-dim to
  match. See "Downloading Wikidata embeddings" below for how to obtain this data; without it, `kg_entities`
  silently falls back to random vectors (not an error, just check the "Wikidata: Tìm thấy X/Y" log line).
- The upstream "Optimize Model" change added `Dropout(p=0.5)` in `model.py` but nothing ever called
  `model.eval()` before `test()` in `main.py`, so dropout stayed active during evaluation too, injecting
  noise into every test-time metric. Fixed by adding `model.eval()` before the `test()` call in `main.py`.
- That same upstream change upgraded the SBERT model in both `1_node_extraction_bert.py` and
  `3_kg_embedding.py` from `all-MiniLM-L6-v2` (384-dim) to `all-mpnet-base-v2` (768-dim), and grew the
  semantic (`kg1_entities`) branch slice from the first 128 dims to the first 256 dims — but `utils.py`'s
  `entity_kg1_attr` tensor was still hardcoded to 128-dim, which would crash `load_dataset` with a shape
  mismatch the first time it ran against MPNet-generated data. Fixed by updating that allocation to 256-dim.
  Note this SBERT upgrade means `completed_data/` must be regenerated from **step 1** onward (not just
  steps 3–4) any time the SBERT model or embedding dimensions change, since news/entity/topic feature
  vectors are baked into the step-1 output files.

## Downloading Wikidata embeddings

`3_kg_embedding.py` needs a local copy of OpenKE's pre-trained Wikidata TransE embeddings to produce real
(non-random) `kg_entities` vectors. `download_wikidata.sh` (repo root) downloads and extracts OpenKE's full
Wikidata release (~13GB zip) into `data/pretrained/Wikidata/`, staging the download with resume-on-stall
support. It requires `python3 -m zipfile` for extraction since `unzip` isn't assumed to be installed. The
OpenKE Wikidata-5M(TransR) release is a separate, larger dataset (`download_transr.sh`) that turned out to
lack a usable entity-ID mapping for our purposes and isn't currently wired into the pipeline — the full
Wikidata release's `entity2id.txt` + `entity2vec.bin` pair (100-dim) is what's actually used.

`data/**/completed_data/` and `data/pretrained/` are gitignored — they hold large generated/downloaded
artifacts (SBERT features, the graph pickle, the >8GB Wikidata TransE embeddings) and are expected to be
absent from a fresh checkout.

## Architecture

**Heterogeneous graph** (`HeteroData`, built in `utils.load_dataset`): node types `news`, `entities`,
`topic`, `kg_entities` (Wikidata TransE, 100-dim), `kg1_entities` (semantic/DBpedia branch, MPNet-derived,
256-dim). Edge
types: `news-has-entities`, `news-on-topic`, `entities-similar-entities` (cosine sim > threshold), and two
parallel `news-has-kg_entities` / `news-has-kg1_entities` + `kg*_entities-to-entities` relations linking
each knowledge-graph branch back into the entity nodes. IDs across all node types are unified via a single
`mapindex.txt` produced by the pipeline.

**Model** (`Congrat` in `src/model.py`): three parallel stacks of `HeteroConv` (GATv2Conv per edge type) —
`self.convs` sees only the `kg_entities` branch, `self.convs1` sees only the `kg1_entities` branch, and
`self.convs2` sees both branches together. Each stack's `news` output is projected (`fc1`/`fc2` MLP) to
produce three views (`aug_one`, `aug_two`, `aug_three`) of the same news nodes. Training combines:
- a symmetric InfoNCE-style contrastive loss (`semi_loss`) between `aug_three` (both KGs) and each
  single-KG view, weighted by `--alpha`
- a supervised `CrossEntropyLoss` on the concatenation of all three views fed through `lin2`

Training/eval use a transductive train/test mask (`shuffle_data` in `utils.py`, `--train_ratio`/`--test_ratio`)
over the news nodes rather than separate train/test graphs. `main.py` reinitializes and retrains the model
10 times in a loop (line-by-line accumulation of results), appending each run's metrics to
`./Para_analysis.txt` from `test()` in `model.py`.

Key CLI flags (`src/main.py`): `--dataset` (`AAAI`/`FakeNewsNet`/`Liar`/`PAN2020`, currently only AAAI's
pipeline is implemented here), `--alpha` (contrastive vs. classification loss weight), `--hidden_channels`,
`--gnn_layers`, `--epochs`, `--learning_rate`.
