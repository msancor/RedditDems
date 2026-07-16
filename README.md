# RedditDems

Code accompanying the paper on the sociodemographic composition of attention flows across Reddit communities (2019–2023). It builds an Attention-Flow Graph (AFG) between subreddits from Reddit submission data, projects sociodemographic scores (age, gender, partisanship, affluence) onto users and communities, and analyzes how different demographic groups move between communities.

## Repository contents

- **`main.ipynb`** — Main pipeline: builds the Attention-Flow Graph from raw Reddit submissions, rescales edge weights, projects user demographic scores, builds the demographic-weighted AFG, extracts shared URLs, and computes submission/author counts.
- **`subreddit_filter.ipynb`** — Filters out likely bot accounts (by posting breadth and username pattern) and filters subreddits to those with sufficient monthly activity and relevance to the study.
- **`plots.ipynb`** — Generates all paper figures and tables from the processed/shared data in `data_shared/`. This is the only notebook that can be run without the raw Reddit data.
- **`modules/`** — Python package with the core processing functions imported by the notebooks:
  - `spark_init.py` — configures and initializes the local PySpark context.
  - `text_extractor.py` — preprocesses raw submissions and extracts shared URLs/domains.
  - `network_constructor.py` — builds and rescales the Attention-Flow Graph.
  - `demographic_network_constructor.py` — projects demographic scores onto users and builds the demographic-weighted AFG.
  - `disparity_filter.py` — applies the disparity filter to prune non-significant edges from the graph (requires `graph-tool`).
  - `random_walks.py` — runs random walks on the graph to compute gateway/bridge and cross-community reachability statistics.
- **`source/`** — standalone scripts, run separately from the notebooks:
  - `infomap.py` — detects communities on the filtered AFG using the Infomap algorithm (via `igraph`).
  - `seed_communities.py` — uses the Reddit API (`praw`) to expand a list of seed subreddits into their surrounding community and fetch subscriber counts.
- **`data_shared/`** — input/output data shared via Zotero (see [data_shared/README.md](data_shared/README.md)); not fully included in this repository. `data_shared/reddit/` holds raw and intermediate data and is not shared at all.
- **`LICENSE`** — MIT license.

## Setting up the directories

The code expects the following layout, none of which is fully present in this repository:

```
RedditDems/
├── data_shared/
│   ├── reddit/                  # raw + intermediate data (NOT shared, you must create/populate)
│   │   └── submissions/{year}/RS_{year}-{month}.bz2
│   ├── main_data/               # from Zotero
│   └── supplementary_data/      # from Zotero
└── output/                      # figures saved by plots.ipynb (create empty)
```

1. **Shared/processed data** — download `main_data/` and `supplementary_data/` from the accompanying Zotero repository and place them under `data_shared/` (see [data_shared/README.md](data_shared/README.md) for file descriptions). This is enough to run `plots.ipynb` and reproduce all figures and tables.
2. **Raw Reddit data** — `main.ipynb` and `subreddit_filter.ipynb` additionally require the raw Reddit submission dumps, which are not included here due to their size. They are publicly available from the Pushshift archives. Download the monthly submission dumps for 2019–2023 and place them at `data_shared/reddit/submissions/{year}/RS_{year}-{month}.bz2`. These two notebooks also write large intermediate outputs back into `data_shared/reddit/`, which are then aggregated into the smaller files under `data_shared/supplementary_data/` and `data_shared/main_data/`.
3. **Figures output** — create an empty `output/` directory at the repository root before running `plots.ipynb`.

Running `main.ipynb` and `subreddit_filter.ipynb` also requires a working local PySpark installation (see `modules/spark_init.py`); `disparity_filter.py` and `random_walks.py` require `graph-tool`; `source/seed_communities.py` requires Reddit API credentials for `praw`.
