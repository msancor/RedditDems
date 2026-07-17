# Reddit Demographics

Code accompanying the paper *Conspiracy and Environment Communities on Reddit Sort Along Different Demographic Axes* (under submission). It builds an Attention-Flow Graph (AFG) between subreddits from Reddit submission data, projects sociodemographic scores (age, gender, partisanship, affluence) onto users and communities, and analyzes how different demographic groups move between communities.

## Repository contents

- **`main.ipynb`** — Main pipeline: builds the Attention-Flow Graph from raw Reddit submissions, rescales edge weights, projects user demographic scores, builds the demographic-weighted AFG, extracts shared URLs, and computes submission/author counts.
- **`subreddit_filter.ipynb`** — Filters out likely bot accounts (by posting breadth and username pattern) and filters subreddits to those with sufficient monthly activity and relevance to the study.
- **`plots.ipynb`** — Generates all paper figures and tables from the processed/shared data in `data_shared/`. This notebook can run seamlessly without raw Reddit data.
- **`modules/`** — Python modules with the core processing functions imported by the notebooks:
  - `spark_init.py` — configures and initializes the local PySpark context.
  - `text_extractor.py` — preprocesses raw submissions and extracts shared URLs/domains.
  - `network_constructor.py` — builds and rescales the Attention-Flow Graph.
  - `demographic_network_constructor.py` — projects demographic scores onto users and builds the stratified AFG.
  - `disparity_filter.py` — applies the disparity filter to prune non-significant edges from the graph (requires `graph-tool`).
  - `random_walks.py` — runs random walks on the graph to compute gateway/bridge and cross-community reachability statistics.
- **`source/`** — standalone scripts, run separately from the notebooks:
  - `infomap.py` — detects communities on the filtered AFG using the Infomap algorithm (via `igraph`).
  - `seed_communities.py` — uses the Reddit API (`praw`) to expand a list of seed subreddits into their surrounding community and fetch subscriber counts.
- **`data_shared/`** — input/output data shared via Zotero; not fully included in this repository. `data_shared/reddit/` holds raw and intermediate data and is not shared at all.
- **`LICENSE`** — MIT license.

## Setting up the directories

The code expects the following layout:

```
RedditDems/
├── data_shared/
│   ├── reddit/                  # raw + intermediate data (NOT shared, you must create/populate from PushShift)
│   │   └── submissions/{year}/RS_{year}-{month}.bz2
│   ├── main_data/               # from Zotero
│   └── supplementary_data/      # from Zotero
└── output/                      # figures saved by plots.ipynb (create empty)
```

1. **Shared/processed data** — download `main_data/` and `supplementary_data/` from the accompanying Zotero repository and place them under `data_shared/`. This is enough to run `plots.ipynb` and reproduce all figures and tables.
2. **Raw Reddit data** — `main.ipynb` and `subreddit_filter.ipynb` additionally require the raw Reddit submission dumps, which are not included here due to their size. They are publicly available from the [Pushshift dataset](https://arxiv.org/abs/2001.08435). Download the monthly submission dumps for 2019–2023 and place them at `data_shared/reddit/submissions/{year}/RS_{year}-{month}.bz2`. These two notebooks also write large intermediate outputs back into `data_shared/reddit/`, which are then aggregated into the smaller files under `data_shared/supplementary_data/` and `data_shared/main_data/`.
3. **Figures output** — create an empty `output/` directory at the repository root before running `plots.ipynb`.

All the versions of the code are indexed on Zenodo and can be cited with the following DOI: [![DOI](https://zenodo.org/badge/1302914514.svg)](https://doi.org/10.5281/zenodo.21416527)
