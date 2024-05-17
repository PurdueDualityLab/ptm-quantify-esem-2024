This folder contains the data used for several of the metrics, listed below.

| Claim | Metric | Datasets Used | Corresponding Figure | Files Used |
|-------|---------|---------|---------|---------|
| **C2: Popularity impacts PTM selection**        | Turnover of top packages over time     | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip), [PTMTorrent](data/PeaTMOSS_OLD.db), HF Model Metadata [[1]](data/PeaTMOSS_NEW.db)[[2]](data/PeaTMOSS_ANCIENT.db), Ecosyste.ms | Figure 6 | [turnover.ipynb](turnover.ipynb) |
| **C2: Popularity impacts PTM selection**        | Number of descendants of top packages  | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip)| Figure 7 |[number_DirectDescendantsToParentModels.py](research_quantitative-analysis-of-hf-main/src/number_DirectDescendantsToParentModels.py), [number_DirectDescendantsToDownloads.py](research_quantitative-analysis-of-hf-main/src/number_DirectDescendantsToDownloads.py), [descendents.py](descendents.py) | 
| **C3: Documentation quality influences selection** | Documentation quality score | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip), HF Model Metadata [[1]](data/bottom_1k_model_card_scores.parquet)[[2]](data/top_1k_model_card_scores.parquet) | Figure 8 | [model_cards.py](model_cards.py) |

