# What do we know about Hugging Face? A systematic literature review and quantitative validation of qualitative claims

## Datasets

Data from HuggingFace, PeaTMOSS, PTMTorrent, and Ecosyste.ms were used in this paper.
Some of the data was augmented for this paper.

Where feasible, the data used has been included here.

| Dataset       | Source                 | Modified for this paper? | Copy Included in this Repo | Filename           |
|---------------|------------------------|--------------------------|----------------------------|--------------------|
| PeaTMOSS      | [Link](https://github.com/PeaTMOSS-MSR2024/MSR24-PeaTMOSS-Artifact) | Yes | Yes | peatmoss_data.json  |
| HF Model Metadata | [Link](https://huggingface.co/datasets/davanstrien/hf_model_metadata) | No | Yes | N/A |
| PTMTorrent    | [Link](https://github.com/Wenxin-Jiang/PTM-Torrent/) | No | Yes | ptmtorrent_data.csv |
| Ecosyste.ms   | [Link](https://packages.ecosyste.ms) | No  | No  | N/A |

### Ecosyste.ms

Due to the size of the Packages Dataset from Ecosyste.ms, it is not possible to include a copy of the data used in this dataset.
The snapshots used for the Metric of *Turnover of top packages over time* can be found [here](https://packages.ecosyste.ms/open-data).
The snapshots from 2024-03-01, 2023-10-22, 20203-0808, and 2023-11-09 were used.
When loaded into Postgres, they were named as ecosystems_{year}_{month}.

### PeaTMOSS

The PeaTMOSS dataset was augmented in order to improve the dataset.
The table *model_to_base_model* was augmented with data from [this dataset](https://huggingface.co/datasets/librarian-bots/hub_models_with_base_model_info).
Only links between models that were captured in PeaTMOSS were added.
The file PeaTMOSS_DIST.db.zip does not contain the GitHub metadata, as that information was not used in the study.
As a result, it is much more compact than the actual dataset.

## Metrics

The datasets and files used for each metric are present here:

| Claim | Metric | Datasets Used | Corresponding Figure | Files Used |
|-------|---------|---------|---------|---------|
| **C1: Transformers library increases accessibility** | Preservation rate of libraries to descendants | [PeaTMOSS](research_quantitative-analysis-of-hf-main/src/data/number_of_direct_descendant_models_per_download.csv) | Figure 5 | [proportion_DirectDescendantsToLibrary.py](research_quantitative-analysis-of-hf-main/src/proportion_DirectDescendantsToLibrary.py) |
| **C2: Popularity impacts PTM selection**        | Turnover of top packages over time     | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip), [PTMTorrent](data/PeaTMOSS_OLD.db), HF Model Metadata [[1]](data/PeaTMOSS_NEW.db)[[2]](data/PeaTMOSS_ANCIENT.db), Ecosyste.ms | Figure 6 | [turnover.ipynb](turnover.ipynb) |
| **C2: Popularity impacts PTM selection**        | Number of descendants of top packages  | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip)| Figure 7 |[number_DirectDescendantsToParentModels.py](research_quantitative-analysis-of-hf-main/src/number_DirectDescendantsToParentModels.py), [number_DirectDescendantsToDownloads.py](research_quantitative-analysis-of-hf-main/src/number_DirectDescendantsToDownloads.py), [descendents.py](descendents.py) | 
| **C3: Documentation quality influences selection** | Documentation quality score | [PeaTMOSS](data/PeaTMOSS_DIST.db.zip), HF Model Metadata [[1]](data/bottom_1k_model_card_scores.parquet)[[2]](data/top_1k_model_card_scores.parquet) | Figure 8 | [model_cards.py](model_cards.py) |

