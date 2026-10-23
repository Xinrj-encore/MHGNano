# Published Experimental Datasets

This directory contains the prepared model inputs used for the MHGNano PPI
and NAI experiments. Files were copied from the corresponding project data
directories without rewriting the split, network, feature, or hypergraph
contents. `MANIFEST.sha256` records checksums for the distributed files.

## Contents

| Directory | Contents |
| --- | --- |
| `zhenhe/` | Main eukaryotic PPI collection: `cerevisiae`, `melanogaster`, `mouse`, `rerio`, `elegans`, and `human` |
| `yuanhe/` | Main prokaryotic PPI collection: `1326`, `41514`, `77635`, `Actinobacteria`, and `208964` |
| `nano/` | NAI pair labels, nanobody/antigen sequences, mappings, features, hypergraph, and train/validation/test splits |
| `ratio_robustness/` | PPI class-ratio sensitivity inputs: pair-level 1:1, 1:3, 1:5 datasets and a legacy fold-0 1:9 reference |
| `ratio_robustness_pair_disjoint_1to9/` | Pair-disjoint PPI sensitivity inputs at a 1:9 positive-to-negative ratio |
| `zhenhe_pair_disjoint_20261014/` | Additional pair-disjoint eukaryotic split audit indices and manifests; uses networks and features from `zhenhe/` |
| `zhenhe_random_hyperedge/` | Randomized eukaryotic hyperedge controls for the clique-structure ablation; uses labels and features from `zhenhe/` |

The main PPI species directories include processed positive and
candidate-negative pair tables, protein-ID mappings, clique-derived
hypergraphs, supplied five-fold train/validation/test splits, and frozen ESM-2
feature tensors. Where available, 35M, 150M, and 650M feature variants accompany
the 8M inputs. Max/min pooling feature and split variants are not included in
this release. The supplied splits in `zhenhe/` and `yuanhe/` are the historical main-experiment
files and have not been re-sampled or made pair-disjoint for this release. See
[`../reproduction/protocol_audit/PAIR_SPLIT_AUDIT.md`](../reproduction/protocol_audit/PAIR_SPLIT_AUDIT.md)
for the audit. The class-ratio folders are self-contained species data views:
their network, mapping, clique, and feature files match the corresponding main
collection, while the split indices and sampling manifests define the
alternative sampling protocol. The 1:1, 1:3, and 1:5 splits use the
pair-level class-ratio construction. `ratio_robustness/ratio_9/` preserves a
legacy fold-0 1:9 reference and is not pair-disjoint. Some exploratory
`rat` files are also present in these folders; the default eukaryotic
sensitivity runner uses the five species listed in the experiment protocol.
The `zhenhe_pair_disjoint_20261014/` audit folder contains split indices and
manifests rather than duplicate feature tensors. The pair-disjoint 1:9
collection is self-contained. The randomized hyperedge collection contains
the corresponding main species assets, with only the hyperedge file replaced
by its randomized control.

PPI data is grouped according to the manuscript's domain roles:

| Domain | Meta-training | Validation | Held-out targets |
| --- | --- | --- | --- |
| Eukaryotic | `cerevisiae`, `melanogaster`, `mouse` | `rerio` | `elegans`, `human` |
| Prokaryotic | `1326`, `41514`, `77635` | `Actinobacteria` | `208964` |

The `.pt` feature tensors and `.pkl` graph/split files are tracked with Git
LFS. Use `git lfs install` before cloning or pulling to download their
contents. A checkout without Git LFS contains pointer files in place of these
artifacts.

## File Use

For the default 8M model, run the commands from `github_repo/` and set
`--data_root ./data/yuanhe` or `--data_root ./data/zhenhe`, as appropriate.
Each species directory preserves the filenames expected by the released
loader. The `nano/` directory is the target data for NAI evaluation. The larger
feature variants support the ESM-2 scale analysis. Max/min pooling artifacts
are not distributed. Hypergraph structure is stored in
`PPI_cliques_Hyperedge_*.pkl`, while `train_idx_*.pkl`, `val_idx_*.pkl`, and
`test_idx_*.pkl` contain five folds in `[pairs, labels]` form.

The sensitivity and control collections do not replace the historical main
splits. `ratio_robustness/ratio_{1,3,5}/` contains pair-level class-ratio
splits; `ratio_robustness/ratio_9/` is the separate historical 1:9 reference.
The `ratio_robustness_pair_disjoint_1to9/` collection keeps pairs disjoint
across train, validation, and test. `zhenhe_pair_disjoint_20261014/` is an
additional eukaryotic split audit, and `zhenhe_random_hyperedge/` substitutes
randomized hyperedges for the listed eukaryotic species.

The deterministic preparation workflow in `scripts/prepare_ppi_dataset.py`
creates a new dataset in an empty output directory. It does not overwrite or
recreate these historical split files.

## Provenance and Attribution

- PPI interaction data derives from STRING. STRING states that its data and
  downloads are available under [CC BY 4.0](https://string-db.org/cgi/access?footer_active_subpage=licensing).
- Protein features were generated from ESM-2 pretrained representations. See
  the [ESM repository](https://github.com/facebookresearch/esm) and cite the
  ESM-2 publication when using these features.
- NAI pair and sequence data derives from the public
  [DeepNano repository](https://github.com/ddd9898/DeepNano) and
  [DeepNano-data repository](https://github.com/ddd9898/DeepNano-data), based
  on SAbDab-nano. Cite Deng et al., *Nature Machine Intelligence* (2024),
  DOI: [10.1038/s42256-024-00940-5](https://doi.org/10.1038/s42256-024-00940-5).

The MIT license in the MHGNano code repository applies to its software; it
does not replace the terms or attribution requirements of upstream datasets
and pretrained models. Retain the source citations and follow the respective
upstream terms when redistributing or reusing these data.
