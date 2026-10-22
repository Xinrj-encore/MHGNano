# Hyperedge-Size Distribution

`hyperedge_size_by_species.csv` contains the maximal-clique size counts used to prepare the appendix table. The `source` column is relative to the historical `meta/data/` directory; the input pickle files are not included in this repository.

To regenerate the table from those trusted pickle files, run from `github_repo/`:

```bash
python scripts/audit_hyperedge_sizes.py \
  --data-root /path/to/MHGNano/meta/data \
  --output reproduction/hyperedge_size_distribution/hyperedge_size_by_species.csv
```

The required source paths and evaluated species are defined in `scripts/audit_hyperedge_sizes.py`. This script reads only the clique metadata (`num_vertices`, `PPI_edge_list`, and `PPI_cliques_list`); it does not alter the inputs.
