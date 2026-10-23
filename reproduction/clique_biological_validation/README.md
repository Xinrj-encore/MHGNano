# Clique Biological-Coherence Evidence

This directory contains the multi-species aggregate results and representative enriched cliques used to assess whether clique-derived hyperedges capture functionally coherent protein groups. The analysis evaluated up to 1,000 maximal cliques of size at least three per species and compared clique homogeneity with 100 same-size random protein sets per clique.

`multi_species_summary.md` reports annotation coverage, enrichment fraction, mean functional homogeneity, the matched-size random-control mean, and the Mann-Whitney comparison. `representative_enriched_cliques.csv` and its Markdown rendering list one illustrative enriched clique per species. These summaries are evidence for the tested data and annotations, not a claim that every clique is a curated physical complex.

To rerun the analysis, use `scripts/validate_cliques_biological.py` with locally obtained clique pickles, protein maps, GO annotation files, and the GO ontology as required. The raw STRING networks, source annotation databases, and prepared protein artifacts are not redistributed. Do not load pickle files from untrusted sources.
