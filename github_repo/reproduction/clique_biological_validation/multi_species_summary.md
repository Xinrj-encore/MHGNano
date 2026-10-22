# Clique Biological Validation From Base Data

Positive PPI edges were defined by STRING combined_score > 500. Up to 10000 positive edges were used per species, matching the data construction rule in the manuscript pipeline.
Each species was evaluated using up to 1000 maximal cliques with size >= 3 and 100 same-size random controls per clique.

| Dataset | Species | Taxid | Pos. edges | Cliques >=3 | Evaluated | Annot. cov. (%) | Enriched (%) | Homog. | Random | Gain | MWU p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zhenhe | cerevisiae | 4932 | 9670 | 2946 | 1000 | 95.9 | 88.1 | 0.4237 | 0.1439 | 0.2799 | 5.00e-290 |
| zhenhe | elegans | 6239 | 10000 | 3454 | 1000 | 81.8 | 50.9 | 0.2532 | 0.1005 | 0.1527 | 0.0010 |
| zhenhe | melanogaster | 7227 | 10000 | 3096 | 1000 | 87.9 | 88.9 | 0.3281 | 0.0994 | 0.2287 | 2.75e-183 |
| zhenhe | mouse | 10090 | 10000 | 2559 | 1000 | 99.0 | 90.5 | 0.2513 | 0.1010 | 0.1503 | 6.27e-212 |
| zhenhe | rerio | 7955 | 10000 | 2243 | 1000 | 99.4 | 94.4 | 0.3640 | 0.1139 | 0.2501 | 3.49e-290 |
| yuanhe | 1326 | 1326 | 10000 | 3816 | 1000 | 89.5 | 81.6 | 0.3081 | 0.1433 | 0.1649 | 1.03e-179 |
| yuanhe | 208964 | 208964 | 10000 | 2229 | 1000 | 80.9 | 79.8 | 0.3508 | 0.0949 | 0.2559 | 2.16e-185 |
| yuanhe | 41514 | 41514 | 10000 | 3048 | 1000 | 96.1 | 75.0 | 0.3009 | 0.1233 | 0.1776 | 1.42e-142 |
| yuanhe | 77635 | 77635 | 10000 | 4453 | 1000 | 95.2 | 67.6 | 0.3582 | 0.1493 | 0.2088 | 1.72e-159 |
| yuanhe | Actinobacteria | 1803496 | 10000 | 2629 | 1000 | 90.8 | 82.0 | 0.4471 | 0.1476 | 0.2995 | 1.83e-196 |
