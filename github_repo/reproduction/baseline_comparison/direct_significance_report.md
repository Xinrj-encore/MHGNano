# Direct TRGH-PPI and HIGH-PPI significance tests

The tests use 30 direct, no-target-parameter-adaptation runs (seeds 0--29) and the original 30-seed MHGNano-few results.

Because the seed schedules are not matched, all comparisons use two-sided independent Mann--Whitney U tests. Welch t-tests are included as a sensitivity check. Benjamini--Hochberg correction is applied separately to the 60 Mann--Whitney p-values and 60 Welch p-values, including the Overall PPI summary.

The prokaryotic-to-eukaryotic direct runs use different target species for TRGH-PPI (Mus musculus) and HIGH-PPI (C. elegans), so no direct paired TRGH-versus-HIGH test is reported for that setting.

## Results

| Model | Setting | Metric | Mean difference | MW q | Welch q | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| TRGH-PPI | prokaryotic_to_prokaryotic | Precision | -0.2076 | 4.41931e-11 | 1.92524e-17 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_prokaryotic | Recall | +0.1009 | 2.65527e-08 | 4.73e-09 | higher; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_prokaryotic | Acc | -0.0612 | 4.41931e-11 | 1.76638e-10 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_prokaryotic | F1 | -0.0649 | 2.54562e-07 | 2.53965e-08 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_prokaryotic | AUROC | -0.0495 | 4.41931e-11 | 1.26568e-16 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_prokaryotic | AUPRC | -0.1167 | 4.55235e-11 | 3.50303e-15 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | Precision | -0.5639 | 4.41931e-11 | 9.9129e-43 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | Recall | +0.1706 | 7.06274e-10 | 1.1512e-15 | higher; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | Acc | -0.4712 | 4.41931e-11 | 1.92517e-19 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | F1 | -0.3851 | 4.41931e-11 | 3.01645e-41 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | AUROC | -0.2523 | 4.41931e-11 | 3.27887e-21 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_eukaryotic | AUPRC | -0.5360 | 4.41931e-11 | 3.51268e-30 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | Precision | -0.4238 | 4.41931e-11 | 6.50635e-37 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | Recall | +0.4675 | 4.41931e-11 | 3.38175e-37 | higher; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | Acc | -0.6026 | 4.41931e-11 | 1.01805e-28 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | F1 | -0.2251 | 4.41931e-11 | 6.76326e-26 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | AUROC | -0.2157 | 4.41931e-11 | 1.9464e-19 | lower; MW ***, Welch *** |
| TRGH-PPI | eukaryotic_to_prokaryotic | AUPRC | -0.2827 | 4.41931e-11 | 1.393e-35 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | Precision | -0.3178 | 4.41931e-11 | 8.05878e-23 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | Recall | +0.3510 | 1.33919e-10 | 1.22867e-14 | higher; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | Acc | -0.2280 | 4.41931e-11 | 6.42262e-10 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | F1 | -0.0740 | 7.34322e-06 | 3.73315e-07 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | AUROC | -0.1090 | 4.41931e-11 | 2.63304e-13 | lower; MW ***, Welch *** |
| TRGH-PPI | prokaryotic_to_eukaryotic | AUPRC | -0.1783 | 4.91658e-11 | 1.29377e-14 | lower; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | Precision | -0.3783 | 4.41931e-11 | 9.80875e-48 | lower; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | Recall | +0.2725 | 4.41931e-11 | 2.50593e-29 | higher; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | Acc | -0.3408 | 4.41931e-11 | 2.50593e-29 | lower; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | F1 | -0.1873 | 4.41931e-11 | 2.52419e-40 | lower; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | AUROC | -0.1566 | 4.41931e-11 | 3.43738e-29 | lower; MW ***, Welch *** |
| TRGH-PPI | overall_ppi | AUPRC | -0.2784 | 4.41931e-11 | 3.01645e-41 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | Precision | -0.1836 | 7.0094e-11 | 3.61562e-13 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | Recall | -0.2276 | 4.9408e-10 | 1.9521e-14 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | Acc | -0.0659 | 4.41931e-11 | 1.24899e-18 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | F1 | -0.2308 | 4.41931e-11 | 1.22802e-27 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | AUROC | -0.1689 | 4.41931e-11 | 1.9187e-20 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_prokaryotic | AUPRC | -0.2530 | 4.41931e-11 | 1.55786e-34 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | Precision | -0.1952 | 4.41931e-11 | 1.08442e-15 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | Recall | -0.2034 | 6.48935e-11 | 1.10424e-11 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | Acc | -0.0714 | 4.41931e-11 | 3.9667e-19 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | F1 | -0.2096 | 4.41931e-11 | 7.3612e-18 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | AUROC | -0.1411 | 4.41931e-11 | 1.47042e-10 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_eukaryotic | AUPRC | -0.2936 | 4.41931e-11 | 2.38598e-18 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | Precision | -0.2521 | 4.41931e-11 | 2.64782e-25 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | Recall | -0.1619 | 0.000729511 | 5.54261e-05 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | Acc | -0.0887 | 4.41931e-11 | 5.19076e-11 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | F1 | -0.2285 | 4.41931e-11 | 5.05231e-13 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | AUROC | -0.2113 | 8.36433e-11 | 3.42331e-12 | lower; MW ***, Welch *** |
| HIGH-PPI | eukaryotic_to_prokaryotic | AUPRC | -0.1909 | 9.17135e-10 | 8.9619e-15 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | Precision | -0.2354 | 4.55235e-11 | 1.59749e-16 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | Recall | +0.1351 | 2.10768e-06 | 3.94102e-07 | higher; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | Acc | -0.1035 | 4.55235e-11 | 1.77533e-08 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | F1 | -0.0587 | 7.34322e-06 | 1.27531e-06 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | AUROC | -0.1942 | 4.41931e-11 | 4.65784e-16 | lower; MW ***, Welch *** |
| HIGH-PPI | prokaryotic_to_eukaryotic | AUPRC | -0.1696 | 2.5838e-10 | 4.10696e-12 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | Precision | -0.2166 | 4.41931e-11 | 1.14078e-25 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | Recall | -0.1144 | 3.08885e-08 | 2.09505e-09 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | Acc | -0.0824 | 4.41931e-11 | 1.59714e-17 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | F1 | -0.1819 | 4.41931e-11 | 9.5676e-27 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | AUROC | -0.1789 | 4.41931e-11 | 7.48415e-20 | lower; MW ***, Welch *** |
| HIGH-PPI | overall_ppi | AUPRC | -0.2268 | 4.41931e-11 | 3.89693e-25 | lower; MW ***, Welch *** |

The AUPRC subset used in Figure 2 is marked beside each direct-evaluation bar. Lower AUPRC error is better.
