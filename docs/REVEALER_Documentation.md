# Description
Get a list of mutually exclusive binary features (e.g., mutation of gene loci) that are strongly correlated with a continuous target (e.g., geneset enrichment score, gene expression).

## Summary
REVEALER2.0 is a powerful tool to investigate mutually exclusive binary features that are correlated with a continuous target. It loops through the provided features and for each iteration, choose a feature that has the highest conditional information coefficient value based on the provided target and seed in each iteration.

## Introduction
REVEALER2 (repeated evaluation of variables conditional entropy and redundancy 2) is a method for identifying groups of genomic alterations that together associate with a functional activation, gene dependency, or drug response profile. The combination of these alterations explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. REVEALER2 can be applied to a wide variety of problems and allows prior relevant background knowledge to be incorporated into the model. Compared to the original REVEALER, REVEALER2.0 can work on a much larger sample size with much higher speed.

## Algorithm
*PUT PSEUDOCODE HERE*

## Parameters

| Name                    | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| target_file             | This is a file in gct format that contains continuous target data.                                  |
| target_name             | Name of the target in the target file. Take the first row by default.                                |
| feature_file            | This is a file in gct format that contains all the binary features. Should be feature in a row and sample name in a column. |
| prefix                  | Prefix of result files.                                                                              |
| seed_file               | This is a file in gct format that contains seed to start with.                                       |
| seed_name               | Name of seeds to be used. Multiple seed names can be provided and will be collapsed in the program.  |
| gene_locus              | Text file that contains the information of the locus of each gene. This file can be downloaded from the GitHub page. |
| tissue_file             | This is a file in gct format that contains tissue type information of samples.                      |
| gmt_file                | This is a file in gmt format that contains allele information of each feature. This file can be generated in REVEALER_preprocess. |
| minimum_iteration       | This indicates how many iterations REVEALER should run. If set to -1, the iteration will stop automatically when the total IC starts decreasing. |
| k                       | This indicates the number of neighborhoods for the kernel. A higher number will lead to higher accuracy but lower speed. |
| bandwidth_multiplication | This indicates the multiplication value in bandwidth calculation.                                      |
| bandwidth_adjustment    | This indicates the adjustment value in bandwidth calculation.                                         |
| direction               | This indicates the correlation direction between features and the target expected to be observed.    |
| num_top                 | This indicates the number of top-scored features to be plotted in intermediate figures.              |
| low_threshold           | This indicates the lower threshold of feature filtering. If an integer is provided, then features that have fewer than that number of sample mutations are removed. If a float is provided, then features that have less than that ratio of sample mutation among all samples are removed. |
| high_threshold          | This indicates the higher threshold of feature filtering. If an integer is provided, then features that have more than that number of sample mutations are removed. If a float is provided, then features that have more than that ratio of sample mutation among all samples are removed. |
| if_collapse             | This is a boolean option that indicates if exactly the same features are collapsed together in the intermediate figure. |
| subset                  | This is a text file indicating a subset of samples to be utilized. This can be used if only one type of cancer is investigated. |
| figure_format           | Format of generated figures.                                                                         |
| if_pvalue               | This is a boolean option that indicates if p-values are calculated.                                  |
| if_bootstrap            | This is a boolean option that indicates if variants are calculated.                                   |
| if_cluster              | This is a boolean option that indicates if features are clustered with NMF for intermediate figures. |
| if_intermediate         | This is a boolean option that indicates if intermediate figures are generated.                       |
| separator               | This indicates the character that is used to separate gene name and the later part in input files.   |
| gene_set                | This is a file that includes genes that are used. This is used when only a subset of genes is used for analysis. |
| alpha                   | This is the value that is used to raise target values by a specific time.                              |
| gzip                    | This is a boolean option that indicates if result files are gzipped.                                  |
| output_folder           | This indicates the name of the output folder.                                                         |

## Input Files
1. **Target dataset**: GCT file
   - This file contains continuous target phenotype to correlate with features.

2. **Feature dataset**: GCT file
   - This file contains all binary features that are used to compare.

3. **Optional seed database**: GCT file
   - This file contains the seed that is used for the first iteration. If both seed file and seed name are not provided, then it will start without a seed. If a seed name is provided but no seed file is provided, then it will find seeds from the feature file.

4. **Optional gene locus**: txt file
   - This file contains locus information of genes. This can be downloaded from GitHub.

5. **Optional allele information for each feature**: gmt file
   - This file contains what mutations are included in each feature. This file can be generated by REVEALER_preprocess.

6. **Optional sample list**: txt file
   - This file contains a list of samples to use for analysis. If the user wants to run REVEALER on only a subset of samples (for example, samples in only a specific cancer type), this can be used.

7. **Optional gene list**: txt file
   - This file contains a list of genes to use for analysis. If the user wants to run REVEALER on only a subset of genes (for example, genes with known function), this can be used.

## Output Files
1. **Main Result Heatmap**: PDF or PNG image
   - The main result heatmap. The first row is the target distribution, and the following rows are the top features picked in each iteration. Feature names are labeled on the left side, and optional variations and p-values will be labeled on the right side.

2. **Optional intermediate heatmap**: PDF or PNG image
   - The intermediate heatmap that is generated one for each iteration. The first row is the target distribution, and the following rows are the top n features that are picked in this iteration. Feature names are labeled on the left side, and optional variations and p-values will be labeled on the right side.

3. **Optional allele detail for each final feature**: gmt file
   - Allele information is provided for each feature that is selected for the result. This file is generated only when the user provides a gmt file of allele information of all features as an input file.

4. **Optional tissue distribution image**: PDF or PNG
   - If tissue distribution of samples is provided, then tissue distribution result compared to
