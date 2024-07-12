# Description
Preprocessing step of REVEALER2. Mutation information of samples can be combined into a single matrix of features by sample based on the user's requirement. Also, the matrix can be passed in, and various filters can be applied.

## Summary
REVEALER2 is a powerful tool to investigate mutually exclusive binary features that are correlated with a continuous target. This preprocessing step can take a raw mutation file and prepare a ready-to-use input file for REVEALER2. The features can be customly generated and filtered based on their weight, frequency, or classification.

## Introduction
REVEALER2 (repeated evaluation of variables conditional entropy and redundancy 2) is a method for identifying groups of genomic alterations that together associate with a functional activation, gene dependency, or drug response profile. REVEALER2 takes a matrix of binary features by samples as input. This tool can prepare an input matrix for REVEALER2 from a raw MAF format mutation file. Users can provide their own filtering method to generate different types of input matrices. The input matrix can be generated based on mutation classification, weight compared to phenotype, or frequency of mutation.

## Algorithm
### Allele Mode
In this mode, each feature is [Gene Name]_[Allele Name]. No combination is performed on mutations. Filtering will be applied based on the gene list provided by the user or frequency threshold provided by the user or in default. If the user is interested not only in mutually exclusive genes but also in allele level, this mode is recommended. The output will be larger and more sparse compared to other modes, so specifying a gene would be recommended.

### Mutall Mode
In this mode, each feature is [Gene Name]_Mut_All. Basically, all gene-allele pairs of each gene are combined into a single feature. Filtering can be applied based on the gene list provided by the user or frequency threshold provided by the user or in default.

### Class Mode
In this mode, each feature is [Gene Name]_[Classification Name]. For each gene-class combination feature, samples with that mutation will be set as 1. [Gene Name]_Mut_All feature in the above mode is also generated in this mode. Filtering will be applied based on the gene list provided by the user or frequency threshold provided by the user or in default. This is the default mode and recommended for general investigation.

### Weight Mode
In this mode, the user can provide a target, which will be the same as the target in the later REVEALER2 main part, to filter mutations. Each feature will be named as [Gene Name]_weight_[Threshold Provided]. First, the target is normalized and raised based on the user-provided value. Then, IC between the target and each gene-allele combination is calculated (detailed calculation method can be checked in REVEALER documentation). For each gene, all gene-allele combinations that pass the weight threshold will be combined into a single feature. Filtering can be further applied based on the gene list provided by the user or frequency threshold provided by the user or in default.

### Weight With Filter Mode
This mode is mostly the same as the mode above with one extra filtering step. After combining gene-allele combinations after filtering with a weight threshold, the number of positive samples is counted, and if it is less than the number of positive genes for this gene times the given ratio, then this gene is filtered out. This filter is added to filter out genes with most alleles meaningless in this situation.

### Frequency Mode
In this mode, the feature is named as [Gene Name]_frequency_[Top N num provided by the user]. The user needs to provide a target, which is used to break the tie. First, the target is normalized and raised based on the user-provided value. Then, IC between the target and each gene-allele combination is calculated (detailed calculation method can be checked in REVEALER documentation). For each gene, all gene-allele combinations are sorted based on their frequency and IC, with frequency prioritized, and then the top n gene-allele combination is combined as a single feature. Filtering can be further applied based on the gene list provided by the user or frequency threshold provided by the user or in default. This mode is not recommended in general, mainly because with tests using real data, we realized that this mode cannot reflect biological insights into the data, which leads to less meaningful results in the later step.

For all modes, GMT files can be generated indicating detailed allele in each feature, which can be used as input for later REVEALER.

## Parameters

| Name                    | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| input_file              | This is a file in MAF or GCT format that contains mutation information.                              |
| protein_change_identifier | Name of Protein Change identifier. This is the most detailed classification for mutations.        |
| mode                    | This is the mode preprocessing should run with. Available options are: class, freq, weight, weight_filter, mutall, or comb. |
| prefix                  | Prefix of result files.                                                                              |
| phenotype_file          | This is a file in GCT format that contains the target to run weight, weight_filter, or freq mode.  |
| phenotype_name          | Name of the target feature in the phenotype file.                                                   |
| weight_threshold        | Weight threshold to filter by IC.                                                                    |
| class_file              | Class file that includes a list of classes. This is required if a GCT file is provided as input.   |
| direction               | This indicates the correlation direction between features and the target expected to be observed.    |
| frequency_threshold     | This indicates the top frequency threshold number of gene-allele combinations that should be picked for each feature. |
| name_match              | This is a boolean parameter that indicates if the sample of the target and the name in the MAF file is matching. For TCGA example, because the MAF file has a longer annotation for samples, then this parameter should be set to False. |
| gene_list               | This indicates the multiplication value in bandwidth calculation.                                     |
| sample_list             | This is a text file indicating a subset of samples to be utilized. This can be used if only one type of cancer is investigated. |
| ratio                   | Ratio of the number of samples in selected gene-allele combined features over the number of samples in that gene that is acceptable. Used in weight_filter mode. |
| make_figure             | This is a boolean parameter that indicates if a heatmap is needed to be generated for each gene indicating the distribution of each allele in that gene. Recommended only when running on a few genes, otherwise, too many figures are generated. |
| total_ratio             | This ratio is used to filter out too frequent genes that the number of positive samples compared to the total number of samples is more than this value. |
| class_separator         | Separator between gene name and later part if a GCT file is provided.                                 |
| if_gmt                  | This is a boolean option that indicates if a GMT file is generated as output.                        |
| k                       | This indicates the number of neighborhoods for the kernel. A higher number will lead to higher accuracy but lower speed. |
| bandwidth_multiplication | This indicates multiplication value in bandwidth calculation.                                          |
| bandwidth_adjustment    | This indicates adjustment value in bandwidth calculation.                                             |
| gzip                    | This is a boolean option that indicates if result files are gzipped.                                    |
| output_folder           | This indicates the name of the output folder.                                                         |

## Input Files
1. **Annotated mutation**: MAF file
   - This file contains mutation information. Can be generated from VCF files.

2. **Optional target phenotype**: GCT file
   - This file contains a continuous target to calculate IC in weight, weight_filter, and freq mode.

3. **Optional sample list**: txt file
   - This file contains a list of samples to use for analysis. If the user wants to run REVEALER on only a subset of samples (for example, samples in only a specific cancer type), this can be used.

4. **Optional gene list**: txt file
   - This file contains a list of genes to use for analysis. If the user wants to run REVEALER on only a subset of genes (for example, genes with known function), this can be used.

## Output Files
1. **Feature matrix**: GCT file
   - The main result that is a matrix in GCT format, with rows as feature names and columns as sample names. This can be directly used as REVEALER input.

2. **Optional gene-level heatmap**: PDF or PNG image
   - Heatmap showing allele distribution compared to target for each gene.

3. **Optional allele detail for each final feature**: GMT file
   - Allele information is provided for each feature in the result GCT file matrix.
