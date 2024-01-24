# eREVEALER

eREVEALER (**e**nhanced **RE**peated e**V**aluation of variabl**E**s condition**AL** **E**ntropy and **R**edundancy) is a method for identifying groups of genomic alterations that together associate  with a functional activation, gene dependency or drug response profile. The combination of these alterations explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. eREVEALER can be applied to a wide variety of problems and allows prior relevant background knowledge to be incorporated into the model. Compared to original REVEALER, eREVEALER can work on much larger sample size with much higher speed.

## Installation

REVEALER2 can be used in command line, jupyter notebook, and GenePattern. To use in command line or jupyter notebook, user need to install REVEALER2 with following code:

```bash
$ pip install revealer
```

Example of using eREVEALER in jupyter notebook can be found [here](example_notebook/REVEALER_Example.ipynb). REVEALER2 can also be found in GenePattern and directly run on GenePattern server. Details can be found here(link to genepattern module to be added).

##

REVEALER2 is separated into two parts: REVEALER_preprocess and REVEALER. If you start with MAF file or GCT file that you want to have a further filtering, then you should run REVEALER_process and then use output as input for REVEALER. If you have ready-to-use GCT format matrix, then you can directly run REVEALER. Explanation and general usage about REVEALER_preprocess and REVEALER are provided below.
For detailed documentation regarding each parameter and workflow, please check individual documentation for [REVEALER_preprocess](./REVEALER_Documentation.md) and [REVEALER](./REVEALER_preprocess_Documentation.md)

## REVEALER Example

For the preprocessing step, there are few modes available. Detailed explanations of different mode is available in GenePattern documentation. Below are example codes for different mode. 

Following is command line version of the example in the [here](example_notebook/REVEALER_Example.ipynb).

First, download example input file for CCLE dataset maf file from this link [https://depmap.org/portal/download/all/?releasename=DepMap+Public+23Q2&filename=OmicsSomaticMutations.csv](https://depmap.org/portal/download/all/?releasename=DepMap+Public+23Q2&filename=OmicsSomaticMutations.csv) then put in sample_input folder(or anywhere as long as you indicate in command).

## Run file preprocessing:

```bash
$ REVEALER_preprocess \
	--mode class \
	--input_file example_notebook/sample_input/OmicsSomaticMutations.csv \
	--protein_change_identifier ProteinChange \
	--file_separator , \
	--col_genename HugoSymbol \
	--col_class VariantType \
	--col_sample ModelID \
	--prefix CCLE \
	--out_folder example_notebook/sample_input/CCLE
```

### This step is to convert annotation from DepMap to CCLE.
```bash
$ python example_notebook/DepMapToCCLE.py example_notebook/sample_input/NameConvert.csv example_notebook/sample_input/CCLE_class.gct example_notebook/sample_input/CCLE_class_rename.gct
```

## Run REVEALER with generated file and NFE2L2 signature:

```bash
$ REVEALER \
	--target_file example_notebook/sample_input/CCLE_complete_sigs.gct
	--feature_file example_notebook/sample_input/CCLE_class.gct
	--out_folder example_notebook/sample_output/NRF2
	--prefix CCLE_NRF2
	--target_name NFE2L2.V2
	--if_pvalue False
	--if_bootstrap False
	--gene_locus example_notebook/sample_input/allgeneLocus.txt
	--tissue_file example_notebook/sample_input/TissueType_CCLE.gct
```
