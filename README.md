# REVEALER2

REVEALER2 (**RE**peated e**V**aluation of variabl**E**s condition**AL** **E**ntropy and **R**edundancy 2) is a method for identifying groups of genomic alterations that together associate  with a functional activation, gene dependency or drug response profile. The combination of these alterations explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. REVEALER2 can be applied to a wide variety of problems and allows prior relevant background knowledge to be incorporated into the model. Compared to original REVEALER, REVEALER2.0 can work on much larger sample size with much higher speed.

## Installation

REVEALER2 can be used in command line, jupyter notebook, and GenePattern. To use in command line or jupyter notebook, user need to install REVEALER2 with following code:

```bash
$ pip install revealer
```

Example of using REVEALER2 in jupyter notebook can be found here(link to example jupyter notebook to be added). REVEALER2 can also be found in GenePattern and directly run on GenePattern server. Details can be found here(link to genepattern module to be added).

##

REVEALER2 is separated into two parts: REVEALER_preprocess and REVEALER. If you start with MAF file or GCT file that you want to have a further filtering, then you should run REVEALER_process and then use output as input for REVEALER. If you have ready-to-use GCT format matrix, then you can directly run REVEALER. Explanation and general usage about REVEALER_preprocess and REVEALER is provided below.

## REVEALER preprocess

For the preprocessing step, there are few modes available. Detailed explanations of different mode is available in GenePattern documentation. Below are example codes for different mode. Files used in following examples can be found here(link to input files.)

Run with default mode on TCGA MAF data:

```bash
$ REVEALER_preprocess \
	-m class \
	-i tcga_pancancer_082115.vep.filter_whitelisted.maf \
	-pi HGVSp_Short
```

Run with weight compared to NFE2L2 pathway enrichment value:

```bash
$ REVEALER_preprocess \
	-m weight \
	-i tcga_pancancer_082115.vep.filter_whitelisted.maf \
	-pi HGVSp_Short \
	-p TCGA_NFE2L2 \
	-o featureFiles/ \
	-wt 0.02 \
	-pf TCGA_NFE2L2.gct \
	-pn NFE2L2.V2 \
	-nm False
```
