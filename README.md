
# eREVEALER

**eREVEALER** (**e**nhanced **RE**peated e**V**aluation of variabl**E**s condition**AL** **E**ntropy and **R**edundancy) is a powerful method for identifying groups of genomic alterations that, together, associate with functional activation, gene dependency, or drug response profiles. By combining these alterations, eREVEALER explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. eREVEALER extends the capabilities of the original REVEALER by handling larger sample sizes with significantly higher speed.

Preprint is avaiable [here](https://www.biorxiv.org/content/10.1101/2023.11.14.567106v1)

![Alt text](docs/images/REVEALER_schematic.png)

## Overview

eREVEALER consists of two main components: `REVEALER preprocess` and `REVEALER run`. 

- **REVEALER preprocess**: If you start with a MAF file or a GCT file that needs further filtering, run `REVEALER preprocess` first and use its output as the input for `REVEALER run`.
- **REVEALER run**: If you have a ready-to-use GCT format matrix, you can directly run `REVEALER run`.

For detailed documentation regarding each parameter and workflow, refer to the individual documentation for [REVEALER_preprocess](docs/REVEALER_preprocess_Documentation.md) and [REVEALER](docs/REVEALER_Documentation.md).


## Installation

### Python version prerequisite

Please use Python version >= 3.7 and < 3.10

### Create Conda environment

```bash
conda create -n revealer python==3.9
```

### Install via pip

eREVEALER can be used in the command line, Jupyter Notebook, and GenePattern. To use eREVEALER in the command line or Jupyter Notebook, install it via pip:

```bash
pip install revealer
```

### Install via cloning the repository

Alternatively, you can install eREVEALER by cloning the repository and running the setup script.

1. **Clone the repository**:

    ```bash
    git clone git@github.com:yoshihiko1218/eREVEALER.git
    cd REVEALER
    ```

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Install the package**:

    ```bash
    python setup.py install
    ```

## Testing installation with an example
After you finish installing, you can test REVEALER by running 

```bash
REVEALER test 
```
This will take approximately an hour.

## Jupyter notebook Usage

Detailed example of using eREVEALER in Jupyter Notebook can be found [here](example/REVEALER_Example.ipynb). eREVEALER is also available in GenePattern, allowing you to run it directly on the GenePattern server. More details can be found [here](link to genepattern module to be added).

## Command line Usage

The preprocessing step offers various modes, which are explained in detail in the GenePattern documentation. Below are example commands for different modes. 

Here is the command-line version of the example found [here](example/REVEALER_Example.ipynb).

### Download Example Input File

First, download the example input file for the CCLE dataset MAF file from this link: [DepMap Public 23Q2 OmicsSomaticMutations.csv](https://depmap.org/portal/download/all/?releasename=DepMap+Public+23Q2&filename=OmicsSomaticMutations.csv). Save it to the `sample_input` folder (or another location, as long as you indicate the path in the command).

### Run File Preprocessing

```bash
REVEALER preprocess \
    --mode class \
    --input_file example/sample_input/OmicsSomaticMutations.csv \
    --protein_change_identifier ProteinChange \
    --file_separator , \
    --col_genename HugoSymbol \
    --col_class VariantType \
    --col_sample ModelID \
    --prefix CCLE \
    --out_folder example/sample_input/CCLE \
    --mode mutall
```

### Convert Annotation from DepMap to CCLE

```bash
python example_notebook/DepMapToCCLE.py example/sample_input/NameConvert.csv example/sample_input/CCLE_Mut_All.gct example/sample_input/CCLE_Mut_All_rename.gct
```

### Run REVEALER with Generated File and NFE2L2 Signature

```bash
REVEALER run \
    --target_file example_notebook/sample_input/CCLE_complete_sigs.gct \
    --feature_file example_notebook/sample_input/CCLE_Mut_All_rename.gct \
    --out_folder example_notebook/sample_output/NRF2 \
    --prefix CCLE_NRF2 \
    --target_name NFE2L2.V2 \
    --if_pvalue False \
    --if_bootstrap False \
    --gene_locus example_notebook/sample_input/allgeneLocus.txt \
    --tissue_file example_notebook/sample_input/TissueType_CCLE.gct
```

## Contributing

If you would like to contribute to eREVEALER, please submit a pull request or report issues on our [GitHub repository](https://github.com/yoshihiko1218/eREVEALER).

## License

eREVEALER is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
