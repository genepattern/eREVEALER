
# eREVEALER

**eREVEALER** (**e**nhanced **RE**peated e**V**aluation of variabl**E**s condition**AL** **E**ntropy and **R**edundancy) is a powerful method for identifying groups of genomic alterations that, together, associate with functional activation, gene dependency, or drug response profiles. By combining these alterations, eREVEALER explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. eREVEALER extends the capabilities of the original REVEALER by handling larger sample sizes with significantly higher speed.

## Installation

### Install via pip

eREVEALER can be used in command line, Jupyter Notebook, and GenePattern. To use eREVEALER in command line or Jupyter Notebook, install it via pip:

\`\`\`bash
pip install revealer
\`\`\`

### Install via cloning the repository

Alternatively, you can install eREVEALER by cloning the repository and running the setup script.

1. **Clone the repository**:

    \`\`\`bash
    git clone https://github.com/yoshihiko1218/REVEALER.git
    cd REVEALER
    \`\`\`

2. **Install the dependencies**:

    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

3. **Install the package**:

    \`\`\`bash
    python setup.py install
    \`\`\`

## Usage

An example of using eREVEALER in a Jupyter Notebook can be found [here](example_notebook/REVEALER_Example.ipynb). eREVEALER is also available in GenePattern, allowing you to run it directly on the GenePattern server. More details can be found [here](link to genepattern module to be added).

## Overview

eREVEALER consists of two main components: `REVEALER_preprocess` and `REVEALER`. 

- **REVEALER_preprocess**: If you start with a MAF file or a GCT file that needs further filtering, run `REVEALER_preprocess` first and use its output as the input for `REVEALER`.
- **REVEALER**: If you have a ready-to-use GCT format matrix, you can directly run `REVEALER`.

For detailed documentation regarding each parameter and workflow, refer to the individual documentation for [REVEALER_preprocess](./REVEALER_preprocess_Documentation.md) and [REVEALER](./REVEALER_Documentation.md).

## REVEALER Example

The preprocessing step offers various modes, which are explained in detail in the GenePattern documentation. Below are example commands for different modes. 

Here is the command-line version of the example found [here](example_notebook/REVEALER_Example.ipynb).

### Download Example Input File

First, download the example input file for the CCLE dataset MAF file from this link: [DepMap Public 23Q2 OmicsSomaticMutations.csv](https://depmap.org/portal/download/all/?releasename=DepMap+Public+23Q2&filename=OmicsSomaticMutations.csv). Save it to the `sample_input` folder (or another location, as long as you indicate the path in the command).

### Run File Preprocessing

\`\`\`bash
REVEALER_preprocess \
    --mode class \
    --input_file example_notebook/sample_input/OmicsSomaticMutations.csv \
    --protein_change_identifier ProteinChange \
    --file_separator , \
    --col_genename HugoSymbol \
    --col_class VariantType \
    --col_sample ModelID \
    --prefix CCLE \
    --out_folder example_notebook/sample_input/CCLE
\`\`\`

### Convert Annotation from DepMap to CCLE

\`\`\`bash
python example_notebook/DepMapToCCLE.py example_notebook/sample_input/NameConvert.csv example_notebook/sample_input/CCLE_class.gct example_notebook/sample_input/CCLE_class_rename.gct
\`\`\`

### Run REVEALER with Generated File and NFE2L2 Signature

\`\`\`bash
REVEALER \
    --target_file example_notebook/sample_input/CCLE_complete_sigs.gct \
    --feature_file example_notebook/sample_input/CCLE_class.gct \
    --out_folder example_notebook/sample_output/NRF2 \
    --prefix CCLE_NRF2 \
    --target_name NFE2L2.V2 \
    --if_pvalue False \
    --if_bootstrap False \
    --gene_locus example_notebook/sample_input/allgeneLocus.txt \
    --tissue_file example_notebook/sample_input/TissueType_CCLE.gct
\`\`\`

## Contributing

If you would like to contribute to eREVEALER, please submit a pull request or report issues on our [GitHub repository](https://github.com/yoshihiko1218/REVEALER).

## License

eREVEALER is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
