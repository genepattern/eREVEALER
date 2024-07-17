import os
import sys
import requests
import pandas as pd

# Ensure the root of the project is in the PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from revealer.MutMaker import produce_mutation_file
from revealer.REVEALER_Cython import runREVEALER
from revealer.DepMapToCCLE import run

def download_test_file(file_url, output_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Test file downloaded successfully to: {output_path}")
    else:
        print("Failed to download the test file.")

def verify_test(generated_file, ground_truth_file):
    # Read the files
    generated_df = pd.read_csv(generated_file, sep='\t')
    ground_truth_df = pd.read_csv(ground_truth_file, sep='\t')

    # Compare the first columns
    if generated_df.iloc[:, 0].equals(ground_truth_df.iloc[:, 0]):
        print("The test finished running and it is successful!")
    else:
        print("There is problem with test, please take time investigate your package version, if you have any question, please contact author.")

def run_test(args):
    # Set the article ID and version
    article_id = "25880521"
    version = "1"

    # Retrieve the article information from the Figshare API
    article = requests.get(f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}").json()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Downloading file...")
    # Find the link for OmicsSomaticMutations.csv and download it
    for file in article["files"]:
        if file["name"] == "OmicsSomaticMutations.csv":
            download_url = file["download_url"]
            # Set the relative path for the output file
            output_directory = os.path.join(script_dir, "data", "sample_input")
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, "OmicsSomaticMutations.csv")
            # Download the file
            download_test_file(download_url, output_path)
            break

    # Perform any additional testing logic here
    print("Creating input file from mutation file...")
    produce_mutation_file(
        maf_input_file=os.path.join(script_dir, "data", "sample_input", "OmicsSomaticMutations.csv"),
        gct_output_file_prefix='CCLE',
        out_folder=os.path.join(script_dir, "data", "sample_input"),
        file_separator=',',
        protein_change_identifier='ProteinChange',
        col_genename='HugoSymbol',
        col_class='VariantType',
        col_sample='ModelID',
        mode='mutall'
    )

    run(
        os.path.join(script_dir, "data", "sample_input", "NameConvert.csv"),
        os.path.join(script_dir, "data", "sample_input", "CCLE_Mut_All.gct"),
        os.path.join(script_dir, "data", "sample_input", "CCLE_Mut_All_rename.gct")
    )

    # Input too large, please contact jim095@ucsd.edu for original file
    runREVEALER(
        target_file=os.path.join(script_dir, "data", "sample_input", "CCLE_complete_sigs.gct"),
        feature_files=[os.path.join(script_dir, "data", "sample_input", "CCLE_Mut_All_rename.gct")],
        gmt_file=os.path.join(script_dir, "data", "sample_input", "CCLE_Mut_All.gmt"),
        seed_name=['NFE2L2_Mut_All'],
        out_folder=os.path.join(script_dir, "data", "sample_output", "NRF2"),
        prefix='CCLE_NRF2',
        target_name='NFE2L2.V2',
        if_pval=False,
        if_bootstrap=False,
        if_intermediate=True,
        gene_locus=os.path.join(script_dir, "data", "sample_input", "allgeneLocus.txt"),
        tissue_file=os.path.join(script_dir, "data", "sample_input", "TissueType_CCLE.gct"),
        max_iteration = 10
    )

    # Path to the generated file and ground truth file
    generated_file = os.path.join(script_dir,  "data", "sample_output", "NRF2", "CCLE_NRF2_Result.txt")
    ground_truth_file = os.path.join(script_dir, "data", "sample_output", "CCLE_NRF2_groudtruth.txt")

    # Compare the first columns of the generated file and the ground truth file
    verify_test(generated_file, ground_truth_file)

if __name__ == "__main__":
    run_test(sys.argv)
