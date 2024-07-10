import sys
import pandas as pd

def main():
    if len(sys.argv) != 4:
        print("Usage: python DepMapToCCLE.py <NameConvert.csv> <CCLE_class.gct> <output_filename.gct>")
        sys.exit(1)

    # Read the file paths from command line arguments
    name_convert_file, feature_file, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

    # Read feature file
    feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
    feature = feature.drop(columns=feature.columns[0])

    # Read the dat1a
    converter = pd.read_csv(name_convert_file)

    # Create a mapping dictionary from 'ModelID' to 'CCLEName'
    model_to_ccle = dict(zip(converter['ModelID'], converter['CCLEName']))

    # Rename the columns in ccle_class_df using the mapping
    feature.rename(columns=model_to_ccle, inplace=True)

    # Save the result
    with open(outfile, mode = "w") as output_file:
        output_file.writelines("#1.2\n{}\t{}\n".format(feature.shape[0], feature.shape[1] - 1))
        feature.to_csv(output_file, sep= '\t')

if __name__ == "__main__":
    main()

def run(name_convert_file, feature_file, outfile):
    # Read feature file
    feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
    feature = feature.drop(columns=feature.columns[0])

    # Read the dat1a
    converter = pd.read_csv(name_convert_file)

    # Create a mapping dictionary from 'ModelID' to 'CCLEName'
    model_to_ccle = dict(zip(converter['ModelID'], converter['CCLEName']))

    # Rename the columns in ccle_class_df using the mapping
    feature.rename(columns=model_to_ccle, inplace=True)

    # Save the result
    with open(outfile, mode = "w") as output_file:
        output_file.writelines("#1.2\n{}\t{}\n".format(feature.shape[0], feature.shape[1] - 1))
        feature.to_csv(output_file, sep= '\t')
