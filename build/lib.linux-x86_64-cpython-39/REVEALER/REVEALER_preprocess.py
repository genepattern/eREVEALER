from REVEALER.MutMaker import produce_mutation_file
import sys
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    parser._action_groups.append(optional)
    required.add_argument("-i", "--input_file", dest="input_file",
                        help="Name of input file, can be maf file or gct file.", metavar="INPUT_FILE",required=True)
    required.add_argument("-pi", "--protein_change_identifier", dest="protein_change_identifier",
                        help="protein change identifier for dataset.", metavar="PROTEIN_IDENTIFIER",required=True)

    optional.add_argument("-m", "--mode", dest="mode",
                        help="Mode, can be class, freq, weight, all, or comb. For more detail, check documentation. Default is class.", metavar="MODE")
    optional.add_argument("-r", "--ratio", dest="ratio",
                        help="Ratio of selected features by weight that is acceptable. Default is 1/3.", metavar="MODE")
    optional.add_argument("-pf", "--phenotype_file", dest="phenotype_file",
                        help="Name of phenotype file. Required if mode is one of freq, weight, all, or comb.", metavar="PHENOTYPE_FILE")
    optional.add_argument("-c", "--class_file", dest="class_file",
                        help="File for classifiers, required if input file is gct file.", metavar="CLASS_FILE")
    optional.add_argument("-p", "--prefix", dest="prefix",
                        help="prefix for output files. default is Mut.", metavar="PREFIX")
    optional.add_argument("-o", "--out_folder", dest="out_folder",
                        help="Path to directory to put output files. default is current directory.", metavar="OUT_FOLDER")
    optional.add_argument("-cs", "--class_separator", dest="class_separator",
                        help="Separator between gene name and class name. default is '_'", metavar="CLASS_SEPARATOR")
    optional.add_argument("-pn", "--phenotype_name", dest="phenotype_name",
                        help="Name of phenotypr to be used in phenotype file. Default is first row.", metavar="PHENOTYPE_NAME")
    optional.add_argument("-fs", "--file_separator", dest="file_separator",
                        help="Separator of file, default is tab('\\t')", metavar="CLASS_SEPARATOR")
    optional.add_argument("-d", "--direction", dest="direction",
                        help="direction of phenotype the features should match. Default is positive.", metavar="DIRECTION")
    optional.add_argument("-ft", "--frequency_threshold", dest="frequency_threshold",
                        help="Threshold for frequency. Default is 5.", metavar="FREQUENCY_THRESHOLD")
    optional.add_argument("-wt", "--weight_threshold", dest="weight_threshold",
                        help="Threshold for weight. Default is 0.7.", metavar="WEIGHT_THRESHOLD")
    optional.add_argument("-gl", "--gene_list", dest="gene_list",
                        help="List of Gene to run. Can be file or ',' separated names. Default is all gene.", metavar="GENE_LIST")
    optional.add_argument("-nm", "--name_match", dest="name_match",
                        help="Indicate if sample name in phenotype file and input file are matching. Default is True.", metavar="NAME_MATCH")
    optional.add_argument("-mf", "--make_figure", dest="make_figure",
                        help="Indicate if figures are created. Recommended only when you run few genes. Default is False.", metavar="MAKE_FIG")
    optional.add_argument("-ff", "--figure_format", dest="figure_format",
                        help="Format for figure. Default is pdf.", metavar="FIGURE_FORMAT")
    optional.add_argument("-sl", "--sample_list", dest="sample_list",
                        help="List of sample to be used.", metavar="SAMPLE_LIST")
    optional.add_argument("-tr", "--total_ratio", dest="total_ratio",
                        help="Ratio of gene occrence compared to total sample number. default is 0.4", metavar="TOTAL_RATIO")
    optional.add_argument("-ig", "--if_gmt", dest="if_gmt",
                        help="Indicate if gmt file should be generated. Default if True.", metavar="IF_GMT")
    optional.add_argument("-v", "--verbose", dest="verbose",
                        help="Indicate level of verbose. default is 1.", metavar="VERBOSE")
    optional.add_argument("-k", "--k_size", dest="k",
                        help="Size of kernel indicating k variance far from middle. Has to be int. Default is 5", metavar="K")
    optional.add_argument("-bm", "--bandwidth_multiplication", dest="bandwidth_mult",
                        help="Value of bandwidth multiplier. Has to be float or int. Default is 0.65.", metavar="BANDWIDTH_MULT")
    optional.add_argument("-ba", "--bandwidth_adjustment", dest="bandwidth_adj",
                        help="Value of bandwidth adjustment. Has to be float or int. Default is -0.95", metavar="BANDWIDTH_ADJ")
    optional.add_argument("-gz", "--gzip", dest="gzip",
                        help="If output file should be gzipped.", metavar="GZIP") 
    optional.add_argument("-cb", "--combine", dest="combine",
                        help="Combine alleles by gene when making figures", metavar="COMBINE") 

    args = parser.parse_args()
    args = vars(args)

    if args['input_file'][-3:] != 'gct' and args['input_file'][-3:] != 'maf':
        print('only maf file and gct file are accepted as input, please check your input file.')
        sys.exit(1)
    else:
        input_file = args['input_file']

    if args['mode'] == None:
        mode = 'class'
    elif (args['mode'] == 'mutall' or args['mode'] == 'allele' or args['mode'] == 'class' or args['mode'] == 'freq' or args['mode'] == 'weight' or args['mode'] == 'weight_filter' or args['mode'] == 'comb') == False:
        print('Mode not available! Please check instruction for available mode.')
        sys.exit(1)
    else:
        mode = args['mode']

    if (args['mode'] == 'freq' or args['mode'] == 'weight' or args['mode'] == 'weight_filter' or args['mode'] == 'comb') and args['phenotype_file'] == None:
        print('Please indicate phenotype file if use mode '+ args['mode'])
        sys.exit(1)

    if args['input_file'][-3:] == 'gct' and args['class_file'] == None:
        print('Please indicate class names by passing class file.')
        sys.exit(1)

    if args['input_file'][-3:] == 'gct':
        with open(args['class_file'],'r') as f:
            class_list = [line.rstrip() for line in f]
    else:
        class_list = ['Nonsense_Mutation','In_Frame_Del','Silent','Frame_Shift_Ins','Missense_Mutation','Splice_Site',
                  'Frame_Shift_Del','De_novo_Start_OutOfFrame','Nonstop_Mutation','In_Frame_Ins','Start_Codon_SNP',
                  'Start_Codon_Del','Stop_Codon_Ins','Start_Codon_Ins','Stop_Codon_Del','Intron','IGR',"5'Flank",
                  "3'UTR","5'UTR",'Mut_All']

    if args['prefix'] != None:
        prefix = args['prefix']
    else:
        prefix = 'Mut'

    if args['ratio'] != None:
        try:
            if '/' in args['ratio']:
                ratio = float(args['ratio'].split('/')[0])/float(args['ratio'].split('/')[1])
            else:
                ratio = float(args['ratio'])
            print(ratio)
            if ratio > 1:
                print('Ratio has to be value less than 1')
                sys.exit(1)
        except ValueError:
            print('Ratio has to be value less than 1')
            sys.exit(1)
    else:
        ratio = float(1/3)

    if args['out_folder'] != None:
        out_folder = args['out_folder']
    else:
        out_folder = './'

    if args['class_separator'] != None:
        class_separator = args['class_separator']
    else:
        class_separator = '_'

    if args['phenotype_name'] != None:
        try:
            phenotype_name = int(args['phenotype_name'])
        except ValueError:
            phenotype_name = args['phenotype_name']
    else:
        phenotype_name = 0

    if args['file_separator'] != None:
        file_separator = args['file_separator']
    else:
        file_separator = '\t'

    if args['direction'] != None:
        direction = args['direction']
    else:
        direction = 'pos'

    if args['frequency_threshold'] != None:
        frequency_threshold = int(args['frequency_threshold'])
    else:
        frequency_threshold = 5

    if args['weight_threshold'] != None:
        weight_threshold = float(args['weight_threshold'])
    else:
        weight_threshold = 0.7

    if args['total_ratio'] != None:
        total_ratio = float(args['total_ratio'])
    else:
        total_ratio = 0.4


    if args['gene_list'] != None:
        if args['gene_list'][-3:] != 'txt':
            gene_list =  args['gene_list'].split(',')
        else:
            with open(args['gene_list'],'r') as f:
                gene_list = [line.rstrip() for line in f]
    else:
        gene_list = None

    if args['sample_list'] != None:
        with open(args['sample_list'],'r') as f:
            sample_list = [line.rstrip() for line in f]
    else:
        sample_list = None

    if args['name_match'] == None:
        name_match = True
    elif args['name_match'] == 'True':
        name_match = True
    elif args['name_match'] == 'False':
        name_match = False
    else:
        print('Only True or False accepted for --name_match')
        sys.exit(1)

    if args['make_figure'] == None:
        make_figure = False
    elif args['make_figure'] == 'True':
        make_figure = True
    elif args['make_figure'] == 'False':
        make_figure = False
    else:
        print('Only True or False accepted for --make_figure')
        sys.exit(1)
    
    if args['figure_format'] != None:
        figure_format = args['figure_format']
    else:
        figure_format = 'pdf'

    if args['if_gmt'] == None:
        if_gmt = True
    elif args['if_gmt'] == 'True':
        if_gmt = True
    elif args['if_gmt'] == 'False':
        if_gmt = False
    else:
        print('Only True or False accepted for --if_gmt')
        sys.exit(1)

    if args['k'] != None:
        try:
            k = int(args['k'])
        except ValueError:
            print('K has to be int')
            sys.exit(1)
    else:
        k = 30

    if args['bandwidth_mult'] != None:
        try:
            bandwidth_mult = float(args['bandwidth_mult'])
        except ValueError:
            print('Bnadwidth multiplier has to be int or float')
            sys.exit(1)
    else:
        bandwidth_mult = 0.65

    if args['bandwidth_adj'] != None:
        try:
            bandwidth_adj = float(args['bandwidth_adj'])
        except ValueError:
            print('Bnadwidth adjustment has to be int or float')
            sys.exit(1)
    else:
        bandwidth_adj = -0.95

    if args['verbose'] != None:
        try:
            verbose = int(args['verbose'])
        except ValueError:
            verbose = args['verbose']
    else:
        verbose = 1

    if args['gzip'] == None:
        gzip = True
    elif args['gzip'] == 'True':
        gzip = True
    elif args['gzip'] == 'False':
        gzip = False
    else:
        print('Only True or False accepted for gzip')
        sys.exit(1)

    if args['combine'] == None:
        combine = True
    elif args['combine'] == 'True':
        combine = True
    elif args['combine'] == 'False':
        combine = False
    else:
        print('Only True or False accepted for --combine')
        sys.exit(1)

    if verbose > 0:
        print("""
Parameters utilized:
input_file="""+input_file+"""
protein_change_identifier="""+args['protein_change_identifier']+"""
mode="""+mode+"""
class_list="""+str(class_list)+"""
prefix="""+prefix+"""
ratio="""+str(ratio)+"""
out_folder="""+str(out_folder)+"""
class_separator="""+str(class_separator)+"""
phenotype_name="""+str(phenotype_name)+"""
file_separator="""+str(file_separator)+"""
direction="""+direction+"""
frequency_threshold="""+str(frequency_threshold)+"""
weight_threshold="""+str(weight_threshold)+"""
total_ratio="""+str(total_ratio)+"""
gene_list="""+str(gene_list)+"""
sample_list="""+str(sample_list)+"""
name_match="""+str(name_match)+"""
make_figure="""+str(make_figure)+"""
figure_format="""+str(figure_format)+"""
if_gmt="""+str(if_gmt)+"""
verbose="""+str(verbose)+"""
k="""+str(k)+"""
bandwidth_mult="""+str(bandwidth_mult)+"""
bandwidth_adj="""+str(bandwidth_adj)+"""
gzip="""+str(gzip)+"""
combine="""+str(combine))


    if input_file[-3:] == 'gct':
        produce_mutation_file(
            maf_input_file = None,
            gct_input_file = input_file,
            gct_output_file_prefix = prefix,
            phenotype_file = args['phenotype_file'],
            class_list = class_list,
            class_seperator = class_separator,
            phenotype_name = phenotype_name,
            file_separator = file_separator,
            protein_change_identifier = args['protein_change_identifier'],
            mode = mode,
            direction = direction,
            frequency_threshold = frequency_threshold,
            weight_threshold = weight_threshold,
            gene_list = gene_list,
            name_match = name_match,
            make_figure = make_figure,
            figure_format = figure_format,
            out_folder = out_folder,
            ratio = ratio,
            sample_list = sample_list,
            total_ratio = total_ratio,
            if_gmt = if_gmt,
            k = k,
            gzip=gzip,
            combine = combine)

    elif input_file[-3:] == 'maf':
        produce_mutation_file(
            maf_input_file = input_file,
            gct_input_file = None,
            gct_output_file_prefix = prefix,
            phenotype_file = args['phenotype_file'],
            class_list = class_list,
            class_seperator = class_separator,
            phenotype_name = phenotype_name,
            file_separator = file_separator,
            protein_change_identifier = args['protein_change_identifier'],
            mode = mode,
            direction = direction,
            frequency_threshold = frequency_threshold,
            weight_threshold = weight_threshold,
            gene_list = gene_list,
            name_match = name_match,
            make_figure = make_figure,
            figure_format = figure_format,
            out_folder = out_folder,
            ratio = ratio,
            sample_list = sample_list,
            total_ratio = total_ratio,
            if_gmt = if_gmt,
            k = k,
            gzip = gzip,
            combine = combine)