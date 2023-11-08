from REVEALER.REVEALER_Cython import runREVEALER
#from REVEALER.CheckGrid import runCheckGrid
from REVEALER.REVEALER_runbenchmark import runBenchmark
import sys
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    parser._action_groups.append(optional)
    required.add_argument("-tf", "--target_file", dest="target_file",
                        help="Name of target file, has to be gct file.", metavar="TARGET_FILE",required=True)
    required.add_argument("-f", "--feature_file", dest="feature_file",
                        help="Name of feature file, has to be gct file.", metavar="FEATURE_FILE",required=True)

    optional.add_argument("-sf", "--seed_file", dest="seed_file",
                        help="Name of seed file.", metavar="SEED_FILE")
    optional.add_argument("-p", "--prefix", dest="prefix",
                        help="Prefix for results files. Default is REVEALER.", metavar="PREFIX")
    optional.add_argument("-sn", "--seed_name", dest="seed_name",
                        help="Seed names to take from seed file or feature file. If multiple exist, separate by ','. Passing text file(ending with 'txt') with column of seed names also works", 
                        metavar="CLASS_FILE")
    optional.add_argument("-g", "--grid", dest="grid",
                        help="Grid size to put kernel on. has to be int. Default is 34.", metavar="GRID")
    optional.add_argument("-t", "--target_name", dest="target_name",
                        help="Name of target in target file. Can be row name or row index(0 based). Default is first row.", metavar="TARGET_NAME")
    optional.add_argument("-k", "--k_size", dest="k",
                        help="Size of kernel indicating k variance far from middle. Has to be int. Default is 5", metavar="K")
    optional.add_argument("-bm", "--bandwidth_multiplication", dest="bandwidth_mult",
                        help="Value of bandwidth multiplier. Has to be float or int. Default is 0.65.", metavar="BANDWIDTH_MULT")
    optional.add_argument("-ba", "--bandwidth_adjustment", dest="bandwidth_adj",
                        help="Value of bandwidth adjustment. Has to be float or int. Default is -0.95", metavar="BANDWIDTH_ADJ")
    optional.add_argument("-d", "--direction", dest="direction",
                        help="Direction of phenotype the features should match. Default is positive.", metavar="DIRECTION")
    optional.add_argument("-m", "--mode", dest="mode",
                        help="Mode to run REVEALER. Can be single or multiple. If set to multiple, format as sample input is required. Default is single.", metavar="MODE")
    optional.add_argument("-ps", "--parameter_set", dest="parameter_set",
                        help="set of parameter. Required if mode is multiple. Check avaiable format in documentation.", metavar="MODE")
    optional.add_argument("-nt", "--num_top", dest="num_top",
                        help="Number of top features picked for intermediate report. Has to be int. Default is 30.", metavar="NUM_TOP")
    optional.add_argument("-lt", "--low_threshold", dest="low_threshold",
                        help="Lower threshold to remove feature. If int passed, threshold of absolute value is made. If float passsed, threshold made by ratio. Default is 3.", metavar="LOW_THRESHOLD")
    optional.add_argument("-ht", "--high_threshold", dest="high_threshold",
                        help="Higher threshold to remove feature. If int passed, threshold of absolute value is made. If float passsed, threshold made by ratio. Default is 0.2.", metavar="HIGH_THRESHOLD")
    optional.add_argument("-ic", "--if_collapse", dest="collapse",
                        help="Indicate if features are collapsed for intermediate report. Has to be True or False. Default is False.", metavar="IF_COLLAPSE")
    optional.add_argument("-ff", "--figure_format", dest="figure_format",
                        help="Format for figure. Can be any figure format available in matplotlib. Default is pdf.", metavar="FIGURE_FORMAT")
    optional.add_argument("-tn", "--thread_number", dest="thread_number",
                        help="Number of thread used to run the program. Default is 1, -1 indicate all thread used.", metavar="THREAD_NUMBER")
    optional.add_argument("-n", "--normalize", dest="normalize",
                        help="Way to normalize the target. Can be standard or zerobase. Default is zerobase.", metavar="NORMALIZE")
    optional.add_argument("-gl", "--gene_locus", dest="gene_locus",
                        help="File name to indicate gene locus. Check documentation for format. Default is None.", metavar="GENE_LOCUS")
    optional.add_argument("-v", "--verbose", dest="verbose",
                        help="Verbose level for report. Setting 0 to get no report. Default is 1.", metavar="VERBOSE")
    optional.add_argument("-mi", "--max_iteration", dest="max_iteration",
                        help="Number of iteration. Has to be int. -1 indicates stop by automatic detection. Default is -1", metavar="MAX_ITERATION")
    optional.add_argument("-s", "--subset", dest="subset",
                        help="Subset of samples to be chosen. Has to be file with one column with subset of sample names in target file. Default is use all columns.", metavar="SUBSET")
    optional.add_argument("-ip", "--if_pvalue", dest="if_pval", 
                        help="Indicate if pvalues are calculated and plotted on figure. Has to be True or Flase. Default is True.", metavar="IF_PVALUE")
    optional.add_argument("-ib", "--if_bootstrap", dest="if_bootstrap", 
                        help="Indicate if variance are calculated by bootstrap and plotted on figure. Has to be True or Flase. Default is True.", metavar="IF_BOOTSTRAP")
    optional.add_argument("-icl", "--if_cluster", dest="if_cluster", 
                        help="Indicate if features are clustered using NMF for intermediate report. Has to be True or Flase. Default is False.", metavar="IF_CLUSTER")
    optional.add_argument("-ii", "--if_intermediate", dest="if_intermediate", 
                        help="Indicate if intermediate reportes are generated. Has to be True or Flase. Default is True.", metavar="IF_INTERMEDIATE")
    optional.add_argument("-o", "--out_folder", dest="out_folder",
                        help="Path to directory to put output files. default is current directory.", metavar="OUT_FOLDER")
    optional.add_argument("-sep", "--separator", dest="separator",
                        help="Separator between gene name and later part.", metavar="SEPARATOR")
    optional.add_argument("-gs", "--gene_set", dest="gene_set",
                        help="Gene Set that is extracted.", metavar="GENE_SET")
    optional.add_argument("-gmt", "--gmt_file", dest="gmt_file",
                        help="gmt file for allele information.", metavar="GMT_FILE")
    optional.add_argument("-a", "--alpha", dest="alpha",
                        help="Power to raise value. Default is 1.", metavar="ALPHA")
    optional.add_argument("-gc", "--grid_check", dest="grid_check",
                        help="Indicate if run Grid Check mode. Default is False.", metavar="GRID_CHECK")
    optional.add_argument("-tif", "--tissue_file", dest="tissue_file",
                        help="File of tissue information.", metavar="TISSUE_FILE")
    optional.add_argument("-gz", "--gzip", dest="gzip",
                        help="If output file should be gzipped.", metavar="GZIP")
    optional.add_argument("-nh", "--neighborhood", dest="neighborhood",
                        help="number of neighborhood.", metavar="NEIGHBORHOOD")

    args = parser.parse_args()
    args = vars(args)
    print(args)

    if args['feature_file'][-3:] != 'gct':
        print('Feature file has to be in gct format.')
        sys.exit(1)

    if args['target_file'][-3:] != 'gct':
        print('Target file has to be in gct format.')
        sys.exit(1)

    if args['gmt_file'] != None:
        if args['gmt_file'][-3:] != 'gmt':
            print('gmt file has to be in gmt format.')
            sys.exit(1)
        else:
            gmt_file = args['gmt_file']
    else:
        gmt_file = None

    if args['seed_file'] != None:
        if args['seed_file'][-3:] != 'gct':
            print('Seed file has to be in gct format.')
            sys.exit(1)
        else:
            seed_file = args['seed_file']
    else:
        seed_file = args['feature_file']

    if args['prefix'] != None:
        prefix = args['prefix']
    else:
        prefix = 'REVEALER'

    if args['seed_name'] != None:
        if args['seed_name'][-3:] != 'txt':
            seed_name =  args['seed_name'].split(',')
        else:
            with open(args['seed_name'],'r') as f:
                seed_name = [line.rstrip() for line in f]
    else:
        seed_name = None

    if args['grid'] != None:
        try:
            grid = int(args['grid'])
        except ValueError:
            print('Grid has to be int')
            sys.exit(1)
    else:
        grid = 34

    if args['target_name'] != None:
        try:
            target_name = int(args['target_name'])
        except ValueError:
            target_name = args['target_name']
    else:
        target_name = 0

    if args['k'] != None:
        try:
            k = int(args['k'])
        except ValueError:
            print('K has to be int')
            sys.exit(1)
    else:
        k = 5

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

    if args['direction'] != None:
        if args['direction'] == 'positive':
            direction = 'pos'
        elif args['direction'] == 'negative':
            direction = 'neg'
        else:
            print('Direction has to be either positive or negative')
            sys.exit(1)
    else:
        direction = 'pos'

    if args['mode'] != None:
        if args['mode'] == 'single':
            mode = 'single'
        elif args['mode'] == 'multiple':
            mode = 'multiple'
            if args['parameter_set'] == None:
                print('Please specify parameter set if use mode multiple.')
                sys.exit(1)
            else:
                if args['parameter_set'][-3:] == 'txt':
                    params = pd.read_csv(args['parameter_set'],sep='\t')
                    prefix = params['prefix'].tolist()
                    grid = params['grid'].tolist()
                    k = params['k'].tolist()
                    bandwidth_mult = params['bandwidth_mult'].tolist()
                    bandwidth_adj = params['bandwidth_adj'].tolist()
                else:
                    paramall = args['parameter_set'].split(';')
                    paramlist = []
                    for i in paramall:
                        paramlist.append(i.split(','))
                    params = pd.DataFrame(paramlist,columns=['prefix','grid','k','bandwidth_mult','bandwidth_adj'])
                    prefix = params['prefix'].tolist()
                    grid = params['grid'].tolist()
                    k = params['k'].tolist()
                    bandwidth_mult = params['bandwidth_mult'].tolist()
                    bandwidth_adj = params['bandwidth_adj'].tolist()
        else:
            print('Mode has to be either single or multiple')
            sys.exit(1)
    else:
        mode = 'single'

    if args['num_top'] != None:
        try:
            num_top = int(args['num_top'])
        except ValueError:
            print('Num_top has to be int')
            sys.exit(1)
    else:
        num_top = 30

    if args['low_threshold'] != None:
        try:
            low_threshold = float(args['low_threshold'])
        except ValueError:
            print('Low_threshold has to be int or float')
            sys.exit(1)
    else:
        low_threshold = 3

    if args['high_threshold'] != None:
        try:
            high_threshold = float(args['high_threshold'])
        except ValueError:
            print('High_threshold has to be int or float')
            sys.exit(1)
    else:
        high_threshold = 100

    if args['collapse'] == None:
        collapse = True
    elif args['collapse'] == 'True':
        collapse = True
    elif args['collapse'] == 'False':
        collapse = False
    else:
        print('Only True or False accepted for collapse')
        sys.exit(1)

    if args['normalize'] == None:
        normalize = 'zerobase'
    elif args['normalize'] == 'standard':
        normalize = 'standard'
    elif args['normalize'] == 'zerobase':
        normalize = 'zerobase'
    else:
        print('Only standard or zerobase accepted for normalize')
        sys.exit(1)

    if args['gene_locus'] == None:
        gene_locus = 'None'
    else:
        gene_locus = args['gene_locus']

    if args['verbose'] != None:
        try:
            verbose = int(args['verbose'])
        except ValueError:
            print('verbose has to be int')
            sys.exit(1)
    else:
        verbose = 1

    if args['max_iteration'] != None:
        try:
            max_iteration = int(args['max_iteration'])
        except ValueError:
            print('max_iteration has to be int')
            sys.exit(1)
    else:
        max_iteration = -1

    if args['thread_number'] != None:
        try:
            thread_number = int(args['thread_number'])
        except ValueError:
            print('thread_number has to be int')
            sys.exit(1)
    else:
        thread_number = 1

    if args['figure_format'] != None:
        figure_format = args['figure_format']
    else:
        figure_format = 'pdf'

    if args['subset'] != None:
        with open(args['subset'],'r') as f:
            subset = [line.rstrip() for line in f]
    else:
        subset = 'no'

    if args['if_pval'] == None:
        if_pval = True
    elif args['if_pval'] == 'True':
        if_pval = True
    elif args['if_pval'] == 'False':
        if_pval = False
    else:
        print('Only True or False accepted for if_pvalue')
        sys.exit(1)

    if args['if_bootstrap'] == None:
        if_bootstrap = True
    elif args['if_bootstrap'] == 'True':
        if_bootstrap = True
    elif args['if_bootstrap'] == 'False':
        if_bootstrap = False
    else:
        print('Only True or False accepted for if_bootstrap')
        sys.exit(1)

    if args['if_cluster'] == None:
        if_cluster = True
    elif args['if_cluster'] == 'True':
        if_cluster = True
    elif args['if_cluster'] == 'False':
        if_cluster = False
    else:
        print('Only True or False accepted for if_cluster')
        sys.exit(1)

    if args['if_intermediate'] == None:
        if_intermediate = True
    elif args['if_intermediate'] == 'True':
        if_intermediate = True
    elif args['if_intermediate'] == 'False':
        if_intermediate = False
    else:
        print('Only True or False accepted for if_intermediate')
        sys.exit(1)

    if args['out_folder'] != None:
        out_folder = args['out_folder']
    else:
        out_folder = './'

    if args['separator'] != None:
        separator = args['separator']
    else:
        separator = '_'

    if args['gene_set'] != None:
        if args['gene_set'][-3:] != 'txt':
            gene_set =  args['gene_set'].split(',')
        else:
            with open(args['gene_set'],'r') as f:
                gene_set = [line.rstrip() for line in f]
    else:
        gene_set = None    


    if args['alpha'] != None:
        try:
            alpha = float(args['alpha'])
        except ValueError:
            print('alpha has to be int or float')
            sys.exit(1)
    else:
        alpha = 1

    if args['grid_check'] == None:
        grid_check = False
    elif args['grid_check'] == 'True':
        grid_check = True
    elif args['grid_check'] == 'False':
        grid_check = False
    else:
        print('Only True or False accepted for grid_check')
        sys.exit(1)

    if args['tissue_file'] != None:
        if args['tissue_file'][-3:] != 'gct':
            print('tissue file has to be in gct format.')
            sys.exit(1)
        else:
            tissue_file = args['tissue_file']
    else:
        tissue_file = None

    if args['gzip'] == None:
        gzip = True
    elif args['gzip'] == 'True':
        gzip = True
    elif args['gzip'] == 'False':
        gzip = False
    else:
        print('Only True or False accepted for gzip')
        sys.exit(1)

    if args['neighborhood'] != None:
        try:
            neighborhood = int(args['neighborhood'])
        except ValueError:
            print('neighborhood has to be int')
            sys.exit(1)
    else:
        neighborhood = 4

    if verbose > 0:
        print("""
Parameters utilized:
target_file="""+args['target_file']+"""
feature_file="""+args['feature_file']+"""
seed_file="""+str(seed_file)+"""
prefix="""+prefix+"""
seed_name="""+str(seed_name)+"""
grid="""+str(grid)+"""
target_name="""+str(target_name)+"""
k="""+str(k)+"""
bandwidth_mult="""+str(bandwidth_mult)+"""
bandwidth_adj="""+str(bandwidth_adj)+"""
direction="""+direction+"""
mode="""+mode+"""
num_top="""+str(num_top)+"""
low_threshold="""+str(low_threshold)+"""
high_threshold="""+str(high_threshold)+"""
collapse="""+str(collapse)+"""
normalize="""+normalize+"""
gene_locus="""+gene_locus+"""
verbose="""+str(verbose)+"""
max_iteration="""+str(max_iteration)+"""
thread_number="""+str(thread_number)+"""
figure_format="""+figure_format+"""
subset="""+str(subset)+"""
if_pval="""+str(if_pval)+"""
if_bootstrap="""+str(if_bootstrap)+"""
if_cluster="""+str(if_cluster)+"""
if_intermediate="""+str(if_intermediate)+"""
out_folder="""+out_folder+"""
gzip="""+str(gzip))

    #if grid_check == False:
    runREVEALER(target_file=args['target_file'], # gct file for target(continuous or binary)
                    feature_file=args['feature_file'], # gct file for features(binary)
                    seed_file=seed_file, # file for seed, if not provided, feature file is used directly 
                    prefix=prefix, # prefix for result files 
                    seed_name=seed_name, # names for seed, should be a list of string indicating the name of seed
                    grid=grid, # number of grid, default is 34
                    target_name=target_name, # name/index of target in target file. can be int n for nth row, or string s for row with index s
                    k=k, # size of kernel for k standard deviation away
                    bandwidth_mult=bandwidth_mult, # multiplication for bandwidth
                    bandwidth_adj=bandwidth_adj, # adjustion value for bandwidth
                    direction=direction, # direction that feature should match with target
                    mode=mode, # indicate if multiple parameter set is passes. if True, then prefix, k, grid, bandwidth_mult, and bandwidth_adj has to be a list
                    num_top=num_top, # number of top matches shown in intermediate file
                    low_threshold=low_threshold, # lowest threshold that feature with less than this value occurence will be removed
                    high_threshold=high_threshold, # highest threshold that feature with less than this value occurence will be removed
                    collapse=collapse, # indicate if same features are collapsed together for intermediate files
                    normalize=normalize, # normalize method for target
                    gene_locus=gene_locus, # gene_locus file indicating gene name and location of that gene
                    verbose=verbose, # verbose level(if 0, no report)
                    max_iteration=max_iteration, # maximum of iteration for best CIC discovery, automatic detection by IC value if -1 
                    thread_number=thread_number, # number of core used for parallel computing.
                    figure_format=figure_format, # format for result figure
                    subset=subset, # if list of string passed, only columns in this list is picked for calculation
                    if_pval=if_pval, # if True, p-values are calculated for all result
                    if_bootstrap=if_bootstrap, # if True, variance by bootstrap is calculated for all result
                    if_cluster=if_cluster, # if True, features in intermediate files are clustered with NMF 
                    if_intermediate=if_intermediate, # if True, intermediate result with top CIC value features are reported
                    out_folder=out_folder, # folder to put output files inside
                    gene_set = gene_set, # set of gene to run REVEALER
                    gene_separator = separator,  #separator between gene and later part
                    gmt_file = gmt_file,
                    alpha = alpha,
                    tissue_file = tissue_file,
                    gzip = gzip,
                    neighborhood = neighborhood
                    )
    #else:
        # runCheckGrid(target_file=args['target_file'], # gct file for target(continuous or binary)
        #             feature_file=args['feature_file'], # gct file for features(binary)
        #             seed_file=seed_file, # file for seed, if not provided, feature file is used directly 
        #             prefix=prefix, # prefix for result files 
        #             seed_name=seed_name, # names for seed, should be a list of string indicating the name of seed
        #             grid=grid, # number of grid, default is 34
        #             target_name=target_name, # name/index of target in target file. can be int n for nth row, or string s for row with index s
        #             k=k, # size of kernel for k standard deviation away
        #             bandwidth_mult=bandwidth_mult, # multiplication for bandwidth
        #             bandwidth_adj=bandwidth_adj, # adjustion value for bandwidth
        #             direction=direction, # direction that feature should match with target
        #             mode=mode, # indicate if multiple parameter set is passes. if True, then prefix, k, grid, bandwidth_mult, and bandwidth_adj has to be a list
        #             num_top=num_top, # number of top matches shown in intermediate file
        #             low_threshold=low_threshold, # lowest threshold that feature with less than this value occurence will be removed
        #             high_threshold=high_threshold, # highest threshold that feature with less than this value occurence will be removed
        #             collapse=collapse, # indicate if same features are collapsed together for intermediate files
        #             normalize=normalize, # normalize method for target
        #             gene_locus=gene_locus, # gene_locus file indicating gene name and location of that gene
        #             verbose=verbose, # verbose level(if 0, no report)
        #             max_iteration=max_iteration, # maximum of iteration for best CIC discovery, automatic detection by IC value if -1 
        #             thread_number=thread_number, # number of core used for parallel computing.
        #             figure_format=figure_format, # format for result figure
        #             subset=subset, # if list of string passed, only columns in this list is picked for calculation
        #             if_pval=if_pval, # if True, p-values are calculated for all result
        #             if_bootstrap=if_bootstrap, # if True, variance by bootstrap is calculated for all result
        #             if_cluster=if_cluster, # if True, features in intermediate files are clustered with NMF 
        #             if_intermediate=if_intermediate, # if True, intermediate result with top CIC value features are reported
        #             out_folder=out_folder, # folder to put output files inside
        #             gene_set = gene_set, # set of gene to run REVEALER
        #             gene_separator = separator,  #separator between gene and later part
        #             gmt_file = gmt_file,
        #             alpha = alpha
        #             )

