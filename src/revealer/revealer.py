import argparse
from revealer.REVEALER_Cython import runREVEALER
from revealer.MutMaker import produce_mutation_file
from revealer.REVEALER_test import run_test
import sys

def main():
    parser = argparse.ArgumentParser(description='REVEALER: A tool for identifying significant mutually exclusive features.')
    subparsers = parser.add_subparsers(dest='subcommand')

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocessing MAF file for REVEALER input.')
    preprocess_required = preprocess_parser.add_argument_group('required arguments')
    preprocess_optional = preprocess_parser.add_argument_group('optional arguments')
    
    preprocess_required.add_argument("-i", "--input_file", dest="input_file", required=True,
                                     help="Name of input file, can be maf file or gct file.")
    preprocess_required.add_argument("-pi", "--protein_change_identifier", dest="protein_change_identifier", required=True,
                                     help="protein change identifier for dataset.")
    preprocess_required.add_argument("-cg", "--col_genename", dest="col_genename", required=True,
                                     help="column with gene name information.")
    preprocess_required.add_argument("-cc", "--col_class", dest="col_class", required=True,
                                     help="column with class information.")
    preprocess_required.add_argument("-csa", "--col_sample", dest="col_sample", required=True,
                                     help="column with sample name information.")
    
    preprocess_optional.add_argument("-m", "--mode", dest="mode", default='class',
                                     help="Mode, can be class, freq, weight, all, or comb. For more detail, check documentation. Default is class.")
    preprocess_optional.add_argument("-r", "--ratio", dest="ratio", type=float, default=1/3,
                                     help="Ratio of selected features by weight that is acceptable. Default is 1/3.")
    preprocess_optional.add_argument("-pf", "--phenotype_file", dest="phenotype_file", default=None,
                                     help="Name of phenotype file. Required if mode is one of freq, weight, all, or comb.")
    preprocess_optional.add_argument("-c", "--class_file", dest="class_file", default=None,
                                     help="File for classifiers, required if input file is gct file.")
    preprocess_optional.add_argument("-p", "--prefix", dest="prefix", default='Mut',
                                     help="prefix for output files. default is Mut.")
    preprocess_optional.add_argument("-o", "--out_folder", dest="out_folder", default='./',
                                     help="Path to directory to put output files. default is current directory.")
    preprocess_optional.add_argument("-cs", "--class_separator", dest="class_separator", default='_',
                                     help="Separator between gene name and class name. default is '_'")
    preprocess_optional.add_argument("-pn", "--phenotype_name", dest="phenotype_name", default=0,
                                     help="Name of phenotypr to be used in phenotype file. Default is first row.")
    preprocess_optional.add_argument("-fs", "--file_separator", dest="file_separator", default='\t',
                                     help="Separator of file, default is tab('\\t')")
    preprocess_optional.add_argument("-d", "--direction", dest="direction", default='pos',
                                     help="direction of phenotype the features should match. Default is positive.")
    preprocess_optional.add_argument("-ft", "--frequency_threshold", dest="frequency_threshold", type=int, default=5,
                                     help="Threshold for frequency. Default is 5.")
    preprocess_optional.add_argument("-wt", "--weight_threshold", dest="weight_threshold", type=float, default=0.7,
                                     help="Threshold for weight. Default is 0.7.")
    preprocess_optional.add_argument("-gl", "--gene_list", dest="gene_list", default=None,
                                     help="List of Gene to run. Can be file or ',' separated names. Default is all gene.")
    preprocess_optional.add_argument("-nm", "--name_match", dest="name_match", action='store_true', default=True,
                                     help="Indicate if sample name in phenotype file and input file are matching. Default is True.")
    preprocess_optional.add_argument("-mf", "--make_figure", dest="make_figure", action='store_true', default=False,
                                     help="Indicate if figures are created. Recommended only when you run few genes. Default is False.")
    preprocess_optional.add_argument("-ff", "--figure_format", dest="figure_format", default='pdf',
                                     help="Format for figure. Default is pdf.")
    preprocess_optional.add_argument("-sl", "--sample_list", dest="sample_list", default=None,
                                     help="List of sample to be used.")
    preprocess_optional.add_argument("-tr", "--total_ratio", dest="total_ratio", type=float, default=0.4,
                                     help="Ratio of gene occrence compared to total sample number. default is 0.4")
    preprocess_optional.add_argument("-ig", "--if_gmt", dest="if_gmt", action='store_true', default=True,
                                     help="Indicate if gmt file should be generated. Default if True.")
    preprocess_optional.add_argument("-v", "--verbose", dest="verbose", type=int, default=1,
                                     help="Indicate level of verbose. default is 1.")
    preprocess_optional.add_argument("-k", "--k_size", dest="k", type=int, default=30,
                                     help="Size of kernel indicating k variance far from middle. Has to be int. Default is 5")
    preprocess_optional.add_argument("-bm", "--bandwidth_multiplication", dest="bandwidth_mult", type=float, default=0.65,
                                     help="Value of bandwidth multiplier. Has to be float or int. Default is 0.65.")
    preprocess_optional.add_argument("-ba", "--bandwidth_adjustment", dest="bandwidth_adj", type=float, default=-0.95,
                                     help="Value of bandwidth adjustment. Has to be float or int. Default is -0.95")
    preprocess_optional.add_argument("-gz", "--gzip", dest="gzip", action='store_true', default=True,
                                     help="If output file should be gzipped.")
    preprocess_optional.add_argument("-cb", "--combine", dest="combine", action='store_true', default=True,
                                     help="Combine alleles by gene when making figures")

    # Run subcommand
    run_parser = subparsers.add_parser('run', help='Run REVEALER analysis.')
    run_required = run_parser.add_argument_group('required arguments')
    run_optional = run_parser.add_argument_group('optional arguments')

    run_required.add_argument("-tf", "--target_file", dest="target_file", required=True,
                              help="Name of target file, has to be gct file.")
    run_required.add_argument("-f", "--feature_files", dest="feature_files", required=True,
                              help="Name of feature file, has to be gct file.")

    run_optional.add_argument("-sf", "--seed_files", dest="seed_files", default=None,
                              help="Name of seed file.")
    run_optional.add_argument("-p", "--prefix", dest="prefix", default='REVEALER',
                              help="Prefix for results files. Default is REVEALER.")
    run_optional.add_argument("-sn", "--seed_name", dest="seed_name", default=None,
                              help="Seed names to take from seed file or feature file. If multiple exist, separate by ','. Passing text file(ending with 'txt') with column of seed names also works")
    run_optional.add_argument("-g", "--grid", dest="grid", type=int, default=34,
                              help="Grid size to put kernel on. has to be int. Default is 34.")
    run_optional.add_argument("-t", "--target_name", dest="target_name", default=0,
                              help="Name of target in target file. Can be row name or row index(0 based). Default is first row.")
    run_optional.add_argument("-k", "--k_size", dest="k", type=int, default=5,
                              help="Size of kernel indicating k variance far from middle. Has to be int. Default is 5")
    run_optional.add_argument("-bm", "--bandwidth_multiplication", dest="bandwidth_mult", type=float, default=0.65,
                              help="Value of bandwidth multiplier. Has to be float or int. Default is 0.65.")
    run_optional.add_argument("-ba", "--bandwidth_adjustment", dest="bandwidth_adj", type=float, default=-0.95,
                              help="Value of bandwidth adjustment. Has to be float or int. Default is -0.95")
    run_optional.add_argument("-d", "--direction", dest="direction", default='pos',
                              help="Direction of phenotype the features should match. Default is positive.")
    run_optional.add_argument("-m", "--mode", dest="mode", default='single',
                              help="Mode to run REVEALER. Can be single or multiple. If set to multiple, format as sample input is required. Default is single.")
    run_optional.add_argument("-ps", "--parameter_set", dest="parameter_set", default=None,
                              help="set of parameter. Required if mode is multiple. Check available format in documentation.")
    run_optional.add_argument("-nt", "--num_top", dest="num_top", type=int, default=30,
                              help="Number of top features picked for intermediate report. Has to be int. Default is 30.")
    run_optional.add_argument("-lt", "--low_threshold", dest="low_threshold", type=float, default=3,
                              help="Lower threshold to remove feature. If int passed, threshold of absolute value is made. If float passed, threshold made by ratio. Default is 3.")
    run_optional.add_argument("-ht", "--high_threshold", dest="high_threshold", type=float, default=100,
                              help="Higher threshold to remove feature. If int passed, threshold of absolute value is made. If float passed, threshold made by ratio. Default is 0.2.")
    run_optional.add_argument("-ic", "--if_collapse", dest="collapse", action='store_true', default=True,
                              help="Indicate if features are collapsed for intermediate report. Has to be True or False. Default is False.")
    run_optional.add_argument("-ff", "--figure_format", dest="figure_format", default='pdf',
                              help="Format for figure. Can be any figure format available in matplotlib. Default is pdf.")
    run_optional.add_argument("-tn", "--thread_number", dest="thread_number", type=int, default=1,
                              help="Number of thread used to run the program. Default is 1, -1 indicate all thread used.")
    run_optional.add_argument("-n", "--normalize", dest="normalize", default='zerobase',
                              help="Way to normalize the target. Can be standard or zerobase. Default is zerobase.")
    run_optional.add_argument("-gl", "--gene_locus", dest="gene_locus", default='None',
                              help="File name to indicate gene locus. Check documentation for format. Default is None.")
    run_optional.add_argument("-v", "--verbose", dest="verbose", type=int, default=1,
                              help="Verbose level for report. Setting 0 to get no report. Default is 1.")
    run_optional.add_argument("-mi", "--max_iteration", dest="max_iteration", type=int, default=-1,
                              help="Number of iteration. Has to be int. -1 indicates stop by automatic detection. Default is -1")
    run_optional.add_argument("-s", "--subset", dest="subset", default='no',
                              help="Subset of samples to be chosen. Has to be file with one column with subset of sample names in target file. Default is use all columns.")
    run_optional.add_argument("-ip", "--if_pvalue", dest="if_pval", action='store_true', default=True,
                              help="Indicate if pvalues are calculated and plotted on figure. Has to be True or False. Default is True.")
    run_optional.add_argument("-ib", "--if_bootstrap", dest="if_bootstrap", action='store_true', default=True,
                              help="Indicate if variance are calculated by bootstrap and plotted on figure. Has to be True or False. Default is True.")
    run_optional.add_argument("-icl", "--if_cluster", dest="if_cluster", action='store_true', default=False,
                              help="Indicate if features are clustered using NMF for intermediate report. Has to be True or False. Default is False.")
    run_optional.add_argument("-ii", "--if_intermediate", dest="if_intermediate", action='store_true', default=True,
                              help="Indicate if intermediate reports are generated. Has to be True or False. Default is True.")
    run_optional.add_argument("-o", "--out_folder", dest="out_folder", default='./',
                              help="Path to directory to put output files. default is current directory.")
    run_optional.add_argument("-sep", "--separator", dest="separator", default='_',
                              help="Separator between gene name and later part.")
    run_optional.add_argument("-gs", "--gene_set", dest="gene_set", default=None,
                              help="Gene Set that is extracted.")
    run_optional.add_argument("-gmt", "--gmt_file", dest="gmt_file", default=None,
                              help="gmt file for allele information.")
    run_optional.add_argument("-a", "--alpha", dest="alpha", type=float, default=1,
                              help="Power to raise value. Default is 1.")
    run_optional.add_argument("-gc", "--grid_check", dest="grid_check", action='store_true', default=False,
                              help="Indicate if run Grid Check mode. Default is False.")
    run_optional.add_argument("-tif", "--tissue_file", dest="tissue_file", default=None,
                              help="File of tissue information.")
    run_optional.add_argument("-gz", "--gzip", dest="gzip", action='store_true', default=True,
                              help="If output file should be gzipped.")
    run_optional.add_argument("-nh", "--neighborhood", dest="neighborhood", type=int, default=4,
                              help="number of neighborhood.")
    run_optional.add_argument("-fr", "--feature_remove", dest="feature_remove", default=None,
                              help="features to remove manually.")
    run_optional.add_argument("-c", "--color", dest="color", default='blue',
                              help="tune of color palette.")
    run_optional.add_argument("-lw", "--line_width", dest="line_width", type=float, default=None,
                              help="width of line.")
    run_optional.add_argument("-fs", "--feature_set", dest="feature_set", default=None,
                              help="set of feature to use.")

    test_parser = subparsers.add_parser('test', help='Testing the work of REVEALER after download.')

    args = parser.parse_args()

    if args.subcommand == 'preprocess':
        preprocess(args)
    elif args.subcommand == 'run':
        run(args)
    elif args.subcommand == 'test':
        run_test(args)
    else:
        parser.print_help()

def preprocess(args):
    if args.input_file[-3:] != 'gct' and args.input_file[-3:] != 'maf' and args.input_file[-3:] != 'csv':
        print('only maf file and gct file are accepted as input, please check your input file.')
        sys.exit(1)
    else:
        input_file = args.input_file

    if args.mode not in ['mutall', 'allele', 'class', 'freq', 'weight', 'weight_filter', 'comb']:
        print('Mode not available! Please check instruction for available mode.')
        sys.exit(1)

    if (args.mode in ['freq', 'weight', 'weight_filter', 'comb']) and args.phenotype_file is None:
        print('Please indicate phenotype file if use mode ' + args.mode)
        sys.exit(1)

    if args.input_file[-3:] == 'gct' and args.class_file is None:
        print('Please indicate class names by passing class file.')
        sys.exit(1)

    if args.input_file[-3:] == 'gct':
        with open(args.class_file, 'r') as f:
            class_list = [line.rstrip() for line in f]
    else:
        class_list = ['Nonsense_Mutation', 'In_Frame_Del', 'Silent', 'Frame_Shift_Ins', 'Missense_Mutation', 'Splice_Site',
                      'Frame_Shift_Del', 'De_novo_Start_OutOfFrame', 'Nonstop_Mutation', 'In_Frame_Ins', 'Start_Codon_SNP',
                      'Start_Codon_Del', 'Stop_Codon_Ins', 'Start_Codon_Ins', 'Stop_Codon_Del', 'Intron', 'IGR', "5'Flank",
                      "3'UTR", "5'UTR", 'Mut_All']

    if args.gene_list is not None:
        if args.gene_list[-3:] != 'txt':
            gene_list = args.gene_list.split(',')
        else:
            with open(args.gene_list, 'r') as f:
                gene_list = [line.rstrip() for line in f]
    else:
        gene_list = None

    if args.sample_list is not None:
        with open(args.sample_list, 'r') as f:
            sample_list = [line.rstrip() for line in f]
    else:
        sample_list = None

    if args.verbose > 0:
        print(f"""
    Parameters utilized:
    input_file={input_file}
    protein_change_identifier={args.protein_change_identifier}
    mode={args.mode}
    class_list={str(class_list)}
    prefix={args.prefix}
    ratio={str(args.ratio)}
    out_folder={str(args.out_folder)}
    class_separator={str(args.class_separator)}
    phenotype_name={str(args.phenotype_name)}
    file_separator={str(args.file_separator)}
    direction={args.direction}
    frequency_threshold={str(args.frequency_threshold)}
    weight_threshold={str(args.weight_threshold)}
    total_ratio={str(args.total_ratio)}
    gene_list={str(gene_list)}
    sample_list={str(sample_list)}
    name_match={str(args.name_match)}
    make_figure={str(args.make_figure)}
    figure_format={str(args.figure_format)}
    if_gmt={str(args.if_gmt)}
    verbose={str(args.verbose)}
    k={str(args.k)}
    bandwidth_mult={str(args.bandwidth_mult)}
    bandwidth_adj={str(args.bandwidth_adj)}
    gzip={str(args.gzip)}
    combine={str(args.combine)}
    """)

    produce_mutation_file(
        maf_input_file=input_file if input_file[-3:] == 'maf' else None,
        gct_input_file=input_file if input_file[-3:] == 'gct' else None,
        gct_output_file_prefix=args.prefix,
        phenotype_file=args.phenotype_file,
        class_list=class_list,
        class_seperator=args.class_separator,
        phenotype_name=args.phenotype_name,
        file_separator=args.file_separator,
        protein_change_identifier=args.protein_change_identifier,
        mode=args.mode,
        direction=args.direction,
        frequency_threshold=args.frequency_threshold,
        weight_threshold=args.weight_threshold,
        gene_list=gene_list,
        name_match=args.name_match,
        make_figure=args.make_figure,
        figure_format=args.figure_format,
        out_folder=args.out_folder,
        ratio=args.ratio,
        sample_list=sample_list,
        total_ratio=args.total_ratio,
        if_gmt=args.if_gmt,
        k=args.k,
        gzip=args.gzip,
        combine=args.combine,
        col_genename=args.col_genename,
        col_class=args.col_class,
        col_sample=args.col_sample
    )



def run(args):
    feature_files = args.feature_files.split(',')
    for feature_file in feature_files:
        if feature_file[-3:] != 'gct':
            print('Feature files has to be in gct format.')
            sys.exit(1)
    
    if args.target_file[-3:] != 'gct':
        print('Target file has to be in gct format.')
        sys.exit(1)

    if args.gmt_file is not None:
        if args.gmt_file[-3:] != 'gmt':
            print('gmt file has to be in gmt format.')
            sys.exit(1)
        else:
            gmt_file = args.gmt_file
    else:
        gmt_file = None

    if args.seed_files is not None:
        if args.seed_files[-3:] != 'gct':
            print('Seed file has to be in gct format.')
            sys.exit(1)
        else:
            seed_files = args.seed_files.split(',')
    else:
        seed_files = args.feature_files.split(',')

    if args.seed_name is not None:
        if args.seed_name[-3:] != 'txt':
            seed_name = args.seed_name.split(',')
        else:
            with open(args.seed_name, 'r') as f:
                seed_name = [line.rstrip() for line in f]
    else:
        seed_name = None

    if args.direction == 'positive':
        direction = 'pos'
    elif args.direction == 'negative':
        direction = 'neg'
    else:
        print('Direction has to be either positive or negative')
        sys.exit(1)

    if args.mode == 'multiple':
        if args.parameter_set is None:
            print('Please specify parameter set if use mode multiple.')
            sys.exit(1)
        else:
            if args.parameter_set[-3:] == 'txt':
                params = pd.read_csv(args.parameter_set, sep='\t')
                prefix = params['prefix'].tolist()
                grid = params['grid'].tolist()
                k = params['k'].tolist()
                bandwidth_mult = params['bandwidth_mult'].tolist()
                bandwidth_adj = params['bandwidth_adj'].tolist()
            else:
                paramall = args.parameter_set.split(';')
                paramlist = []
                for i in paramall:
                    paramlist.append(i.split(','))
                params = pd.DataFrame(paramlist, columns=['prefix', 'grid', 'k', 'bandwidth_mult', 'bandwidth_adj'])
                prefix = params['prefix'].tolist()
                grid = params['grid'].tolist()
                k = params['k'].tolist()
                bandwidth_mult = params['bandwidth_mult'].tolist()
                bandwidth_adj = params['bandwidth_adj'].tolist()
    else:
        prefix = args.prefix
        grid = args.grid
        k = args.k
        bandwidth_mult = args.bandwidth_mult
        bandwidth_adj = args.bandwidth_adj

    if args.subset != 'no':
        with open(args.subset, 'r') as f:
            subset = [line.rstrip() for line in f]
    else:
        subset = args.subset

    if args.gene_set is not None:
        if args.gene_set[-3:] != 'txt':
            gene_set = args.gene_set.split(',')
        else:
            with open(args.gene_set, 'r') as f:
                gene_set = [line.rstrip() for line in f]
    else:
        gene_set = None

    if args.feature_remove is not None:
        if args.feature_remove[-3:] != 'txt':
            feature_remove = args.feature_remove.split(',')
        else:
            with open(args.feature_remove, 'r') as f:
                feature_remove = [line.rstrip() for line in f]
    else:
        feature_remove = None

    if args.feature_set is not None:
        if args.feature_set[-3:] != 'txt':
            feature_set = args.feature_set.split(',')
        else:
            with open(args.feature_set, 'r') as f:
                feature_set = [line.rstrip() for line in f]
    else:
        feature_set = None

    if args.verbose > 0:
        print(f"""
    Parameters utilized:
    target_file={args.target_file}
    feature_files={str(feature_files)}
    seed_files={str(seed_files)}
    prefix={prefix}
    seed_name={str(seed_name)}
    grid={str(grid)}
    target_name={str(args.target_name)}
    k={str(k)}
    bandwidth_mult={str(bandwidth_mult)}
    bandwidth_adj={str(bandwidth_adj)}
    direction={direction}
    mode={args.mode}
    num_top={str(args.num_top)}
    low_threshold={str(args.low_threshold)}
    high_threshold={str(args.high_threshold)}
    collapse={str(args.collapse)}
    normalize={args.normalize}
    gene_locus={args.gene_locus}
    verbose={str(args.verbose)}
    max_iteration={str(args.max_iteration)}
    thread_number={str(args.thread_number)}
    figure_format={args.figure_format}
    subset={str(subset)}
    if_pval={str(args.if_pval)}
    if_bootstrap={str(args.if_bootstrap)}
    if_cluster={str(args.if_cluster)}
    if_intermediate={str(args.if_intermediate)}
    out_folder={args.out_folder}
    gzip={str(args.gzip)}
    """)

    runREVEALER(target_file=args.target_file,
                feature_files=feature_files,
                seed_files=seed_files,
                prefix=prefix,
                seed_name=seed_name,
                grid=grid,
                target_name=args.target_name,
                k=k,
                bandwidth_mult=bandwidth_mult,
                bandwidth_adj=bandwidth_adj,
                direction=direction,
                mode=args.mode,
                num_top=args.num_top,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
                collapse=args.collapse,
                normalize=args.normalize,
                gene_locus=args.gene_locus,
                verbose=args.verbose,
                max_iteration=args.max_iteration,
                thread_number=args.thread_number,
                figure_format=args.figure_format,
                subset=subset,
                if_pval=args.if_pval,
                if_bootstrap=args.if_bootstrap,
                if_cluster=args.if_cluster,
                if_intermediate=args.if_intermediate,
                out_folder=args.out_folder,
                gene_set=gene_set,
                gene_separator=args.separator,
                gmt_file=gmt_file,
                alpha=args.alpha,
                tissue_file=args.tissue_file,
                gzip=args.gzip,
                neighborhood=args.neighborhood,
                feature_remove=feature_remove,
                color=args.color,
                linewidth=args.line_width,
                feature_set=feature_set)

if __name__ == '__main__':
    main()
