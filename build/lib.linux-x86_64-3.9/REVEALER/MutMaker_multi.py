import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import multiprocess as mp
import matplotlib.colors as clr
import sys
import argparse
import os

def make_gene_dic_class(gene_ds, gene, sample_dict):

    """
    Function to make dictionary by gene with class.
    """

    subdict = {}
    muttypeall = gene_ds['Variant_Classification'].unique().tolist()
    subdict[gene+'_All'] = [0]*len(sample_dict)
    muttype = [x for x in muttypeall if str(x) != 'nan']
    for submuttype in muttype:
        mut_ds = gene_ds[gene_ds['Variant_Classification'] == submuttype]
        subdict[gene+'_'+submuttype] = [0]*len(sample_dict)
        for i in mut_ds.index.tolist():
            subdict[gene+'_'+submuttype][sample_dict[mut_ds.loc[i]['Tumor_Sample_Barcode']]] = 1
            subdict[gene+'_All'][sample_dict[mut_ds.loc[i]['Tumor_Sample_Barcode']]] = 1
    return subdict

def make_gene_dic_type(gene_ds, gene, sample_dict, protein_change_identifier):

    """
    Function to make dictionary by gene with protein change.
    """

    subdict = {}
    
    muttypeall = gene_ds[protein_change_identifier].unique().tolist()
    muttype = [x for x in muttypeall if str(x) != 'nan']
    for submuttype in muttype:
        subdict[gene+'_'+submuttype] = [0]*len(sample_dict)
        for i in gene_ds[gene_ds[protein_change_identifier] == submuttype].index.tolist():
            subdict[gene+'_'+submuttype][sample_dict[gene_ds[gene_ds[protein_change_identifier] == submuttype].loc[i]['Tumor_Sample_Barcode']]] = 1
    return subdict

def select_by_weight(genedf, phenotype, weight_threshold, direction):
    
    """
    Function to subselect features by weight with given threshold.
    """

    weights = []
    for i in genedf.index.tolist():
        weight_vec = phenotype.iloc[0] * genedf.loc[i]
        weight = weight_vec.sum()/genedf.loc[i].sum()
        weights.append(weight)
        
    genedf['weights'] = weights
    if direction == 'pos':
        return genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False)
    else:
        return genedf[genedf['weights'] <= weight_threshold].sort_values(by='weights')


def select_by_weight_filter(genedf, phenotype, weight_threshold, direction, ratio):
    
    """
    Function to subselect features by weight with filter with given threshold and ratio.
    """

    weights = []
    for i in genedf.index.tolist():
        weight_vec = phenotype.iloc[0] * genedf.loc[i]
        weight = weight_vec.sum()/genedf.loc[i].sum()
        weights.append(weight)
        
    genedf['weights'] = weights
    if direction == 'pos':
        if (len(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index) != 0) and (seedcomball(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False),genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()).count(1) < (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
            return pd.DataFrame(columns=genedf.columns.tolist())
        else:
            return genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False)
        
    else:
        if (len(genedf[genedf['weights'] <= weight_threshold].sort_values(by='weights').index) != 0) and (seedcomball(genedf[genedf['weights'] <= weight_threshold].sort_values(by='weights'),genedf[genedf['weights'] <= weight_threshold].sort_values(by='weights').index.tolist()).count(1) < (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
            return pd.DataFrame(columns=genedf.columns.tolist())
        else:
            return genedf[genedf['weights'] <= weight_threshold].sort_values(by='weights')

def select_by_frequency(genedf, phenotype, frequency_threshold, direction):

    """
    Function to subselect features by frequency with given threshold.
    """

    counts = []
    weights = []
    for i in genedf.index.tolist():
        counts.append(genedf.loc[i].sum())
        weight_vec = phenotype.iloc[0] * genedf.loc[i]
        weight = weight_vec.sum()/genedf.loc[i].sum()
        weights.append(weight)
    genedf['counts'] = counts
    genedf['weights'] = weights
    
    if direction == 'pos':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, False])
    elif direction == 'neg':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, True])
    
    if frequency_threshold > len(genedf.index.tolist()):
        return genedf
    else:
        return genedf.iloc[:frequency_threshold]
    
def plotfreq(gct_output_file_prefix,frequency_threshold,phenotype,key,genedf,figformat,direction,out_folder):

    """
    Function to plot features separated with target with frequency and weight labeled.
    """

    counts = []
    weights = []
    for i in genedf.index.tolist():
        counts.append(genedf.loc[i].sum())
        weight_vec = phenotype.iloc[0] * genedf.loc[i]
        weight = weight_vec.sum()/genedf.loc[i].sum()
        weights.append(weight)
    genedf['counts'] = counts
    genedf['weights'] = weights
    
    if direction == 'pos':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, False])
    elif direction == 'neg':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, True])
    
    plotcomb = genedf.iloc[:,:-2]
    
    fig = plt.figure()
    fig.set_figheight(len(genedf.index.tolist())/2.0+1)
    fig.set_figwidth(11.6)
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 10), colspan=90,rowspan=5)
    ax = sns.heatmap(phenotype.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,cbar=False,
                     center=plotcomb.iloc[0].mean())
    ax.set_ylabel('Target',rotation=0,labelpad=45)
    ax.yaxis.set_label_coords(-0.1,0.3)
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Count',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 108), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Weight',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 10), colspan=90,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 108), colspan=8,rowspan=5)
    ax.set_axis_off()
    
    for i in range(0,len(counts)):
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 100), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%d"%(genedf['counts'].tolist()[i]),ha='center', va='center')
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 108), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%.3f"%(genedf['weights'].tolist()[i]),ha='center', va='center')

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Top'+str(frequency_threshold)+'_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    return genedf


def plotweight(gct_output_file_prefix,weight_threshold,phenotype,key,genedf,figformat,direction,out_folder):

    """
    Function to plot features with target, labeled with wieght.
    """

    weights = []
    for i in genedf.index.tolist():
        weight_vec = phenotype.iloc[0] * genedf.loc[i]
        weight = weight_vec.sum()/genedf.loc[i].sum()
        weights.append(weight)
        
    genedf['weights'] = weights
    
    if direction == 'pos':
        genedf = genedf.sort_values(['weights'], ascending=[False])
    elif direction == 'neg':
        genedf = genedf.sort_values(['weights'], ascending=[True])
    
    plotcomb = genedf.iloc[:,:-2]
    
    fig = plt.figure()
    fig.set_figheight(len(genedf.index.tolist())/2.0+1)
    fig.set_figwidth(10.8)
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(0, 10), colspan=90,rowspan=5)
    ax = sns.heatmap(phenotype.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,cbar=False,
                     center=plotcomb.iloc[0].mean())
    ax.set_ylabel('Target',rotation=0,labelpad=45)
    ax.yaxis.set_label_coords(-0.1,0.3)
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(0, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Weight',ha='center', va='center')

    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(5, 10), colspan=90,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(5, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    
    for i in range(0,len(weights)):
        ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=((i+2)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i].split('\n')[0]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i].split('\n')[0]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=((i+2)*5, 100), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%.3f"%(genedf['weights'].tolist()[i]),ha='center', va='center')
        

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Match'+str(weight_threshold)+'_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    return genedf



def plotclass(gct_output_file_prefix,key,genedf,figformat,out_folder):

    """
    Function to plot features separated with class with target.
    """

    plotcomb = genedf
    
    fig = plt.figure()
    fig.set_figheight(len(plotcomb.index.tolist())/2.0)
    fig.set_figwidth(10)
    
    for i in range(0,len(plotcomb.index.tolist())):
        ax = plt.subplot2grid(shape=(5*(len(plotcomb.index.tolist())+2),100), loc=((i)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Class_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    return genedf



def drawTarget(comb):

    """
    Function to draw continuous target.
    """

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap(comb.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,
                     cbar=False,center=comb.iloc[0].mean())
    return f

def drawFeature(comb,featurecmap,seedName=None,seedID=None):

    """
    Function to draw binary feature.
    """

    if seedName != None:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.loc[[seedName]].to_numpy(),cmap=featurecmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)
        return f
    else:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.iloc[[seedID]].to_numpy(),cmap=featurecmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)
        return f
    
def drawSeed(seed,seedcmap):

    """
    Function to combine all seed from comb that are in seedName.
    """

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap([seed],cmap=seedcmap,annot=False,yticklabels=False,xticklabels=False,cbar=False)
    return f



def seedcomball(comb,seedName):

    """
    Function to combine all seed from comb that are in seedName.
    """

    currentseed = comb.loc[seedName[0]].tolist()
    if len(seedName) == 1:
        return currentseed
    for subseed in seedName[1:]:
        currentseed = seedCombine(currentseed,comb.loc[subseed].tolist())
    return currentseed

def seedCombine(currentseed,newseed):

    """
    Function to combine two seed.
    """

    seed = []
    for i in range(len(currentseed)):
        if currentseed[i] == 1 or newseed[i] == 1:
            seed.append(1)
        else:
            seed.append(0)
    return seed


def produce_mutation_file(
    maf_input_file = None, # Input file with maf format.
    gct_input_file = None, # Input file with gct format.
    gct_output_file_prefix = 'Mut', # Prefix for output file
    phenotype_file = None, # Phenotype required if mode is freq or weight
    class_list = ['Nonsense_Mutation','In_Frame_Del','Silent','Frame_Shift_Ins','Missense_Mutation','Splice_Site',
                  'Frame_Shift_Del','De_novo_Start_OutOfFrame','Nonstop_Mutation','In_Frame_Ins','Start_Codon_SNP',
                  'Start_Codon_Del','Stop_Codon_Ins','Start_Codon_Ins','Stop_Codon_Del','Intron','IGR',"5'Flank",
                  "3'UTR","5'UTR",'Mut_All'], # list of class
    class_seperator = '_', # Separator between gene name and later information
    phenotype_name = 0, # Name of Phenotype, can be int or string
    file_separator = '\t', # Separator for file
    protein_change_identifier = 'Protein_Change', # Identifier for protein change column
    mode = 'class', # Mode to run the program 
    direction = 'pos', # Direction of features matching phenotype
    frequency_threshold = 5, # Threshold for frequency
    weight_threshold = 0.7, # Threshold for weight
    thread_number = 1, # Number of thread used for parallel computing
    gene_list = None, # Gene list if only part of gene is ran
    name_match = True, # Indicate if column name is perfect matching
    make_figure = False, # Indicate if heatmap for each gene is generated
    figure_format='pdf', # Format of figure
    out_folder='.', # Folder to put results
    ratio = float(1/3), # Ratio of selected features by weight that is acceptable
    sample_list = None
    ):

    """
    Function to create mutation gct file with given file using given mode.
    """
    
    seedcmap = clr.LinearSegmentedColormap.from_list('custom greys', [(.9,.9,.9),(0.5,0.5,0.5)], N=256)
    featurecmap = clr.LinearSegmentedColormap.from_list('custom greys', [((176)/(255),
                                                                              (196)/(255),
                                                                              (222)/(255)),
                                                                              (0,0,(139)/(255))], N=256)
    
    if out_folder[-1] != '/':
        out_folder = out_folder + '/'

    # if output folder does not exist, make it
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Use all core if thread_number is -1.
    if thread_number == -1:
        thread_number = mp.cpu_count()
    
    # Check if input file is passed.
    if maf_input_file == None and gct_input_file == None:
        print('Please indicate input file.')
        exit(1)
    
    # This part is ran if input file is maf file
    if maf_input_file != None:
        #read file
        print('Reading input file...')
        ds= pd.read_csv(maf_input_file, sep=file_separator, header=0, index_col=None, dtype=str)
        ds=ds.loc[:,['Hugo_Symbol','Variant_Classification', 'Tumor_Sample_Barcode', protein_change_identifier]]

        # This part is ran when mode is class or all
        if mode == 'class' or mode == 'all':
            # Dictionary for all data
            final_dict = {}
            
            print('start making gct by class')
            ds = ds[ds['Variant_Classification'].notna()]
            
            if sample_list == None:
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
            else:
                sample_set = set(sample_list)
                
            # Make dictionary for sample name and index combination and take subset
            sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            # Make list of count dictionary with each dictionary indicating one gene 
            print('make count dictionary')
            results = []
            for gene in gene_list:
                results.append(make_gene_dic_class(ds[ds['Hugo_Symbol'] == gene],gene,sample_dict))

            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe 
            genedfdic = {}
            for i in range(len(gene_list)):
                subgenedf = pd.DataFrame(columns = list(sample_set))
                for j in results[i].keys():
                    subgenedf.loc[j] = results[i][j]
                genedfdic[gene_list[i]] = subgenedf

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotclass(gct_output_file_prefix,key,genedf,figure_format,out_folder)

            # Put calculated results to one large dictionary
            for i in results:
                final_dict.update(i)

            # Create large dataframe to save.
            resultdf = pd.DataFrame(columns = list(sample_dict.keys()))
            for key, row in final_dict.items():
                resultdf.loc[key] = row

            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_class.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

        # This part is ran when mode is top or all
        if mode == 'freq' or mode == 'all':
            
            print('start making gct by top')
            
            # Keep only defined protein change
            ds = ds[ds[protein_change_identifier].notna()]
            
            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0)
            phenotype = phenotype.drop(columns=phenotype.columns[0])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            ds['Tumor_Sample_Barcode'].replace('-','_',regex=True,inplace=True)
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new
            
            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames
            
            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]
                
            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
                phenotype = phenotype[sample_list]
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
                phenotype = phenotype[sample_list]
            
            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)
                
            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            sample_set = phenotype.columns.tolist()
            
            # Make dictionary for sample name and index combination and take subset
            sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))
            subds = ds[ds['Tumor_Sample_Barcode'].isin(list(sample_set))]

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = subds['Hugo_Symbol'].unique().tolist()
            else:
                inte = [] 
                allgene = subds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            # Make list of count dictionary with each dictionary indicating one gene 
            print('make count dictionary')
            results = []
            for gene in gene_list:
                results.append(make_gene_dic_type(gene_ds = subds[subds['Hugo_Symbol'] == gene],gene = gene,sample_dict = sample_dict,protein_change_identifier = protein_change_identifier))

            # Create gene dataframe for each gene and put in dictioary
            genedfdic = {}
            for i in range(len(gene_list)):
                subgenedf = pd.DataFrame(columns = phenotype.columns)
                for j in results[i].keys():
                    subgenedf.loc[j] = results[i][j]
                genedfdic[gene_list[i]] = subgenedf

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotfreq(gct_output_file_prefix,frequency_threshold,phenotype,key,genedf,figure_format,
                                         directionn,out_folder)

            # Extract mutation by frequency threshold
            print('extract mutations by frequency')
            resultdflist = []
            for key, genedf in genedfdic.items():
                resultdflist.append(select_by_frequency(genedf = genedf, phenotype = phenotype, frequency_threshold = frequency_threshold, direction = direction))

            # Make big dataframe and put all result
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for subdf in resultdflist:
                subsubdf = subdf.iloc[:,:-2]
                name = subdf.index.tolist()[0].split('_')[0]
                currentseed = [0]*len(resultdf.columns)
                for i in subdf.index:
                    currentseed = seedCombine(currentseed,subdf.loc[i])
                resultdf.loc[name+'_frequency_threshold'+str(frequency_threshold)] = currentseed

            print('writing top result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_frequency_threshold_'+str(frequency_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'weight' or mode == 'all' or mode == 'weight_filter':
            print('start making gct by match')
            
            # Keep only defined protein change
            ds = ds[ds[protein_change_identifier].notna()]
            
            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0).drop(columns=['Description'])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            ds['Tumor_Sample_Barcode'].replace('-','_',regex=True,inplace=True)
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new
            
            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames

            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]
                
            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]
            
            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)
                
            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            sample_set = phenotype.columns.tolist()
            
            # Make dictionary for sample name and index combination and take subset
            sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))
            subds = ds[ds['Tumor_Sample_Barcode'].isin(list(sample_set))]

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = subds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = subds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            # Make list of count dictionary with each dictionary indicating one gene 
            print('make count dictionary')
            results = []
            for gene in gene_list:
                results.append(make_gene_dic_type(gene_ds = subds[subds['Hugo_Symbol'] == gene],gene = gene,sample_dict = sample_dict,protein_change_identifier = protein_change_identifier))

            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe
            genedfdic = {}
            for i in range(len(gene_list)):
                subgenedf = pd.DataFrame(columns = phenotype.columns)
                for j in results[i].keys():
                    subgenedf.loc[j] = results[i][j]
                genedfdic[gene_list[i]] = subgenedf

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotweight(gct_output_file_prefix,weight_threshold,phenotype,key,genedf,figure_format,
                                           direction,out_folder)

            #Remove gene with more than 40% are mutated
            for gene in gene_list:
                nummut = seedcomball(genedfdic[gene],genedfdic[gene].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*0.4:
                    del genedfdic[gene]

            # Extract mutation by weight threshold
            print('extract mutations by weight')
            resultdflist = []


            
            if mode == 'weight':
                for key, genedf in genedfdic.items():
                    resultdflist.append(select_by_weight(genedf = genedf, phenotype = phenotype, weight_threshold = weight_threshold, direction = direction))

            elif mode == 'weight_filter':
                for key, genedf in genedfdic.items():
                    resultdflist.append(select_by_weight_filter(genedf = genedf, phenotype = phenotype, weight_threshold = weight_threshold, direction = direction, ratio = ratio))
            

            # Make big dataframe and put all result
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for subdf in resultdflist:
                if len(subdf.index) != 0:
                    subsubdf = subdf.iloc[:,:-2]
                    name = subdf.index.tolist()[0].split('_')[0]
                    currentseed = [0]*len(resultdf.columns)
                    for i in subdf.index:
                        currentseed = seedCombine(currentseed,subdf.loc[i])
                    resultdf.loc[name+'_topmatch'+str(weight_threshold)] = currentseed

            print('writing match result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_topmatch_'+str(weight_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')
               
        if mode == 'allele' or mode == 'all':

            ds = ds.dropna(subset=[protein_change_identifier])  

            # Make list of sample and its unique index
            if sample_list == None:
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
            else:
                sample_set = set(sample_list)
                    
            # Make dictionary for sample name and index combination and take subset
            sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            allelelist = ds[protein_change_identifier].unique().tolist()

            counter = []
            for i in allelelist:
                counter.append(ds[protein_change_identifier].tolist().count(i))
            
            selectallele = []
            for i in range(len(counter)):
                if counter[i] >= frequency_threshold:
                    selectallele.append(allelelist[i])
            
            subds = ds[ds[protein_change_identifier].isin(selectallele)]

            # Make dictionary with each allele
            print('make count dictionary by allele')
            results = []
            for gene in gene_list:
                results.append(make_gene_dic_type(gene_ds = subds[subds['Hugo_Symbol'] == gene],gene = gene,sample_dict = sample_dict,protein_change_identifier = protein_change_identifier))

            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe
            genedfdic = {}
            for i in range(len(gene_list)):
                subgenedf = pd.DataFrame(columns = list(sample_set))
                for j in results[i].keys():
                    subgenedf.loc[j] = results[i][j]
                genedfdic[gene_list[i]] = subgenedf

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotclass(gct_output_file_prefix,key,genedf,figure_format,out_folder)

            resultdf = pd.concat(list(genedfdic.values()))

            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')


        if mode == 'comb':
            ds = ds[ds['Variant_Classification'].notna()]

            if sample_list == None:
                #Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
            else:
                sample_set = set(sample_list)
            
            # Make dictionary for sample name and index combination and take subset
            sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            results = []
            print('make count dictionary by class')
            for gene in gene_list:
                results.append(make_gene_dic_class(ds[ds['Hugo_Symbol'] == gene],gene,sample_dict))
            
            results2 = []
            print('make count dictionary by allele')
            for gene in gene_list:
                results2.append(make_gene_dic_type(gene_ds = ds[ds['Hugo_Symbol'] == gene],gene = gene,sample_dict = sample_dict,protein_change_identifier = protein_change_identifier))
            
            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe
            genedfdic = {}
            for i in range(len(gene_list)):
                subgenedf = pd.DataFrame(columns = list(sample_set))
                for j in results[i].keys():
                    subgenedf.loc[j] = results[i][j]
                for j in results2[i].keys():
                    subgenedf.loc[j] = results2[i][j]
                genedfdic[gene_list[i]] = subgenedf

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotclass(gct_output_file_prefix,key,genedf,figure_format,out_folder)

            resultdf = pd.concat(list(genedfdic.values()))

            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_all.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')
            
    else:
        ingct = pd.read_csv(gct_input_file,skiprows=[0,1],sep='\t',index_col=0)
        ingct = ingct.drop(columns=ingct.columns[0])
        
        if mode == 'class' or mode == 'all':
            subindex = []
            newclass_list = []
            
            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)
                
            # Check if end of index match any class, if match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)
                        
            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            resultdf = ingct.loc[subindex]
            
            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None
            
            # Write to file
            with open(gct_output_file_prefix + '_class.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')
                
        if mode == 'freq' or mode == 'all':
            print('freq')
            
            newclass_list = []
            subindex = []
            
            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)
            
            # Check if end of index match any class, if no match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)
                    
            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            ingct = ingct.loc[~ingct.index.isin(subindex)]
            
            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0)
            phenotype = phenotype.drop(columns=phenotype.columns[0])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new

            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames
            
            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]
                
            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]
            
            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)
                
            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            ingct = ingct[phenotype.columns.tolist()]
            
            #Remove row with 0 value
            rmrow = []
            for i in ingct.index.tolist():
                if sum(ingct.loc[i]) == 0:
                    rmrow.append(i)
            ingct = ingct.drop(index=rmrow)
            
            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            # Gene is taken as element before first class separator, may not work perfectly, has to restrict input format 
            genenamedic = {}
            if gene_list == None:
                for i in ingct.index.tolist():
                    genename = i[:i.find(class_seperator)]
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
            else:
                for i in ingct.index.tolist():
                    genename = i[:i.rfind(class_seperator)]
                    if genename not in gene_list:
                        continue
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
                
            # If no gene in dictionary, exit
            if len(genenamedic.keys()) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)
            
            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe
            genedfdic = {}
            for key, genenames in genenamedic.items():
                genedfdic[key] = ingct.loc[genenames]
            
            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotweight(gct_output_file_prefix,weight_threshold,phenotype,key,genedf,figure_format,
                                           direction,out_folder)

            #Remove gene with more than 40% are mutated
            for gene in list(genenamedic.keys()):
                nummut = seedcomball(genedfdic[gene],genedfdic[gene].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*0.4:
                    del genedfdic[gene]
            
            # Extract mutation by frequency threshold
            print('extract mutations by frequency')
            resultdflist = []
            for key, genedf in genedfdic.items():
                resultdflist.append(select_by_frequency(genedf = genedf, phenotype = phenotype, frequency_threshold = frequency_threshold, direction = direction))

            # Make big dataframe and put all result
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for subdf in resultdflist:
                subsubdf = subdf.iloc[:,:-2]
                name = subdf.index.tolist()[0].split('_')[0]
                currentseed = [0]*len(resultdf.columns)
                for i in subdf.index:
                    currentseed = seedCombine(currentseed,subdf.loc[i])
                resultdf.loc[name+'_frequency_threshold'+str(frequency_threshold)] = currentseed

            print('writing top result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_frequency_threshold_'+str(frequency_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'weight' or mode == 'all':
            print('weight')
            
            newclass_list = []
            subindex = []
            
            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)
            
            # Check if end of index match any class, if no match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)
                    
            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            ingct = ingct.loc[~ingct.index.isin(subindex)]
            
            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0)
            phenotype = phenotype.drop(columns=phenotype.columns[0])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new

            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames
            
            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]
                
            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]
            
            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)
            
            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            
            ingct = ingct[phenotype.columns.tolist()]
            
            #Remove row with 0 value
            rmrow = []
            for i in ingct.index.tolist():
                if sum(ingct.loc[i]) == 0:
                    rmrow.append(i)
            ingct = ingct.drop(index=rmrow)
            
            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            # Gene is taken as element before first class separator, may not work perfectly, has to restrict input format 
            genenamedic = {}
            if gene_list == None:
                for i in ingct.index.tolist():
                    
                    genename = i[:i.find(class_seperator)]
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
            else:
                for i in ingct.index.tolist():
                    genename = i[:i.rfind(class_seperator)]
                    if genename not in gene_list:
                        continue
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
                        
            # If no gene in dictionary, exit
            if len(genenamedic.keys()) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)
            
            # Make large dictionary combining all small dictionary to large dictionary with gene name and dataframe
            genedfdic = {}
            for key, genenames in genenamedic.items():
                genedfdic[key] = ingct.loc[genenames]
            
            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                for key, genedf in genedfdic.items():
                    newgenedf = plotweight(gct_output_file_prefix,weight_threshold,phenotype,key,genedf,figure_format,
                                           direction,out_folder)

            #Remove gene with more than 40% are mutated
            for gene in list(genenamedic.keys()):
                nummut = seedcomball(genedfdic[gene],genedfdic[gene].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*0.4:
                    del genedfdic[gene]
            
            # Extract mutation by weight threshold
            print('extract mutations by weight')
            resultdflist = []
            for key, genedf in genedfdic.items():
                resultdflist.append(select_by_weight(genedf = genedf, phenotype = phenotype, weight_threshold = weight_threshold, direction = direction))

            # Remove 0 length df
            resultdflist = [l for l in resultdflist if (len(l) != 0)]

            # Make big dataframe and put all result
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for subdf in resultdflist:
                subsubdf = subdf.iloc[:,:-2]
                name = subdf.index.tolist()[0].split('_')[0]
                currentseed = [0]*len(resultdf.columns)
                for i in subdf.index:
                    currentseed = seedCombine(currentseed,subdf.loc[i])
                resultdf.loc[name+'_topmatch'+str(weight_threshold)] = currentseed

            print('writing top result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_topmatch_'+str(weight_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')


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
    optional.add_argument("-gl", "--gene_list_file", dest="gene_list_file",
                        help="List of Gene to run. Default is all gene.", metavar="GENE_LIST_FILE")
    optional.add_argument("-nm", "--name_match", dest="name_match",
                        help="Indicate if sample name in phenotype file and input file are matching. Default is True.", metavar="NAME_MATCH")
    optional.add_argument("-mf", "--make_figure", dest="make_figure",
                        help="Indicate if figures are created. Recommended only when you run few genes. Default is False.", metavar="MAKE_FIG")
    optional.add_argument("-ff", "--figure_format", dest="figure_format",
                        help="Format for figure. Default is pdf.", metavar="FIGURE_FORMAT")
    optional.add_argument("-tn", "--thread_number", dest="thread_number",
                        help="Number of thread used to run the program. Default is 1, -1 indicate all thread used.", metavar="THREAD_NUMBER")
    optional.add_argument("-sl", "--sample_list", dest="sample_list",
                        help="List of sample to be used.", metavar="SAMPLE_LIST")

    args = parser.parse_args()
    args = vars(args)

    if args['input_file'][-3:] != 'gct' and args['input_file'][-3:] != 'maf':
        print('only maf file and gct file are accepted as input, please check your input file.')
        sys.exit(1)
    else:
        input_file = args['input_file']

    if args['mode'] == None:
        mode = 'class'
    elif (args['mode'] == 'class' or args['mode'] == 'freq' or args['mode'] == 'weight' or args['mode'] == 'weight_filter' or args['mode'] == 'comb') == False:
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

    if args['gene_list_file'] != None:
        with open(args['gene_list_file'],'r') as f:
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

    if args['thread_number'] != None:
        thread_number = int(args['thread_number'])
    else:
        thread_number = 1


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
            thread_number = thread_number,
            gene_list = gene_list,
            name_match = name_match,
            make_figure = make_figure,
            figure_format = figure_format,
            out_folder = out_folder,
            ratio = ratio,
            sample_list = sample_list)

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
            thread_number = thread_number,
            gene_list = gene_list,
            name_match = name_match,
            make_figure = make_figure,
            figure_format = figure_format,
            out_folder = out_folder,
            ratio = ratio,
            sample_list = sample_list)