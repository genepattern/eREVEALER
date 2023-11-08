# Download and install latest test version(which is version 0.0.158 now)
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ REVEALER==0.0.158

# Run Preprocess (Generate feature files)
REVEALER_preprocess -m weight \
					-i featureFiles/tcga_pancancer_082115.vep.filter_whitelisted.maf \
					-pi HGVSp_Short \
					-p TCGA_NFE2L2 \
					-o featureFiles/ \
					-wt 0.02 \
					-bm 0.5 \
					-k 30 \
					-pf featureFiles/TCGA_NFE2L2.gct \
					-pn NFE2L2.V2 \
					-nm False

#m: Mode to use, allele, weight, class, mutall, weight_filter, freq are availeble
#i: input maf file
#pi: protein identifier
#p: prefix for output files
#o: output folder
#wt: threshold for weight(IC)
#bm: bandwidth multiplier
#k: size of neighborhood for kernel
#pf: phenotype file, file that contain target (pathway enrichment, etc)
#pn: phenotype name, name of phenotype in file
#nm: If column name is matching. Set this to False if TCGA file is used.
#-gl: gene list

# Run REVEALER
REVEALER -tf featureFiles/TCGA_other_pathways.gct \
		 -f featureFiles/TCGA_Mut_All.gct \
		 -t CTGF_CYR61 \
		 -p TCGA_CTGF_CYR61_mutall_HNSC \
		 -o Results/TCGA_CTGF_CYR61_mutall_HNSC \
		 -lt 10 \
		 -ht 500 \
		 -mi 15 \
		 -tn 1 \
		 -a 1.5 \
		 -ip False \
		 -ib False \
		 -k 30 \
		 -gl featureFiles/allgeneLocus.txt \
		 -sf featureFiles/TCGA_mut_all_cna.gct \
		 -sn YAP1_AMP \
		 -tif featureFiles/TissueType_TCGA.gct \
		 -gmt featureFiles/TCGA_Mut_All.gmt \
		 -s featureFiles/TCGA_HNSC_list.txt

#tf: Target File (pathway enrichment value, protein enrichment value, etc.)
#f: Feature File (Mutations information)
#t: Target name (Name of target in target file)
#p: Prefix (Prefix of result files)
#o: Output (Output folder name)
#lt: Lower Threshold to remove features
#ht: Higher Threshold to remove features
#mi: Minimum iteration (Manually set number of iteration. If not provided, stop when IC decrease.)
#tn: Thread Number (It seems that 1 would work better)
#a: Alpha (power to raise target values)
#ip: If calculate P-values (Much faster with False)
#ib: If calculate variant with bootstrap (Much faster with False)
#k: Number of neighborhood for kernel
#gl: Gene location information
#sf: Seed File (Use this when seed if from different file. For example, use Mut_All for seed while features are allele level)
#sn: Seed Name (Name of seed in seed file. If multiple, separate by comma)
#tif: Tissue file (Indicate Tumor type of samples)
#gmt: allele information of each feature in gmt format
#s: subset of samples, use this if want to run only with specific cancer type