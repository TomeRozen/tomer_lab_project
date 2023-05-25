import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysam
import pybedtools as pbt
import amsample as a
import mmsample as m
import numpy as np
import tools as t
import matplotlib.pyplot as plt
import pandas as pd
import pybedtools as pbt
import gintervals as gint
import gcoordinates as gcoord
from scipy.stats import mannwhitneyu
import seaborn as sns

def load_gc():
    infile = "/sci/home/krystal_castle/workbench/roam-python/cpg_coords"
    return t.load_object(infile)


def calc_meth_in_cgi(ams,gc):
    """
    Compares methylation values in cgis and in all positions
    :param ams: amsample object
    :param gc: object mapping every cpg to its position in hg19 reference genome
    :return: None
    """
    path = "/sci/home/krystal_castle/workbench/roam-python/cpgIslandExt.txt"
    cgi_data = pd.read_csv(path, sep="\t", header=None)
    cgi_data.columns = ["bin", "chrom", "chromStart", "chromEnd", "name", "length", "cpgNum", "gcNum", "perCpg",
                        "perGc", "obsExp"]

    # filter out cgi not in chrom1
    cgi_data = cgi_data[cgi_data["chrom"] == "chr1"]

    methylation_values = np.array(ams.methylation["methylation"][0])
    cgi_methylation_list = []
    non_cgi_methylation_list = []
    for i in range(1, len(cgi_data)):
        prev_end = cgi_data["chromEnd"][i-1] - 1
        idx_start = cgi_data["chromStart"][i] + 1
        idx_end = cgi_data["chromEnd"][i] - 1
        cpg_idx_prev_end = np.where(gc == prev_end)[0][0]
        cpg_idx_start = np.where(gc == idx_start)[0][0]
        cpg_idx_end = np.where(gc == idx_end)[0][0]
        meth_in_cgi = np.array(methylation_values[cpg_idx_start:cpg_idx_end])
        meth_btwn_cgi = np.array(methylation_values[cpg_idx_prev_end:cpg_idx_start])
        cgi_methylation_list.append(meth_in_cgi)
        non_cgi_methylation_list.append(meth_btwn_cgi)

    cgi_mean = np.array([np.nanmean(cgi_methylation_list[i]) for i in range(len(cgi_methylation_list))])
    non_cgi_mean = np.array([np.nanmean(non_cgi_methylation_list[i]) for i in range(len(non_cgi_methylation_list))])
    data = [cgi_mean[~np.isnan(cgi_mean)], non_cgi_mean[~np.isnan(non_cgi_mean)]]
    u_test_res = mannwhitneyu(data[0], data[1])
    plt.boxplot(data)
    plt.xticks(ticks=[1, 2], labels=["cgi", "non-cgi"])
    plt.ylabel("Methylation")
    plt.title("CGI methylation | "+ams.name+" | p = " + f'{u_test_res.pvalue:.3e}')
    plt.show()
    print("Mean of CpG Islands: " + str(np.nanmean(cgi_mean)) + ", Mean of non-CpG Islands: " + str(np.nanmean(non_cgi_mean)))
    pass


def calc_meth_in_hk(ams,gc):
    """
    Compares methylation values in house keeping genes and in all positions
    :param ams: amsample object
    :param gc: object mapping every cpg to its position in hg19 reference genome
    :return: None
    """
    genes = get_genes_pbt()

    # include only intersection in merging
    genes_no_dups = genes.groupby(g=[1, 2, 3, 6], c='4,5', o='distinct').cut([0, 1, 2, 4, 5, 3])  # is stream nec?

    # definition of promoter length
    before = 5000
    after = 1000
    chrom = 0

    # create bedtools object containing every gene promoter in the chromosome
    proms = gint.Gintervals(chr_names=[chrom])
    proms.calc_prom_coords(genes_no_dups, before, after)
    prom_string = ""
    for prom in range(len(proms.start[chrom])):
        prom_string += f"\n{proms.chr_names[chrom]} {proms.start[chrom][prom]} {proms.end[chrom][prom]} {proms.iname[chrom][prom]} 0 {proms.strand[chrom][prom]}"
    chrom_proms = pbt.BedTool(prom_string, from_string=True)

    # create bedtools object for cpgs
    #gc_string = ""
    # for i in range(len(gc.coords[chrom])):
    #     gc_string += f"\n{0} {gc.coords[chrom][i]} {gc.coords[chrom][i]+1}"
    # chrom_gc = pbt.BedTool(gc_string, from_string=True)
    # chrom_gc.saveas("/vol/sci/bio/data/liran.carmel/arielleb/project_files/additional_files/gc_string.bed")
    chrom_gc = pbt.BedTool("/sci/home/krystal_castle/workbench/roam-python/gc_string.bed")

    # find all the cpgs in house keeping genes and their genes
    intersects1 = chrom_proms.intersect(chrom_gc)
    #intersects1.saveas("/sci/home/krystal_castle/workbench/roam-python/prom_int_gc.bed")
    intersect_df = intersects1.to_dataframe()

    #Find all the cpgs not in housekeeping genes
    subtract1 = chrom_gc.subtract(chrom_proms)
    subtract_df = subtract1.to_dataframe()

    # get metylation values of file
    methylation_values = np.array(ams.methylation["methylation"][0])
    hk_methylation_list = []
    i = 0

    # calculate methylation list per every gene promoter
    while True:
        cur_gene = intersect_df["name"][i]
        idx_start = intersect_df["start"][i] # idx of first cpg in promoter
        while i < len(intersect_df) and intersect_df["name"][i] == cur_gene:
            i += 1
        idx_end = intersect_df["start"][i - 1] # idx of last cpg in promoter

        # convert to idx of the gc list
        cpg_idx_start = np.where(gc.coords[chrom] == idx_start)[0][0]
        cpg_idx_end = np.where(gc.coords[chrom] == idx_end)[0][0]

        # get methylation list
        meth_in_cgi = np.array(methylation_values[cpg_idx_start:cpg_idx_end])
        hk_methylation_list.append(meth_in_cgi)
        if i == len(intersect_df):
            break


    # calculate methylation list for non-promoter cpgs
    # create dataframe with non-promoter cpgs and their index
    cpgs = pd.DataFrame(data=gc.coords[0])
    cpgs['idx'] = range(1,len(cpgs)+1)
    cpgs.columns = ["start","idx"]
    merged1 = subtract_df.merge(cpgs, how='left', on='start')
    # add methylation values for relevant indices to dataframe
    meth_df = pd.DataFrame(data=methylation_values)
    meth_df['idx'] = range(1, len(meth_df)+1)
    meth_df.columns = ['methylation', 'idx']
    merged2 = merged1.merge(meth_df, how='left', on='idx')
    # create list containing only methylation values for relavent indices
    non_hk_methylation_list = np.array(merged2['methylation'])


    # calculate mean methylation per promoter
    hk_mean = np.array([np.nanmean(hk_methylation_list[i]) for i in range(len(hk_methylation_list))])

    # data of promoters and data of all cpgs in chromosome
    data = [hk_mean[~np.isnan(hk_mean)], non_hk_methylation_list[~np.isnan(non_hk_methylation_list)]]
    u_test_res = mannwhitneyu(data[0], data[1])
    plt.boxplot(data)
    plt.xticks(ticks=[1, 2], labels=["hk gene", "non-hk gene"])
    print(round(u_test_res.pvalue,2))
    plt.ylabel("Methylation")
    plt.title(f'HK promoter methylation | '+ams.name+' | p = ' + f'{u_test_res.pvalue:.3e}')
    plt.show()
    plt.savefig("hk_genes.png")
    print(np.nanmean(hk_mean))
    pass


def get_genes_pbt():
    """
    read a table of house keeping genes and a table of all genes with start and end positions, merge tables to get
    start and end positions of hk genes and create a bed file
    :return: house keeping genes positions bedfile
    """
    path = "/sci/home/krystal_castle/workbench/roam-python/UCSC_Genes_sorted.txt"
    path2 = "/sci/home/krystal_castle/workbench/roam-python/HK_genes.txt"
    all_genes_table = pd.read_csv(path, sep="\t", header=None)
    hk_genes_table = pd.read_csv(path2, sep="\t", header=None)
    all_genes_table.columns = ["chrom", "chromStart", "chromEnd", 'name', "code", "strand"]
    hk_genes_table.columns = ['name', "accession"]
    all_genes_table['name'] = all_genes_table['name'].astype(str)
    hk_genes_table['name'] = hk_genes_table['name'].astype(str)
    hk_genes_table['name'] = hk_genes_table['name'].str.replace(' ', '')
    merged = all_genes_table.merge(hk_genes_table, on="name", how="inner")  # 13562 rows
    genes = pbt.BedTool.from_dataframe(merged)
    return genes


####New Code from Arielle####
def calc_meth_in_cgi_noplot(meth_list,gc):
    """
    Compares methylation values in cgis and in all positions
    :param ams: amsample object
    :param gc: object mapping every cpg to its position in hg19 reference genome
    :return: None
    """
    path = "/sci/home/krystal_castle/workbench/roam-python/cpgIslandExt.txt"
    cgi_data = pd.read_csv(path, sep="\t", header=None)
    cgi_data.columns = ["bin", "chrom", "chromStart", "chromEnd", "name", "length", "cpgNum", "gcNum", "perCpg",
                        "perGc", "obsExp"]

    # filter out cgi not in chrom1
    cgi_data = cgi_data[cgi_data["chrom"] == "chr1"]

    methylation_values = np.array(meth_list) #np.array(ams.methylation[0])
    # methylation_list = np.array([])
    methylation_list = []
    for i in range(len(cgi_data)):
        idx_start = cgi_data["chromStart"][i] + 1
        idx_end = cgi_data["chromEnd"][i] - 1
        cpg_idx_start = np.where(gc == idx_start)[0][0]
        cpg_idx_end = np.where(gc == idx_end)[0][0]
        meth_in_cgi = np.array(methylation_values[cpg_idx_start:cpg_idx_end])
        # methylation_list = np.concatenate((methylation_list, meth_in_cgi), axis=None)
        methylation_list.append(meth_in_cgi)

    methylation_list = np.array([np.nanmean(methylation_list[i]) for i in range(len(methylation_list))])
    # methylation_list = np.array(methylation_list)
    return methylation_list, methylation_values




def calc_meth_in_hk_noplot(meth_list,gc):
    """
    Compares methylation values in house keeping genes and in all positions
    :param ams: amsample object
    :param gc: object mapping every cpg to its position in hg19 reference genome
    :return: None
    """
    genes = get_genes_pbt()

    # include only intersection in merging
    genes_no_dups = genes.groupby(g=[1, 2, 3, 6], c='4,5', o='distinct').cut([0, 1, 2, 4, 5, 3])  # is stream nec?

    # definition of promoter length
    before = 5000
    after = 1000
    chrom = 0

    # create bedtools object containing every gene promoter in the chromosome
    proms = gint.Gintervals(chr_names=[chrom])
    proms.calc_prom_coords(genes_no_dups, before, after)
    prom_string = ""
    for prom in range(len(proms.start[chrom])):
        prom_string += f"\n{proms.chr_names[chrom]} {proms.start[chrom][prom]} {proms.end[chrom][prom]} {proms.iname[chrom][prom]} 0 {proms.strand[chrom][prom]}"
    chrom_proms = pbt.BedTool(prom_string, from_string=True)

    # create bedtools object for cpgs
    gc_string = ""
    # for i in range(len(gc.coords[chrom])):
    #     gc_string += f"\n{0} {gc.coords[chrom][i]} {gc.coords[chrom][i]+1}"
    # chrom_gc = pbt.BedTool(gc_string, from_string=True)
    # chrom_gc.saveas("/sci/labs/lirancarmel/arielleb/icore-data/project_files/additional_files/gc_string.bed")
    chrom_gc = pbt.BedTool("/sci/home/krystal_castle/workbench/roam-python/gc_string.bed")

    # find all the cpgs in house keeping genes and their genes
    intersects1 = chrom_proms.intersect(chrom_gc)
    #intersects1.saveas("/sci/labs/lirancarmel/arielleb/icore-data/project_files/additional_files/prom_int_gc.bed")
    intersect_df = intersects1.to_dataframe()

    # get methylation values of file
    methylation_values = np.array(meth_list)
    # methylation_list = np.array([])
    methylation_list = []
    i = 0

    #to generate data for R:
    cpg_idxs = pd.DataFrame.from_dict({'gene':[], 'start':[], 'end':[]})

    # calculate methylation list per every gene promoter
    while True:
        cur_gene = intersect_df["name"][i]
        idx_start = intersect_df["start"][i] # idx of first cpg in promoter
        while i < len(intersect_df) and intersect_df["name"][i] == cur_gene:
            i += 1
        idx_end = intersect_df["start"][i - 1] # idx of last cpg in promoter

        # convert to idx of the gc list
        cpg_idx_start = np.where(gc.coords[chrom] == idx_start)[0][0]
        cpg_idx_end = np.where(gc.coords[chrom] == idx_end)[0][0]
        cpg_idxs = cpg_idxs.append({'gene':cur_gene, 'start':cpg_idx_start,'end':cpg_idx_end}, ignore_index=True)

        # get methylation list
        meth_in_hk = np.array(methylation_values[cpg_idx_start:cpg_idx_end])
        # methylation_list = np.concatenate((methylation_list, meth_in_hk), axis=None)
        methylation_list.append(meth_in_hk)
        if i == len(intersect_df):
            break

    cpg_idxs.to_csv('/sci/home/krystal_castle/workbench/roam-python/hk_border_cpgs.csv')
    methylation_list = np.array([np.nanmean(methylation_list[i]) for i in range(len(methylation_list))])
    # methylation_list = np.array(methylation_list)
    return methylation_list, methylation_values
    # # data of promoters and data of all cpgs in chromosome
    # data = [methylation_list[~np.isnan(methylation_list)], methylation_values[~np.isnan(methylation_values)]]
    # u_test_res = mannwhitneyu(data[0], data[1])
    # plt.boxplot(data)
    # plt.xticks(ticks=[1, 2], labels=["hk gene", "total"])
    # print(round(u_test_res.pvalue,2))
    # plt.title(f'HK genes promoters methylation | Bell Beakers | pvalue = {u_test_res.pvalue:.3e}')#+ams.abbrev)
    #
    # plt.savefig("/sci/labs/lirancarmel/arielleb/icore-data/project_files/plots/bbkers_hk_meth.png")
    # plt.show()


def plot_boxplot(in_cgi_df,in_hk_df, all_df, filename):
    in_cgi_df = in_cgi_df.assign(Positions = "CGI")
    in_hk_df = in_hk_df.assign(Positions = "HK genes promoters")
    all_df = all_df.assign(Positions = "All")
    cdf = pd.concat([in_cgi_df, in_hk_df,all_df])
    mdf = pd.melt(cdf, id_vars=['Positions'], var_name=['Population'])
    ax = sns.boxplot(x="Positions", y="value", hue="Population", data=mdf)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(filename)
    plt.show()

def plot_violinplot(in_cgi_df,in_hk_df, all_df, filename):
    in_cgi_df = in_cgi_df.assign(Positions = "CGI")
    in_hk_df = in_hk_df.assign(Positions = "HK genes promoters")
    all_df = all_df.assign(Positions = "All")
    cdf = pd.concat([in_cgi_df, in_hk_df,all_df])
    mdf = pd.melt(cdf, id_vars=['Positions'], var_name=['Population'])
    ax = sns.violinplot(x="Positions", y="value", hue="Population", data=mdf)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':

    text_infile1 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/nU_pool_Log_EM.txt"
    text_infile2 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Chag_MLE_EM.txt"
    text_infile3 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VnU_MLE_EM.txt"
    text_infile4 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VU_Log_EM.txt"
    text_infile5 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Alt_Log_EM.txt"
    text_infile6 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Ust_Log_EM.txt"

    # text_infile1 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/nU_pool_Log_ref.txt"
    # text_infile2 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Chag_MLE_ref.txt"
    # text_infile3 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VnU_MLE_ref.txt"
    # text_infile4 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VU_Log_ref.txt"
    # text_infile5 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Alt_Log.txt"
    # text_infile6 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Ust_Log.txt"

    # text_infile1 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/nU_pool_Hist.txt"
    # text_infile2 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Chag_Hist.txt"
    # text_infile3 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VnU_Hist.txt"
    # text_infile4 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VU_Hist.txt"
    # text_infile5 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Alt_Hist.txt"
    # text_infile6 = "/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/Ust_Hist.txt"

    gc = load_gc()
    mms = m.Mmsample()
    mms.create_mms_from_text_file()
    mms.merge()
    mms.scale()
    chr1_mm = np.array(mms.methylation[0])

    text_infiles = [text_infile1, text_infile2, text_infile3, text_infile4, text_infile5, text_infile6]
    NAMES = ["nUpool MLE EM", "Chag MLE EM",  "VnU MLE EM", "VU MLE EM", "Alt MLE EM", "Ust MLE EM"]
    #NAMES = ["nUpool Hist", "Chag Hist", "VnU Hist", "VU Hist", "Alt Hist", "Ust Hist"]
    in_cgi_df = pd.DataFrame(columns=NAMES)
    all_df = pd.DataFrame(columns=NAMES)
    in_hk_df = pd.DataFrame(columns=NAMES)
    for i in range(len(text_infiles)):
        text_infile = text_infiles[i]
        NAME = NAMES[i]
        ams = a.Amsample()
        ams.parse_infile(text_infile)
        meth_list = ams.methylation['methylation'][0]
        #in_cgi, all = calc_meth_in_cgi_noplot(meth_list, gc.coords[0])
        in_hk, _ = calc_meth_in_hk_noplot(meth_list, gc)
        in_cgi_df.loc[:,NAME] = in_cgi
        all_df.loc[:,NAME] = all
        in_hk_df.loc[:,NAME] = in_hk
    #
    in_cgi, all = calc_meth_in_cgi_noplot(chr1_mm, gc.coords[0])
    in_hk, _ = calc_meth_in_hk_noplot(chr1_mm, gc)
    in_cgi_df.loc[:,"Bone5"] = in_cgi
    all_df.loc[:,"Bone5"] = all
    in_hk_df.loc[:,"Bone5"] = in_hk
    plot_boxplot(in_cgi_df, in_hk_df, all_df, "MLE_EM_boxplot.png")
    plot_violinplot(in_cgi_df, in_hk_df,all_df, "MLE_EM_violinplot.png")
    #plt.savefig("/sci/labs/lirancarmel/arielleb/icore-data/project_files/plots/cgi_hk_all.png")



#if __name__ == '__main__':
#    text_infile = "/sci/labs/lirancarmel/krystal_castle/backup/python_dumps/Ust_Ishim_chr1_meth.txt"

#    ams = a.Amsample()
#    ams.parse_infile(text_infile)
#    gc = load_gc()

#    calc_meth_in_cgi(ams,gc.coords[0])
#    genes = get_genes_pbt()
#    calc_meth_in_hk(ams, gc)
#     genes = get_genes_pbt()
#     genes_no_dups = genes.groupby(g=[1, 2, 3, 6], c='4,5', o='distinct').cut([0, 1, 2, 4, 5, 3])
#     result = pd.DataFrame(genes_no_dups)
#     result.to_csv('/sci/home/krystal_castle/workbench/roam-python/hk_start_stop.csv')

