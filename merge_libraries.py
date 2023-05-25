import copy
import pandas as pd
import amsample as a
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import mmsample as m
from scipy.stats.kde import gaussian_kde



def pool_files(ams1, ams2, name, abbrev):
    ams3 = copy.deepcopy(ams1)
    ams3.name = name
    ams3.abbrev = abbrev
    no_chr = len(ams3.chr_names)
    for chrom in range(no_chr):
        ams3.no_c[chrom] = list(np.array(ams1.no_c[chrom]) + np.array(ams2.no_c[chrom]))
        ams3.no_t[chrom] = list(np.array(ams1.no_t[chrom]) + np.array(ams2.no_t[chrom]))
        no_ct = ams3.no_c[chrom] + ams3.no_t[chrom]
        ams3.diagnostics['effective_coverage'][chrom] = np.nanmean(no_ct)
        #ams3.p_filters = {'method':[], "max_coverage":[], 'max_TsPerCoverage':[],'max_g_to_a':[], 'max_a':[]}
    ams3.dump('filter')

def enrichment_testing(start, end, num_sim):
    length = end - start
    subs = pd.read_csv("/sci/labs/lirancarmel/krystal_castle/backup/alleles/chr1_archaics_wALT.tsv", sep="\t")
    subs.columns = ['Chrom', 'Pos', 'Ref', 'Archaic', 'Altai', 'Vin', 'Chag', 'Den']
    subs[['Alt1', 'Alt2']] = subs.Altai.str.split('/', expand=True)
    subs[['Vin1', 'Vin2']] = subs.Vin.str.split('/', expand=True)
    subs[['Chag1', 'Chag2']] = subs.Chag.str.split('/', expand=True)
    subs = subs[['Chrom', 'Pos', 'Ref', 'Archaic', 'Alt1', 'Alt2', 'Vin1', 'Vin2', 'Chag1', 'Chag2']]
    perfect_subs = subs[(subs['Ref'] != subs['Alt1']) &
                        (subs['Ref'] != subs['Alt2']) &
                        (subs['Ref'] != subs['Vin1']) &
                        (subs['Ref'] != subs['Vin2']) &
                        (subs['Ref'] != subs['Chag1']) &
                        (subs['Ref'] != subs['Chag2'])]
    partial_subs = subs[(subs['Ref'] != subs['Alt1']) |
                        (subs['Ref'] != subs['Alt2']) |
                        (subs['Ref'] != subs['Vin1']) |
                        (subs['Ref'] != subs['Vin2']) |
                        (subs['Ref'] != subs['Chag1']) |
                        (subs['Ref'] != subs['Chag2'])]
    full = sum((perfect_subs['Pos'] >= start) & (perfect_subs['Pos'] <= end))
    partial = sum((partial_subs['Pos'] >= start) & (partial_subs['Pos'] <= end))
    full_dist = list()
    partial_dist = list()
    starts = [random.randint(1,247249719) for i in range(num_sim)]
    ends = [i + length for i in starts]
    for i in range(num_sim):
        full_dist.append(sum((perfect_subs['Pos'] >= starts[i]) & (perfect_subs['Pos'] <= ends[i])))
        partial_dist.append(sum((partial_subs['Pos'] >= starts[i]) & (partial_subs['Pos'] <= ends[i])))
        i += 1
    print("Number of fixed differences: " + str(full))
    print("p value of fixed differences: " + str(sum(np.array(full_dist) > full)/num_sim))
    print("Number of partial differences: " + str(partial))
    print("p value of partial differences: " + str(sum(np.array(partial_dist) > partial) / num_sim))
    print('done')

def mean_meth(filename):
    print("Evaluating " + filename)
    ams = a.Amsample()
    ams.parse_infile(filename)
    result = np.nanmean(ams.methylation['methylation'][0])
    print("Mean methylation of chr1 is " + str(result))

def meth_density(df, title, norm=False):
    df['cov'] = df['no_c'] + df['no_t']
    df = df[(df['cov'] > 0)]
    df = df[~np.isnan(df['meth'])]
    df = df[~np.isnan(df['modern_meth'])]
    X = df['modern_meth']
    Y = df['meth']
    Z, xedges, yedges = np.histogram2d(X, Y, bins=20)
    if norm:
        Z = np.log2(Z+0.001)
        title = title + " log2 Norm"
    plt.pcolormesh(xedges, yedges, Z.T)
    plt.xlabel('Modern Methylation')
    plt.ylabel("Reconstructed Methylation")
    plt.title(title)
    plt.show()

    # k = gaussian_kde(np.vstack([x, y]))
    # xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))

def meth_kde(filename, title, path="/sci/labs/lirancarmel/krystal_castle/backup/python_dumps/"):
    df = pd.read_csv(path + filename + '.csv')
    sns.kdeplot(data=df, x='modern_meth', y='meth', color='red', shade=True)
    plt.title(title)
    plt.xlabel("Modern Methylation")
    plt.ylabel("Reconstructed Methylation")
    plt.savefig(title + '.png')
    plt.clf()

def smooth_meth_kde(mms, ams, title):
    modern_meth, weights = mms.smooth(0, 31)
    ancient_meth = ams.methylation['methylation'][0]
    sns.kdeplot(x=modern_meth, y=ancient_meth, color='red', shade=True)
    plt.title(title)
    plt.xlabel("Modern Methylation")
    plt.ylabel("Reconstructed Methylation")
    plt.savefig(title + '.png')
    plt.clf()

if __name__ == '__main__':
    VnU_path = "/sci/labs/lirancarmel/krystal_castle/backup/python_dumps/Vin_nonUDG_Chr1_filter.txt"
    VU_path = "/sci/labs/lirancarmel/krystal_castle/backup/python_dumps/Vin_UDG_Chr1_filter.txt"
    name = "Vindija_UDG_nonUDG_pooled"
    abbrev = "Pooled Libraries Vindija UDG and nonUDG"
    ams1 = a.Amsample()
    ams1.parse_infile(VnU_path)
    ams2 = a.Amsample()
    ams2.parse_infile(VU_path)
    pool_files(ams1, ams2, name, abbrev)

    # enrichment_testing(160989353, 161008685, 1000)
    # enrichment_testing(21829151, 21883389, 1000)
    # enrichment_testing(101701385, 101705718, 1000)
    # enrichment_testing(119525150, 119527157, 1000)

    # print("TBX15")
    # enrichment_testing(119425666, 119532179, 1000)
    # print("ALPL")
    # enrichment_testing(21835858, 21904905, 1000)
    # print("FLLR")
    # enrichment_testing(160965001, 161008784, 1000)
    # print("RP4-575N6.4")
    # enrichment_testing(101701239, 101702084, 1000)

    # print("SPSB1")
    # enrichment_testing(9352939, 9429591, 1000)
    # enrichment_testing(9358001, 9463166, 1000)
    # print("PDPN")
    # enrichment_testing(13909960, 13944452, 1000)
    # enrichment_testing(13914615, 13987498, 1000)
    # print("ECE1")
    # enrichment_testing(21543740, 21671997, 1000)
    # enrichment_testing(21617442, 21705303, 1000)
    # print("BMP8A")
    # enrichment_testing(39957318, 39991607 , 1000)
    # enrichment_testing(39966137, 40039632, 1000)
    # print("TGFBR3")
    # enrichment_testing(92145902, 92371892, 1000)
    #enrichment_testing(92244724, 92376546, 1000)








    # mean_meth("/sci/labs/lirancarmel/krystal_castle/backup/dmrs/samples/VU_Log_ref.txt")

    
    #meth_kde("Chag_EM31_DDLog_Chr1_meth", "Chagyrskaya Log EM")


    # modern_infile = "bone5.txt"
    # mms = m.Mmsample()
    # mms.create_mms_from_text_file()
    # mms.merge()
    #
    # path = "/sci/labs/lirancarmel/krystal_castle/backup/python_dumps/"
    # samples = [ #"Chag_Hist_Chr1_meth", "Altai_Chr1_Log_meth", "Altai_Hist_Chr1_meth", "Ust_Ishim_Chr1_Hist_meth",
    #            "Ust_Ishim_Chr1_Log_meth", "Vindija_UDG_Hist_Chr1_meth", "Vindija_UDG_Log_Chr1_meth",
    #            "Vindija_UDG_nonUDG_pooled_hist_meth", "Vindija_UDG_nonUDG_pooled_log_meth", "Vin_Merge_Chr1_Dual_Deam_hand_calculated_meth",
    #            "Vin_Merge_Chr1_EM31_DDLog_meth", "Vin_Merge_Chr1_Hist_meth", "nonUDG_Vin_Chag_pooled_hist_meth", "nonUDG_Vin_Chag_pooled_log_meth",
    #            "Vin_UDG_Chr1_log_EM_meth", "Vindija_UDG_nonUDG_pooled_log_EM_meth", "nonUDG_Vin_Chag_pooled_log_EM_meth",
    #            "Altai_Chr1_Log_EM_meth", "Ust_Ishim_Chr1_Log_EM_meth", "Chag_Dual_Deam_Log_meth"]
    # titles = [ #"Chagyrskaya Hist", "Altai Log ref", "Altai Hist", "Ust Ishim Hist",
    #           "Ust Ishim Log ref", "Vindija UDG Hist", "Vindija UDG Log ref",
    #           "Vin pooled Hist", "Vin pooled Log ref", "Vindija nonUDG Log ref", "Vindija nonUDG Log EM",
    #           "Vindija nonUDG Hist", "nonUDG pooled Hist", "nonUDG pooled Log ref", "Vindija UDG Log EM",
    #           "Vin pooled Log EM", "nonUDG pooled Log EM", "Altai Log EM", "Ust Ishim Log EM", "Chagyrskaya Log ref"]
    #
    #
    # for i in range(len(samples)):
    #     infile = path + samples[i] + '.txt'
    #     ams = a.Amsample()
    #     ams.parse_infile(infile)
    #     smooth_meth_kde(mms, ams, titles[i])
