#-----------------------------------------------#
# Cluster electrodes with layer and R over lags #
#-----------------------------------------------#
from brain_color_prepro import get_e_list, get_n_layers
from podenc_plots import extract_single_correlation

from sktime.distances.elastic_cython import dtw_distance
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram, fcluster
import numpy as np
import glob
import matplotlib.pyplot as plt

# see ML/COS best practices for explanation
def hierarchical_clustering(dist_mat, method = 'complete'):
    if method == 'complete':
        Z = complete(dist_mat)
    elif method == 'single':
        Z = single(dist_mat)
    elif method == 'average':
        Z = average(dist_mat)
    elif method == 'ward':
        Z = ward(dist_mat)

    fig = plt.figure(figsize=(16,8))
    dn = dendrogram(Z)
    plt.title("Dendrogram for " + method + "-linkage")
    fig.savefig("Hierarchical_Clustering_" + method + "-linkage.png")
    return Z

if __name__ == '__main__':
    # use extract correlations to get R
    # run for every layer for each electrode
    # results is list of 160 electrodes, each with an item of size (num lags x num_layers)
    # num lags --> 161
    # = (161 x 48) for gpt2 xl for example --> expected that time be first for dtw_distance, data dimension second

    in_type = 'gpt2-xl-hs'
    num_lags = 161
    num_layers = get_n_layers(in_type[:-3])
    e_list = get_e_list('brain_map_input.txt')
    data_list = []
    #breakpoint()
    for e in e_list:
        if e == '742_G64':
            continue
        e_array = np.zeros((num_lags, num_layers), dtype='float64')
        for l in range(num_layers):
            ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-gpt2-xl-hs' + str(l) + '/*')
            e_array[:,l] = extract_single_correlation(ldir, e)
        data_list.append(e_array)
    # build distance matrix
    # option 2: dtw
    # num series in this case is number of (lags, layers) items (which = num electrodes)
    dist_type = 'dtw'
    num_series = len(data_list)
    print('num electrodes', num_series)
    distance_mat = np.zeros((num_series, num_series))
    for i in range(num_series):
        for j in range(num_series):
            #breakpoint()
            if i != j:
                if dist_type == 'dtw':
                    distance_mat[i,j] = dtw_distance(data_list[i], data_list[j])
                # TODO: fix this so it works for 2D mats. (maybe just convert to list)
                elif dist_type == 'R':
                    distance_mat[i,j] =  pearsonr(data_list[i], data_list[j])

    #breakpoint()        
    # run clustering on this
    #method = 'single'
    #method = 'complete'
    #method = 'ward'
    method = 'average'
    link_matrix = hierarchical_clustering(distance_mat, method)
    
    # get clusters
    # option 1 --> you want 4 clusters. 
    cluster_labels = fcluster(link_matrix, 4, criterion='maxclust')
    print(np.unique(cluster_labels))
    
    # option 2: regulate clusters by cophenetic distance
    # 600 --> height of dendrogram at which two observations trees are joined to form one cluster < 600 
    cluster_labels = fcluster(link_matrix, 600, criterion='distance')
    print(np.unique(cluster_labels))
    # save outputs so you can plot on the brain
    # TODO


