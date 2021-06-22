#------------------------------------------------#
# Prepares text file input for matlab brain plot #
# Author: Eric Ham                               #
#------------------------------------------------#

import os
import pandas as pd
import numpy as np
import matplotlib.colors as clr
from statsmodels.stats.multitest import fdrcorrection as fdr
import argparse
import math

# get number of layers from the model type
def get_n_layers(mt):
    if mt == 'gpt2-xl':
        nl = 48
    elif mt == 'gpt2':
        nl = 12
    elif mt == 'gpt2-large':
        nl = 36
    return nl

# get all electrodes to look through
def get_e_list(in_f, split_token):
    e_list = []
    sep = '_'
    for line in open(in_f, 'r'):
        items = line.strip('\n').split(split_token) # '\t' or ',' depending on input
        if items[0] == '742' and items[1] == 'G64':
            continue
        e_list.append(sep.join(items[:2]))    
    return e_list

# get the top layer for each electrode
def get_e2l(e_list, num_layers, model_type, lag, threshold):
    path_s = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + str(model_type) + '-hs'
    # go through all electrodes
    print(lag)
    e2l  = {}
    for e in e_list:
        print(e)
        l_corrs = []
        p_vals = []
        for l in range(num_layers):
            e_dir = path_s + str(l) + '/777/' + e + '_comp.csv'
            #e_sig = pd.read_csv(e_dir + e + '_comp.csv')
            #with open(e_dir, 'r') as e_f:
            #    e_sig = list(map(float, e_f.readline().strip().split(',')))
            e_sig = pd.read_csv(e_dir)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(e_dir, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            #breakpoint()
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            l_corrs.append(e_Rs[len(e_Rs)//2 + lag])
            p_vals.append(e_Qs[len(e_Ps)//2 + lag])
        # if p value low enought, then significant.
        max_l = np.argmax(l_corrs)
        #breakpoint()
        if p_vals[max_l]  <= threshold:
            e2l[e] = max_l
        else:
            #breakpoint()
            e2l[e] = -1 # this means not significant correlation

    return e2l

# get layer --> color dictionary
def get_dicts(num_gs, num_layers):
    nl_pg = int(num_layers/num_layers) #make >1 for grouping layers
    layers = list(range(num_layers))
    possible_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #g2c = {} # group to color
    #l2g = {} # layer to group
    init_grey = 1
    l2c = {}
    for i in range(num_gs):
        init_grey -= 1/(math.exp(i*0.0001)*(num_gs + 1))
        #g2c[i] = clr.to_rgb(possible_colors[i])
        #l2g.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.ones(nl_pg)*i)))    
        #l2c.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.repeat(np.array([clr.to_rgb(possible_colors[i])]),nl_pg ,axis=0)))) 
        #breakpoint()
        l2c.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.repeat(np.array([[init_grey, 0.0, 0.0]]),nl_pg ,axis=0)))) 
    #return l2g, g2c
    # set white as color for not significant electrodes
    l2c[-1] = clr.to_rgb('w')
    return l2c

# get electrode --> color dictionary
def get_e2c(e2l, l2c):
    e2c = {}
    for e, l in e2l.items():
        e2c[e] = l2c[l]
    #breakpoint()
    return e2c
 
def save_e2c(e2c, in_f, lag, model_type):
    out_f = open('brain_map_text_out_lag' + str(lag) + '_' + model_type + '_Qsig.txt', 'w')
    sep = '\t'
    for line in open(in_f, 'r'):
        items = line.strip('\n').split('\t')
        print(sep.join(items[1:]))
        #breakpoint()
        if items[0] == '742' and items[1] == 'G64':
            out_f.write(sep.join(items[1:]) + '\t0.0 0.0 0.0\n')
        else:
            #breakpoint()
            out_f.write(sep.join(items[1:]) + '\t' + sep.join(list(map(str,e2c['_'.join(items[:2])]))) + '\n')
    
    out_f.close()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lag', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # load file
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    e_list = get_e_list(in_f, '\t')
    # get top layer for each electrode
    model_type = 'gpt2'
    num_layers = get_n_layers(model_type)
    lag = args.lag
    adjusted_lag = int(lag/25) #this adjusts for binning  
    threshold = 0.01
    e2l = get_e2l(e_list, num_layers, model_type, adjusted_lag, threshold)
    # divide layers into groups and assign colors 
    #num_gs = 4 # up to 7 (see line below) num groups
    num_gs = num_layers
    #layer2group, group2color = get_dicts(num_gs, num_layers)    
    layer2color = get_dicts(num_gs, num_layers)
    # assign color to electrode
    e2c = get_e2c(e2l, layer2color)    
    # save output    
    
    #breakpoint()
    save_e2c(e2c, in_f, lag, model_type)

