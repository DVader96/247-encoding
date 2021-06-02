#------------------------------------------------#
# Prepares text file input for matlab brain plot #
# Author: Eric Ham                               #
#------------------------------------------------#

import os
import pandas as pd
import numpy as np
import matplotlib.colors as clr

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
def get_e_list(in_f):
    e_list = []
    sep = '_'
    for line in open(in_f, 'r'):
        items = line.split('\t')
        e_list.append(sep.join(items[:2]))    
    return e_list

# get the top layer for each electrode
def get_e2l(e_list, num_layers, model_type, lag):
    path_s = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + str(model_type) + '-hs'
    # go through all electrodes
    print(lag)
    e2l  = {}
    for e in e_list:
        l_corrs = []
        for l in range(num_layers):
            e_dir = path_s + str(l) + '/777/' + e + '_comp.csv'
            #e_sig = pd.read_csv(e_dir + e + '_comp.csv')
            with open(e_dir, 'r') as e_f:
                e_sig = list(map(float, e_f.readline().strip().split(',')))
            l_corrs.append(e_sig[len(e_sig)//2 + lag])
        e2l[e] = np.argmax(l_corrs)

    return e2l

# get layer --> color dictionary
def get_dicts(num_gs, num_layers):
    nl_pg = int(num_layers/4)
    layers = list(range(num_layers))
    possible_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #g2c = {} # group to color
    #l2g = {} # layer to group
    l2c = {}
    for i in range(num_gs):
        #g2c[i] = clr.to_rgb(possible_colors[i])
        #l2g.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.ones(nl_pg)*i)))    
        l2c.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.repeat(np.array([clr.to_rgb(possible_colors[i])]),nl_pg ,axis=0)))) 
    #return l2g, g2c
    return l2c

# get electrode --> color dictionary
def get_e2c(e2l, l2c):
    e2c = {}
    for e, l in e2l.items():
        e2c[e] = l2c[l]
    return e2c
 
def save_e2c(e2c, in_f):
    out_f = open('brain_map_text_out_lag-900.txt', 'w')
    sep = '\t'
    for line in open(in_f, 'r'):
        items = line.strip('\n').split('\t')
        out_f.write(sep.join(items[1:]) + '\t' + sep.join(list(map(str,e2c['_'.join(items[:2])]))) + '\n')
    
    out_f.close()

if __name__ == '__main__':
    # load file
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    e_list = get_e_list(in_f)
    # get top layer for each electrode
    model_type = 'gpt2-xl'
    num_layers = get_n_layers(model_type)
    lag = -900
    adjusted_lag = int(lag/25) #this adjusts for binning  
    e2l = get_e2l(e_list, num_layers, model_type, adjusted_lag)
    # divide layers into groups and assign colors 
    num_gs = 4 # up to 7 (see line below) num groups
    #layer2group, group2color = get_dicts(num_gs, num_layers)    
    layer2color = get_dicts(num_gs, num_layers)
    # assign color to electrode
    e2c = get_e2c(e2l, layer2color)    
    # save output    
    
    #breakpoint()
    save_e2c(e2c, in_f)

