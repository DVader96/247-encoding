import glob
import os
from copy import deepcopy

from statsmodels.stats.multitest import fdrcorrection as fdr
import matplotlib.pyplot as plt
import numpy as np
import math
from brain_color_prepro import get_e_list, get_n_layers
import matplotlib.image as img
import pandas as pd

# TODO: this file is work in progress
def sigmoid(x):
    return 1/(1+math.exp(-x))

def extract_correlations(directory_list):
    #breakpoint()
    all_corrs = []
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*.csv'))
        for file in file_list:
            if file[-16:] == '742_G64_comp.csv':
                continue
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return mean_corr

def extract_single_correlation(directory_list, elec):
    count = 0
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, elec + '*.csv'))
        for file in file_list:
            if file[-16:] == '742_G64_comp.csv':
                continue
            count +=1
            
            breakpoint()
            # ha is a sanity check
            #with open(file, 'r') as csv_file:
            #    ha = list(map(float, csv_file.readline().strip().split(',')))
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            #all_corrs.append(ha)
            # add electrode to dict with index
    if count == 0:
        breakpoint()
    #hat = np.stack(all_corrs)
    return e_Rs


def extract_single_sig_correlation(directory_list, elec):
    count = 0
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, elec + '*.csv'))
        for file in file_list:
            if file[-16:] == '742_G64_comp.csv':
                continue
            count +=1
            
            #breakpoint()
            #with open(file, 'r') as csv_file:
            #    ha = list(map(float, csv_file.readline().strip().split(',')))
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            for i in range(len(e_Rs)):
                if e_Qs[i] > 0.01:
                    #breakpoint()
                    e_Rs[i] = np.nan
            #breakpoint()
            #all_corrs.append(ha)
            # add electrode to dict with index
    if count == 0:
        breakpoint()
    #hat = np.stack(all_corrs)
    return e_Rs#, test_eRs

def get_brain_im(elec):
    patient = elec.split('_')[0]
    elecn = elec[len(patient) + 1:] # skip patient and underscore
    file_n = '/scratch/gpfs/eham/247-encoding-updated/data/images/' + patient + '_left_both' + patient + '_' + elecn + '_single.png'
    try: 
        im = img.imread(file_n)    
    except:
        print('no file for ' + elec)

    return im

def all_rest_true_plots(w_type, topn, emb, true, rest, full):
    fig, ax = plt.subplots()
    lags = np.arange(-2000, 2001, 25)
    ax.plot(lags, emb, 'k', label='contextual') #**
    ax.plot(lags, true, 'r', label='true')
    ax.plot(lags, rest, 'orange', label='not true')
    ax.plot(lags, full, 'b', label = 'all')
    ax.legend()
    ax.set(xlabel='lag (s)', ylabel='correlation', title=w_type + ' top' + str(topn))
    ax.grid()

    fig.savefig("comparison_new_" + w_type + "_top" + str(topn) + "_no_norm_pca.png")
    #fig.savefig("comparison_old_p_weight_test.png")
    #plt.show()
    

def get_signals(topn, w_type):
    emb_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-emb/*')
    emb = extract_correlations(emb_dir)

    true_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-true-top' + str(topn) + '/*')
    true = extract_correlations(true_dir)
    
    rest_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-rest-top' + str(topn) + '/*')
    rest = extract_correlations(rest_dir)
     
    full_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-all-top' + str(topn) + '/*')
    full = extract_correlations(full_dir)
  
    return emb, true, rest, full
# 161 lags
def plot_electrodes(num_layers, in_type, in2, elec_list, num_lags, ph):
    avg_list = []
    init_grey = 0
    #breakpoint()
    #lags = np.arange(-2000, 2001, 25)
    lags = np.arange(-5000, 5001, 25)
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        #print(e)
        fig, ax = plt.subplots()
        big_e_array = np.zeros((num_layers, num_lags))
        for layer in range(num_layers):
            if ph:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*phase-shuffle*/*')
            else:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*5000/*')
            if e == '742_G64':
                print('bad electrode found')
                return
            #breakpoint()
            big_e_array[layer] = extract_single_correlation(ldir, e)
        
        #init_grey/=(1+1e-2) 
        #breakpoint() 
        
        e_layer = np.argmax(big_e_array, axis=0)
        avg_list.append(e_layer)
        #print(i, init_grey)
        ax.plot(lags, e_layer, color = str(init_grey), label=e) 
    
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
        ax.grid()
        if ph:
            fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + 'phase_shuffle.png')
        else:
            fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + '_5000.png')
        plt.close()
    
    # get baseline
    #ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-*' + in2[:-3] + '*phase_shuffle*/*')
    #breakpoint()
    #shuf = extract_correlations(ldir)
    #breakpoint()
    fig, ax = plt.subplots()
    layer_mean = np.mean(avg_list, axis = 0)
    layer_stde = np.std(avg_list, axis=0)/(np.sqrt(len(layer_mean)))
    ax.plot(lags, layer_mean, '-o', color = 'orange', label='avg')
    #ax.plot(lags, shuf, color = 'k', label='phase shuffled top lag 0 layer')
    ax.fill_between(lags, layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    ax.set_ylim([0, num_layers + 1])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
    ax.grid()
    if ph:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + 'phase_shuffle.png')
    else:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + '_5000.png')
    plt.close()

def plot_sig_electrodes(num_layers, in_type, in2, elec_list, num_lags, ph):
    avg_list = []
    #test_list = []
    init_grey = 0
    #breakpoint()
    #lags = np.arange(-2000, 2001, 25)
    lags = np.arange(-5000, 5001, 25)
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        #print(e)
        #fig, ax = plt.subplots()
        big_e_array = np.zeros((num_layers, num_lags))
        #test = np.zeros((num_layers, num_lags))
        for layer in range(num_layers):
            if ph:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*phase-shuffle*/*')
            else:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*5000/*')
            if e == '742_G64':
                print('bad electrode found')
                return
            # R values per layer per lag for a given electrode
            
            #breakpoint()
            big_e_array[layer] = extract_single_sig_correlation(ldir, e)
        #init_grey/=(1+1e-2) 
        #breakpoint() 
        # top R values for each lag, recorded as the layer for which they occured
        #e_layer = np.argmax(big_e_array, axis=0)
        # need to handle each column separatetly for all nan cases
        e_layer = []
        for i in range(big_e_array.shape[1]):
            try:
                e_layer.append(np.nanargmax(big_e_array[:,i]))
            except ValueError:
                #breakpoint()
                e_layer.append(np.nan) # append nan if all in column are nan
        # list of top R values for each lag for each electrode
        avg_list.append(e_layer)
        #test_list.append(np.argmax(test, axis=0))
        #print(i, init_grey)
        #ax.plot(lags, e_layer, color = str(init_grey), label=e) 
    
        #ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        #ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
        #ax.grid()
        #if ph:
        #    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + 'phase_shuffle.png')
        #else:
        #    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + '_5000.png')
        #plt.close()
    
    # get baseline
    #ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-*' + in2[:-3] + '*phase_shuffle*/*')
    #breakpoint()
    #shuf = extract_correlations(ldir)
    #breakpoint()
    fig, ax = plt.subplots(figsize=[20,10])
    #layer_mean = np.mean(avg_list, axis = 0)
    #layer_stde = np.std(avg_list, axis=0)/(np.sqrt(len(layer_mean)))
    #test_ela = np.array(test_list)
    #layer_sum = np.sum(test_ela, axis = 0)
    #non_zero = np.count_nonzero(test_ela, axis = 0)
    #layer_u_test = layer_sum/non_zero
    
    breakpoint()
    ela = np.array(avg_list)
    layer_mean = np.nanmean(ela, axis=0)
    layer_std = np.nanstd(ela, axis=0)
    sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela), axis=0))
    layer_stde = layer_std/sqrt_size
    ax.plot(lags, layer_mean, '-o', markersize=2,color = 'orange', label='avg')
    #ax.plot(lags, shuf, color = 'k', label='phase shuffled top lag 0 layer')
    ax.fill_between(lags, layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    ax.set_ylim([0, num_layers + 1])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
    ax.grid()
    if ph:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + 'phase_shuffle.png')
    else:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + '_5000_SIG.png')
    plt.close()
   
def plot_layers(num_layers, in_type, elec):
    if elec == '742_G64':
        print('bad electrode, stop')
        return
    fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=[30, 10])
    lags = np.arange(-2000, 2001, 25)
    init_grey = 1 
    #max_cors = []
    #zero_cors = []
    #lag_300_cors = []
    #lag_n300_cors = []
    m0 = []
    m250 = []
    m500 = []
    p250 = []
    p500 = []
    m1000 = []
    p1000 = []
    lag_avg = [] 
    for i in range(num_layers):
        #breakpoint()
        ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-gpt2-xl-hs' + str(i) + '/*')
        if elec == 'all':
            layer = extract_correlations(ldir)
        else: 
            layer = extract_single_correlation(ldir, elec)
        #max_cors.append(np.max(layer))
        #zero_cors.append(layer[len(layer)//2])
        #lag_300_cors.append(layer[len(layer)//2 + 12]) # 12*25 = 300. so 12 right from 0 is 300ms
        #lag_n300_cors.append(layer[len(layer)//2 + -12]) # 12*25 = 300. so 12 right from 0 is 300ms
        l0 = len(layer)//2
        m1000.append(layer[l0 - 40])
        m500.append(layer[l0 - 20])
        m250.append(layer[l0 - 10])
        m0.append(layer[l0])
        p250.append(layer[l0 + 10])
        p500.append(layer[l0 + 20])
        p1000.append(layer[l0 + 40])
        lag_avg.append(np.mean(layer))
        rgb = np.random.rand(3,)
        init_grey -= 1/(math.exp(i*0.001)*(num_layers+1))
        #init_grey /= 1.25
        if i != 0:
            ax.plot(lags, layer, color=str(init_grey), label='layer' + str(i)) #**
        else:
            ax.plot(lags, layer, color='b', label='layer' + str(i)) #**
    if elec == 'all':
        breakpoint()
        out_layer = extract_correlations(glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + in_type[:-3] + '-lm_out-old_test/*'))
    else:
        
        out_layer = extract_single_correlation(glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + in_type[:-3] + '-lm_out-old_test/*'), elec)
    #max_cors.append(np.max(out_layer))
    #zero_cors.append(out_layer[len(out_layer)//2])
    #lag_300_cors.append(out_layer[len(out_layer)//2 + 12])
    #lag_n300_cors.append(out_layer[len(out_layer)//2 + -12])
    l0 = len(out_layer)//2
    m1000.append(out_layer[l0 - 40])
    m500.append(out_layer[l0 - 20])
    m250.append(out_layer[l0 - 10])
    m0.append(out_layer[l0])
    p250.append(out_layer[l0 + 10])
    p500.append(out_layer[l0 + 20])
    p1000.append(out_layer[l0 + 40])
    lag_avg.append(np.mean(out_layer))
    ax.plot(lags, out_layer, color = 'r', label='contextual') 
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='correlation', title= in_type + ' Encoding Over Layers ' + elec)
    ax.grid()
    #fig.savefig("/scratch/gpfs/eham/247-encoding-updated/results/figures/comparison_new_" + in_type +'_' + elec +  str(num_layers) + "layers_no_norm_pca.png")
    #fig.savefig("comparison_old_p_weight_test.png")
    #plt.show()
    #plt.close() 
    #fig2, ax2 = plt.subplots() #figure()
    #plt.plot(range(len(max_cors)), max_cors, '-o', color='r', label='max')
    #breakpoint()
    c = 1.2
    init_grey = 1/c
    
    ax2.plot(range(len(m0)), m500, '-o',color= (0, init_grey,0), label='-500ms')
    init_grey /= c
    ax2.plot(range(len(m0)), m250, '-o',color= (0, init_grey, 0), label = '-250ms')
    init_grey /=c
    ax2.plot(range(len(m0)), m0, '-o', color=(0, init_grey, 0), label='0ms')
    init_grey /=c 
    ax2.plot(range(len(m0)), p250, '-o',color= (0, init_grey, 0), label='250ms')
    init_grey /=c 
    ax2.plot(range(len(m0)), p500, '-o',color= (0, init_grey, 0), label = '500ms')
    ax2.plot(range(len(m0)), lag_avg, '-o', color = (1, 0, 0), label='avg over lags') 
    #plt.title('Corr vs depth')
    #plt.xlabel('Layer')
    #plt.ylabel('R')
    ax2.set(xlabel='Layer', ylabel='R', title='Corr vs Depth ' + elec)
    ax2.legend()
    #plt.legend()
    #fig2.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Corr_vs_Depth' +in_type +'_' +  elec + '.png')
    # brain plot
    #breakpoint()
    if elec != 'all':
        brain_im = get_brain_im(elec)
        ax3.imshow(brain_im)


    fig.tight_layout()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/All_Combined_range' + in_type + '_' + elec + 'old_test.png')
    plt.close()
    return m1000, m500, m250, m0, p250, p500, p1000

if __name__ == '__main__':
    #plot_layers(11, 'key')
    # averagei
    in_type = 'gpt2-xl-hs'
    num_lags = len(np.arange(-5000, 50001, 25))
    in2 = 'gpt2-hs'
    num_layers = get_n_layers(in2[:-3])
    e_list = get_e_list('brain_map_input.txt', '\t')
    #e_list = get_e_list('old_test.txt', ',')
    print(len(e_list))
    print(e_list)
    #plot_layers(num_layers, 'gpt2-xl-hs', 'all')
    ph = False
    #plot_electrodes(num_layers, in_type, in2, e_list, num_lags, ph)
    plot_sig_electrodes(num_layers, in_type, in2, e_list, num_lags, ph)
    #plot_layers(48, in_type, 'all')
    '''
    # individual electrodes
    e_list = get_e_list('brain_map_input.txt')
    e_lags = [[],[],[],[],[],[],[]]
    labels = ['-1000ms', '-500ms', '-250ms', '0ms', '250ms', '500ms', '1000ms']
    in_type = 'gpt2-xl-hs'
    for elec in e_list: 
        print(elec)
        #get_brain_im(elec)
        m1000, m500, m250, m0, p250, p500, p1000 = plot_layers(48, in_type, str(elec))
        e_lags[0].append(m1000)
        e_lags[1].append(m500)
        e_lags[2].append(m250)
        e_lags[3].append(m0)
        e_lags[4].append(p250)
        e_lags[5].append(p500)
        e_lags[6].append(p1000)
    
    fig, ax2 = plt.subplots()
    c = 1.3
    init_grey = 1/c
    #plot_clr = [.9, .1, 0]
    for i, lag in enumerate(e_lags):
        avg = np.mean(lag, axis = 0)
        #ax2.plot(range(len(avg)), avg, '-o',color= (0, init_grey,0), label=labels[i])
        if i > len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (init_grey, 0, 0), label=labels[i])
            init_grey *= c
        elif i == len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (0, 0, 0), label=labels[i])
        elif i < len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (0, init_grey, 0), label=labels[i])
            init_grey /= c
 
        #ax2.plot(range(len(avg)), avg, '-o',color= (plot_clr[0], plot_clr[1], 0), label=labels[i])
        #plot_clr[0] /= 1.2
        #plot_clr[1] *= 1.2
        #init_grey /= c
    
    ax2.set(xlabel='Layer', ylabel='R', title='Corr vs Depth ' + elec)
    ax2.legend()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test2_LayerPlot_avg_electrodes_' + in_type + '.png')
    plt.close()
    #max_l2_grad = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-max-l2-grad/*')
    #l2_grad = extract_correlations(max_l2_grad)
    '''
    #fig, ax = plt.subplots()
    #lags = np.arange(-2000, 2001, 25)
    #ax.plot(lags,l2_grad, 'r', label='true grad')
    #ax.legend()
    #ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
    #ax.grid()

    #fig.savefig("comparison_new_max_l2_grad_no_norm_pca2.png")
#

    #topn_list = [0, 3, 5]
    #w_list = ['reg','pmint', 'pw']
    #for topn in topn_list:
    #    for wtype in w_list:
    #        emb, true, rest, full = get_signals(topn, wtype)
    #        all_rest_true_plots(wtype, topn, emb, true, rest, full)
    ''' 
    reg_true = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-true0-update-no-norm/*')
    r_true = extract_correlations(reg_true)

    reg_all = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-all0-update-no-norm/*')
    r_all = extract_correlations(reg_all)

    reg_rest = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-rest0-update-no-norm/*')
    r_rest = extract_correlations(reg_rest)
    '''
    #emb = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-emb-emb0-update-no-norm/*')
    #emb_v = extract_correlations(emb)

    #all_rest_true_plots('reg-real-no-norm', 0, emb_v, r_true, r_rest, r_all)

    #true_abs = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-abs-abs0-update-no-norm/*') 
    #true_abs_v = extract_correlations(true_abs)

    #sum_abs = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-abs_sum-abs_sum0-update-no-norm/*')
    #sum_abs_v = extract_correlations(sum_abs)

    #rest_abs = true_abs_v

    #all_rest_true_plots('abs', 0, emb_v, true_abs_v, rest_abs, sum_abs_v)
    #sgd_emb = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-train-emb/*'))
    #sgd_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-rest/*'))
    #sgd_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-true/*'))
    #sgd_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-all/*'))
    #adam_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-rest/*'))
    #adam_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-true/*'))
    #adam_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-all/*'))
    #eval_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-rest/*'))
    #eval_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-true/*'))
    #eval_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-all/*'))


    #all_rest_true_plots('sgd', 0, sgd_emb, sgd_true, sgd_rest, sgd_all)
    #all_rest_true_plots('eval', 0, emb_v, eval_true, eval_rest, eval_all)
    
'''
top1_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-top1-pca-new/*')

#python_dir_list = glob.glob(os.path.join(os.getcwd(), 'test-NY*'))
top1_mean_corr = extract_correlations(top1_dir_list)

#matlab_dir_list = glob.glob(os.path.join(os.getcwd(), 'NY*'))
#m_mean_corr = extract_correlations(matlab_dir_list)

w_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-weight-pca-new/*')
w_mean_corr = extract_correlations(w_dir_list)

top1w_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-top1_weight-pca-new/*')
top1w_mean_corr = extract_correlations(top1w_dir_list)

dLdC_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-dLdC/*')
dLdC_mean_corr = extract_correlations(dLdC_dir_list)

wpw_dir_list =  glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-wpw/*')
wpw_mean_corr = extract_correlations(wpw_dir_list)

concat_dLdC_true = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-concat-dLdC-true/*')
concat_dLdC_t_mean_corr = extract_correlations(concat_dLdC_true)

one_over_pmt = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-one_over_pmt/*')
one_over_pmt_mean_corr = extract_correlations(one_over_pmt)



#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-old-pw-test/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw-nopca/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-no-norm/*')
# no norm
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw-no-norm-pca/*')
# re norm
p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pw-pca2/*')
p_mean_corr = extract_correlations(p_weight_dir_list)

#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg/*')
#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg-nopca/*')
# no norm
#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg-no-norm-pca/*')
# re norm
no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-avg-pca/*') 
no_w_avg_mean_corr = extract_correlations(no_w_avg)

#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint/*')
#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint-nopca/*')
# no norm
#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint-no-norm-pca/*')
# re norm
pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pmint-pca2/*') 
pmint_w_mean_corr = extract_correlations(pmint_w)

#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true/*')
#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true-nopca/*')
# no norm
#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true-no-norm-pca/*')
# re norm
true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-true-pca/*') 
true_mean_corr = extract_correlations(true_dir_list)

#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb/*')
#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb-nopca/*')
# no norm
#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb-no-norm-pca/*')
# re norm
reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-emb-pca/*') 
reg_mean_corr = extract_correlations(reg_dir_list)

# verify if you take no norm pw, normalize it, you get an effect
#pw_norm_ver = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pw-pca/*')
#pw_norm_ver_corr = extract_correlations(pw_norm_ver)

fig, ax = plt.subplots()
lags = np.arange(-2000, 2001, 25)
ax.plot(lags, true_mean_corr, 'r', label='true grad')
#ax.plot(lags, pw_norm_ver_corr, 'b', label='norm pw')
#ax.plot(lags, top1_mean_corr, 'b', label='top1 grad') #**
ax.plot(lags, reg_mean_corr, 'k', label='contextual') #**
#ax.plot(lags, w_mean_corr, 'g', label='true weight') #**
#ax.plot(lags, top1w_mean_corr, 'orange', label = 'top1 weight')
ax.plot(lags, p_mean_corr, 'orange', label='p weighted') #**
#ax.plot(lags, dLdC_mean_corr, 'purple', label='dLdC')  #**
#ax.plot(lags, wpw_mean_corr, 'magenta', label = 'wpw') #**
#ax.plot(lags, m_mean_corr, 'r', label='matlab')
#ax.plot(lags, concat_dLdC_t_mean_corr, 'plum', label = 'concatdLdCtrue') #**
#ax.plot(lags, one_over_pmt_mean_corr, 'chartreuse', label = 'grad weight 1/(p-t)') #**
ax.plot(lags, no_w_avg_mean_corr, 'burlywood', label = 'uniform avg')
ax.plot(lags, pmint_w_mean_corr, 'lightcoral', label = 'p-t weighted')
ax.legend()
ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
ax.grid()

fig.savefig("comparison_new_no_norm_pca2.png")
#fig.savefig("comparison_old_p_weight_test.png")
plt.show()
'''
