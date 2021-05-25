import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import math

# TODO: this file is work in progress
def sigmoid(x):
    return 1/(1+math.exp(-x))

def extract_correlations(directory_list):
    all_corrs = []
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*.csv'))
        for file in file_list:
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return mean_corr

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
def plot_layers(num_layers, in_type):
    
    fig, ax = plt.subplots()
    lags = np.arange(-2000, 2001, 25)
    init_grey = 1 
    max_cors = []
    for i in range(num_layers):
        #breakpoint()
        ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + in_type + str(i) + '/*')
        layer = extract_correlations(ldir)
        max_cors.append(np.max(layer))
        #breakpoint()
        rgb = np.random.rand(3,)
        init_grey -= 1/(math.exp(i*0.01)*(num_layers+1))
        ax.plot(lags, layer, color=str(init_grey), label='layer' + str(i)) #**

    ax.plot(lags, extract_correlations(glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-lm-out/*')), color = 'r', label='contextual') 
    ax.legend()
    ax.set(xlabel='lag (s)', ylabel='correlation', title= in_type + ' Encoding Over Layers')
    ax.grid()

    fig.savefig("/scratch/gpfs/eham/247-encoding-updated/results/figures/comparison_new_" + in_type + str(num_layers) + "layers_no_norm_pca.png")
    #fig.savefig("comparison_old_p_weight_test.png")
    #plt.show()
    
    fig2 = plt.figure()
    plt.plot(range(len(max_cors)), max_cors, '-o')
    plt.title('Corr vs depth')
    plt.xlabel('Layer')
    plt.ylabel('R')
    fig2.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Corr_vs_Depth.png')

if __name__ == '__main__':
    #plot_layers(11, 'key')
    plot_layers(12, 'hs')
    #max_l2_grad = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-max-l2-grad/*')
    #l2_grad = extract_correlations(max_l2_grad)
    
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
