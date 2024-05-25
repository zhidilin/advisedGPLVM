import argparse
import sys
sys.path.append('../')
import torch
import random
import numpy as np
from tqdm import tqdm
from dataset_mine import load_dataset
from numpy.random import RandomState
from visualizer import Visualizer
from metrics import (knn_classify,
                     mean_squared_error,
                     r_squared)
import matplotlib.pylab as plt

from utility.eval_metric import _evaluate_metric
from models.gp_rff import RFF_GPLVM

def save_models(model, optimizer, epoch, losses, result_dir, data_name, save_model=True):
    '''

    Parameters
    ----------
    model
    optimizer
    epoch
    losses
    result_dir  :           result saving path
    data_name   :           data name
    jj          :           number of experiment repetition
    save_model  :           indication if to saving model

    Returns
    -------

    '''
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'losses': losses}
    if save_model:
        log_dir = result_dir + f"{data_name}_epoch{epoch}.pt"
        torch.save(state, log_dir)

random_seed = 8
def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

reset_seed(random_seed)
device = "cuda:1"
# device = 'cpu'

""" ##-------------------------------- data loader ------------------------------------------------"""
# Load Dataset
rng = RandomState(random_seed)

## ['bridges', 'congress', 's-curve', 'mnist', 'pm25', 'exchange', 'fiji', 'highschool','cifar', 'cmu', 'cmu1', 'cmu2', 'cmu3', 'cmu4', 'hippo', 'montreal', 'newsgroups',
# 'simdata1','spam', 'spikes', 'yale']
# test_split = None
# ds = load_dataset(rng, 'bridges', 'gaussian')
# test_split = 0.6
# ds = load_dataset(rng, 'mnist', 'gaussian', test_split=test_split)

test_split = 0.6
ds = load_dataset(rng, 'brendan_faces', 'gaussian', test_split=test_split)

""" ##-------------------------------- Parameters Settings ------------------------------------------------"""
setting_dict = {}
setting_dict['num_m'] = 2              # if num_m = 1, it is using SE kernel
setting_dict['num_sample_pt'] = 50
setting_dict['num_total_pt'] = setting_dict['num_m'] * setting_dict['num_sample_pt']
setting_dict['num_batch'] = 1
setting_dict['lr_hyp'] = .01
setting_dict['iter'] = 10000
setting_dict['num_repexp'] = 1
setting_dict['kl_option'] = True  # if adding X regularization in loss function
setting_dict['noise_err'] = 100.0
setting_dict['latent_dim'] = ds.latent_dim
setting_dict['N'] = ds.Y.shape[0]

if setting_dict['num_m'] ==1:
    model_name = f"RFF_GPLVM_SE_{setting_dict['num_sample_pt']}"
else:
    model_name = f"RFF_GPLVM_SM_{setting_dict['num_m']}_{setting_dict['num_sample_pt']}"
res_dir = f'/gplvm-sm-proj/GPLVM-SM-new/experiments/{model_name}/{ds.name}/'
viz = Visualizer(res_dir+'figures', ds)

if test_split is None:
    Y = ds.Y / np.linalg.norm(ds.Y,2)
    Y_original = Y
else:
    if test_split==0:
        Y = ds.Y / 255  # np.linalg.norm(ds.Y,2)
        Y_original = Y
    else:
        Y = ds.Y_ma / 255 # np.linalg.norm(ds.Y_ma)
        Y_original = Y
# Y = ds.Y
print(Y.shape)
setting_dict['noise_err'] = .05 * Y.std()

GPLVM_model = RFF_GPLVM(setting_dict['num_batch'],
                        setting_dict['num_sample_pt'],
                        setting_dict,
                        Y,
                        device=device).to(device)
# GPLVM_model.likelihood.noise=0.5
# GPLVM_model.likelihood.requires_grad_(False)

optimizer = torch.optim.Adam(GPLVM_model.parameters(), lr=setting_dict['lr_hyp'])

viz.plot_iteration(0, Y=0, F=0, K=0, X=GPLVM_model.mu_x.cpu().detach().numpy())

epochs_iter = tqdm(range(setting_dict['iter']+1), desc="Epoch")
for i in epochs_iter:

    GPLVM_model.train()

    optimizer.zero_grad()
    losstotal = GPLVM_model.compute_loss(batch_y = Y, kl_option=setting_dict['kl_option'])
    losstotal.backward()
    optimizer.step()

    if i%200==0:
        print(f'\nELBO: {losstotal.item()}')
        print(f"X_KL: {GPLVM_model._kl_div_qp().item()}")

        F, K = GPLVM_model.f_eval(batch_y=Y_original, x_star=None)
        F = F.cpu().detach().numpy()  #  shape: N_star * obs_dim
        K = K.cpu().detach().numpy()

        # viz.plot_iteration(i + 1,  Y=0,  F=0,  K=0, X=GPLVM_model.mu_x.cpu().detach().numpy())

        # viz.plot_F(i+1, F)
        save_models(model=GPLVM_model, optimizer=optimizer, epoch=i, losses=losstotal,
                    result_dir=res_dir, data_name='s-curse', save_model=False)

        # Log metrics.
        # ------------
        mse_Y = mean_squared_error(F, ds.Y/255)
        print(f'MSE Y:  {mse_Y}')
        # knn_acc = knn_classify(GPLVM_model.mu_x.cpu().detach().numpy(), ds.labels, rng)
        # print('\nKNN acc', knn_acc)

        if ds.has_true_F:
            mse_F = mean_squared_error(F, ds.F)
            print(f'MSE F:  {mse_F}')

        if ds.has_true_K:
            mse_K = mean_squared_error(K, ds.K)
            print(f'MSE K:  {mse_K}')

        if ds.has_true_X:
            r2_X = r_squared(GPLVM_model.mu_x.cpu().detach().numpy(), ds.X)
            print(f'R2 X: {r2_X}')

        print("\n")

# knn_acc = knn_classify(GPLVM_model.mu_x.cpu().detach().numpy(), ds.labels, rng)
# print('\nKNN acc', knn_acc)

num_row = 3
num_col = 7
num = num_row * num_col

# plot predictive images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(F.reshape(-1, 28, 20)[i], cmap='gray')
plt.tight_layout()
plt.savefig(res_dir+f'figures/{ds.name}_predictive_image_{test_split}.pdf', bbox_inches='tight')
plt.show()

# plot training images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(Y.reshape(-1, 28, 20)[i], cmap='gray')
plt.tight_layout()
plt.savefig(res_dir+f'figures/{ds.name}_training_image_{test_split}.pdf', bbox_inches='tight')
plt.show()


# plot true images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(ds.Y.reshape(-1, 28, 20)[i], cmap='gray')
plt.tight_layout()
plt.savefig(res_dir+f'figures/{ds.name}_true_image_{test_split}.pdf', bbox_inches='tight')
plt.show()