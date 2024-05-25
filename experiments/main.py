import sys
import matplotlib.pylab as plt
import torch
from tqdm import tqdm
import os
import gpytorch
import numpy as np
from numpy.random import RandomState
import random
sys.path.append('../')
from dataset_mine import load_dataset
from models.gp_rff import RFF_GPLVM
from visualizer import Visualizer
from metrics import (knn_classify,
                     mean_squared_error,
                     r_squared)

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

ds = load_dataset(rng, 's-curve-torch', 'gaussian')   # RBF + periodic 
# ds = load_dataset(rng, 's-curve', 'gaussian')       # RBF 
plt.plot(range(ds.F.shape[0]), ds.F[:, 0])
plt.plot(range(ds.F.shape[0]), ds.F[:, 1])
plt.plot(range(ds.F.shape[0]), ds.F[:, 2])
# plt.plot(range(ds.F.shape[0]), ds.F[:, 3])
plt.show()
Y = ds.Y / np.linalg.norm(ds.Y)  


""" ##-------------------------------- Parameters Settings ------------------------------------------------"""
setting_dict = {}
setting_dict['num_m'] = 2               # if num_m = 1, it is using SE kernel
setting_dict['num_sample_pt'] = 25
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
res_dir = f'/gplvm-sm-proj/GPLVM-SM-new/demo/saved_models/{model_name}/'
viz = Visualizer(res_dir+'figures', ds)

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

        F, K = GPLVM_model.f_eval(batch_y=ds.Y, x_star=None)
        F = (F - 5).cpu().detach().numpy()  # because data generation adds mean=5,  shape: N_star * obs_dim
        K = K.cpu().detach().numpy()

        viz.plot_iteration(i + 1,  Y=0,  F=0,  K=0, X=GPLVM_model.mu_x.cpu().detach().numpy())

        viz.plot_F(i+1, F)
        save_models(model=GPLVM_model, optimizer=optimizer, epoch=i, losses=losstotal,
                    result_dir=res_dir, data_name='s-curse', save_model=False)

        # Log metrics.
        # ------------
        mse_Y = mean_squared_error(F+5., ds.Y)
        print(f'MSE Y:  {mse_Y}')

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

