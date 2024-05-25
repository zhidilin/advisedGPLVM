from models_utility.function_gp import lt_log_determinant
from torch import triangular_solve
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence
from torch.nn import functional as F
import gpytorch

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.FloatTensor)

zitter = 1e-8


class RFF_GPLVM(nn.Module):
    def __init__(self, num_batch, num_sample_pt, param_dict, Y, device=None, ifPCA=True):
        super(RFF_GPLVM, self).__init__()
        self.device = device
        self.name = None
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt  # L/2
        self.latent_dim = param_dict['latent_dim']  # Q
        self.N = param_dict['N']                    # !!!
        self.num_m = param_dict['num_m']            # m
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']

        self.Y = Y

        self.total_num_sample = self.num_samplept * self.num_m  # m * L/2
        # self.likelihood = Gaussian(variance=self.noise, device=device) if likelihood == None else likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()


        # shape: m * 1
        self.log_weight = nn.Parameter(torch.randn(self.num_m, 1, device=self.device), requires_grad=True)

        if self.num_m==1:
            # if SE kernel is used, then self.mu = 0, and requires_grad=False
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=False)  # shape: m * Q
        else:
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=True)  # shape: m * Q

        self.log_std = nn.Parameter(torch.ones(self.num_m, self.latent_dim, device=self.device), requires_grad=True)  # shape: m * Q
        if ifPCA:
            pca = PCA(n_components=self.latent_dim)
            X = pca.fit_transform(self.Y)
        else:
            X = torch.randn(self.N, self.latent_dim, device=self.device)

        self.mu_x = nn.Parameter(torch.tensor(X, device=self.device), requires_grad=True)    # shape: N * Q
        self.log_sigma_x = nn.Parameter(torch.randn(self.N, self.latent_dim, device=self.device), requires_grad=True)


    def _compute_sm_basis(self, x_star=None, f_eval=False):
        multiple_Phi = []
        current_sampled_spectral_list = []

        if f_eval:  # use to evaluate the latent function 'f'
            x = self.mu_x
        else:
            std = F.softplus(self.log_sigma_x)   # shape: N * Q
            eps = torch.randn_like(std)          # don't preselect/prefix it in __init__ function
            x = self.mu_x + eps * std

        SM_weight = F.softplus(self.log_weight)
        SM_std = F.softplus(self.log_std)

        for i_th in range(self.num_m):  # TODO: check if it can be improved without using for
            SM_eps = torch.randn(self.num_samplept, self.latent_dim, device=self.device)
            sampled_spectral_pt = self.mu[i_th] + SM_std[i_th] * SM_eps  # L/2 * Q
            # # randomly change the sign of sampled spectral points
            # sign = torch.randint_like(sampled_spectral_pt,low=0,high=2)
            # sign[sign==0]=-1
            # sampled_spectral_pt = sign * sampled_spectral_pt

            if x_star is not None:
                current_sampled_spectral_list.append(sampled_spectral_pt)

            x_spectral = (2 * np.pi) * x.matmul(sampled_spectral_pt.t())    # N * L/2

            Phi_i_th = (SM_weight[i_th] / self.num_samplept).sqrt() * torch.cat([x_spectral.cos(), x_spectral.sin()], 1)

            multiple_Phi.append(Phi_i_th)

        if x_star is None:
            return torch.cat(multiple_Phi, 1)  #  N * (m * L）

        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstar_spectral = (2 * np.pi) * x_star.matmul(current_sampled.t())

                Phistar_i_th = (SM_weight[i_th] / self.num_samplept).sqrt() * torch.cat([xstar_spectral.cos(), xstar_spectral.sin()], 1)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)  #  N * (m * L),  N_star * (M * L)


    def _compute_gram_approximate(self, Phi):  # shape:  (m*L) x (m*L)
        return Phi.t() @ Phi + (self.likelihood.noise + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()


    def _compute_gram_approximate_2(self, Phi):  # shape:  N x N
        return Phi @ Phi.T


    def _kl_div_qp(self):

        # shape: N x Q
        q_dist = torch.distributions.Normal(loc=self.mu_x, scale= F.softplus(self.log_sigma_x))
        p_dist = torch.distributions.Normal(loc=torch.zeros_like(q_dist.loc), scale=torch.ones_like(q_dist.loc))

        return kl_divergence(q_dist, p_dist).sum().div(self.N * self.latent_dim)

    def compute_loss(self, batch_y, kl_option):
        """
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """
        obs_dim = batch_y.shape[1]
        obs_num = batch_y.shape[0]
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)
        Phi = self._compute_sm_basis()

        # negative log-marginal likelihood
        if Phi.shape[0]>Phi.shape[1]:  # if N > (m*L)
            Approximate_gram = self._compute_gram_approximate(Phi)  # shape:  (m * L） x  (m * L）
            L = torch.cholesky(Approximate_gram)
            Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(batch_y), L, upper=False)[0]

            # todo: need to double-check this part
            neg_log_likelihood = (0.5 / self.likelihood.noise) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            neg_log_likelihood += lt_log_determinant(L)
            neg_log_likelihood += (-self.total_num_sample) * 2 * self.likelihood.noise.sqrt()
            neg_log_likelihood += 0.5 * obs_num * (np.log(2 * np.pi) + 2 * self.likelihood.noise.sqrt())

        else:
            k_matrix = self._compute_gram_approximate_2(Phi=Phi) # shape: N x N
            C_matrix = k_matrix + self.likelihood.noise * torch.eye(self.N, device=self.device)
            L = torch.cholesky(C_matrix) # shape: N x N
            L_inv_y = triangular_solve(batch_y, L, upper=False)[0]


            # compute log-likelihood by ourselves
            constant_term = 0.5 * obs_num * np.log(2 * np.pi) * obs_dim
            log_det_term = torch.diagonal(L, dim1=-2, dim2=-1).sum().log() * obs_dim
            yy = 0.5 * L_inv_y.pow(2).sum()
            neg_log_likelihood = (constant_term + log_det_term + yy).div(obs_dim * obs_num)

        if kl_option:
            kl_x = self._kl_div_qp().div(self.N * 50)
            loss_all = neg_log_likelihood + kl_x
        else:
            loss_all = neg_log_likelihood

        return loss_all


    def f_eval(self, batch_y, x_star=None):
        """
            evaluation of the latent mapping function

            x_star:         prediction input                            shape: N_star * Q
            batch_y:        observations for characterizing the GP      shape: N * obs_dim
        """
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)

        if x_star is None:
            x_star = self.mu_x

        Phi, Phi_star = self._compute_sm_basis(x_star=x_star, f_eval=True)

        cross_matrix = Phi_star @ Phi.T                                  # shape: N_star * N

        k_matrix = self._compute_gram_approximate_2(Phi=Phi)             # shape: N * N
        C_matrix = k_matrix + self.likelihood.noise * torch.eye(self.N, device=self.device)

        L = torch.cholesky(C_matrix)                                    # shape: N x N
        L_inv_y = triangular_solve(batch_y, L, upper=False)[0]          # inv(L) * y
        K_L_inv = triangular_solve(cross_matrix.T, L, upper=False)[0]   # inv(L) * K_{N, N_star}

        f_star = K_L_inv.T @ L_inv_y                          # shape: N_star * obs_dim

        return f_star, k_matrix
