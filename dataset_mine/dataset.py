"""============================================================================
Abstract dataset attributes.
============================================================================"""
import numpy as np
# import torch
import pdb


# -----------------------------------------------------------------------------

class Dataset:

    def __init__(self, rng, name, is_categorical, Y, test_split, X=None,
                 F=None, K=None, R=None, labels=None, latent_dim=None):

        self.rng = rng
        self.name = name
        self.is_categorical = is_categorical
        self.has_true_X = X is not None
        self.has_true_F = F is not None
        self.has_true_K = K is not None
        self.has_labels = labels is not None
        self._latent_dim = latent_dim

        try:
            if is_categorical and labels is None:
                raise ValueError('Labels must be provided for categorical data.')
        except:
            pdb.set_trace()

        self.Y = Y
        self.R = R
        self.F = F
        self.K = K
        self.X = X
        self.R = R
        self.labels = labels

        self.was_split = test_split > 0
        if self.was_split:
            self.Y_ma, self.F_ma, self.mask = self.Y_missing(test_split)
        else:
            self.Y_ma = None
            self.F_ma = None
            self.mask = None

    def Y_missing(self, test_split, fill_value=0):#np.nan):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        Y_missing = np.copy(self.Y).astype(float)
        mask = self.rng.binomial(1, test_split,
                                 size=self.Y.shape).astype(bool)
        Y_missing[mask] = fill_value
        Y_missing = np.ma.array(Y_missing, mask=mask)
        Y_missing = Y_missing.harden_mask()
        if self.F is not None:
            F_missing = np.copy(self.F)
            F_missing[mask] = fill_value
            F_missing = np.ma.array(F_missing, mask=mask)
            F_missing = F_missing.harden_mask()
        else:
            F_missing = None

        return Y_missing, F_missing, mask

    #    @property
    #    def Y_normalized_tensor(self):
    #        Y = torch.Tensor(self.Y)
    #        Y = Y - Y.mean()
    #        Y = Y / Y.std()
    #        return Y

    @property
    def Y_train(self):
        return self.train_mask(self.Y)

    @property
    def Y_test(self):
        return self.test_mask(self.Y)

    @property
    def F_train(self):
        return self.train_mask(self.F)

    @property
    def F_test(self):
        return self.test_mask(self.F)

    def train_mask(self, Y_or_F):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        mask = self.Y.mask
        return Y_or_F[~mask]

    def test_mask(self, Y_or_F):
        if not self.was_split:
            raise ValueError('Data has not been split.')
        mask = self.Y.mask
        return Y_or_F[mask]

    def __str__(self):
        return f"<class 'datasets.Dataset ({self.name})'>"

    def __repr__(self):
        return str(self)

    @property
    def latent_dim(self):
        if self._latent_dim:
            return self._latent_dim
        elif self.has_true_X:
            return self.X.shape[1]
        else:
            return 2