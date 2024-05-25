"""============================================================================
Dataset loading functions.
============================================================================"""

"""============================================================================
Dataset loading functions.
============================================================================"""

from .dataset import Dataset
from GPy import kern
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import (expit as logistic,
                           logsumexp)
from sklearn.datasets import (make_blobs,
                              make_circles,
                              make_moons,
                              make_s_curve,
                              load_breast_cancer)

# -----------------------------------------------------------------------------

# if 'gwg3' in os.getcwd():
#     BASE_DIR = f'{REMOTE_DIR}/datasets'
# else:
#     BASE_DIR = f'{LOCAL_DIR}/datasets'

BASE_DIR = '/gplvm-sm-proj/'

# -----------------------------------------------------------------------------

def load_dataset(rng, name, emissions, test_split=0):
    """Given a dataset string, returns data and possibly true generative
    parameters.
    """
    loader = {
        'bridges': load_bridges,
        '3PhData': load_3PhData,
        'cifar': load_cifar,
        'cmu': load_cmu,
        'cmu1': load_cmu1,
        'cmu2': load_cmu2,
        'cmu3': load_cmu3,
        'cmu4': load_cmu4,
        'congress': load_congress,
        # 'covid19geo': load_covid,
        # 'covid19time': load_covid,
        'covid': load_covid,
        'fiji': load_fiji,
        'highschool': load_highschool,
        'hippo': load_hippocampus,
        'lorenz': load_lorenz,
        'mnist': load_mnist,
        'mnistb': load_mnistb,
        'brendan_faces': load_brendan_faces,
        'montreal': load_montreal,
        'newsgroups': load_20newsgroups,
        'simdata1': load_simdata,
        'simdata2': load_simdata,
        'simdata3': load_simdata,
        'spam': load_spam,
        's-curve-torch': gen_s_curve_torch,
        's-curve': gen_s_curve,
        's-curve-batch': gen_s_curve_batch,
        'spikes': load_spikes,
        'yale': load_yale,
        'ovarian': load_ovarian,
        'cancer': load_cancer,
        'exchange': load_exchange
    }[name]

    if name == 's-curve' or name == 's-curve-batch' or name == 's-curve-torch':
        return loader(rng, emissions, test_split)
    else:
        return loader(rng, test_split)


# -----------------------------------------------------------------------------

def load_pm25(rng, test_split):
    data = pd.read_csv(f"{BASE_DIR}datasets/pm25/PRSA_data_2010.1.1-2014.12.31.csv", na_values=["NA"])
    data = data.dropna()


# -----------------------------------------------------------------------------
def load_3PhData(rng, test_split):
    import tarfile
    with tarfile.open(f'{BASE_DIR}datasets/3PhData/3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

    Y = np.loadtxt(fname='DataTrn.txt')
    labels = np.loadtxt(fname='DataTrnLbls.txt')
    labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

    return Dataset(rng, '3PhData', True, Y, latent_dim=3, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def load_exchange(rng, test_split):
    data = pd.read_csv(f"{BASE_DIR}datasets/exchange/Foreign_Exchange_Rates.csv",
                       skiprows=1, na_values=["ND"])
    data = data.dropna()
    Y = np.log(data.iloc[:, 2:].to_numpy())
    Y = np.log(data.iloc[:, 2:].to_numpy())
    inds = np.arange(0, Y.shape[0], 4)
    Y = Y[inds]
    Y -= Y.mean(axis=0)
    Y /= Y.std(axis=0)
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'exchange', False, Y, latent_dim=2, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cancer(rng, test_split):
    X, Y = load_breast_cancer(return_X_y=True)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return Dataset(rng, 'cancer', True, X, labels=Y, test_split=test_split)


# -----------------------------------------------------------------------------
def load_ovarian(rng, test_split):
    ovarian = np.loadtxt(f"{BASE_DIR}datasets/ovarian_cancer_cleaned.csv")
    ovarian -= ovarian.mean(axis=0)
    ovarian /= ovarian.std(axis=0)
    return Dataset(rng, 'ovarian', False, ovarian, test_split=test_split)


# -----------------------------------------------------------------------------
def load_lorenz(rng, test_split):
    def dyn_lorenz(T, dt=0.01):
        stepCnt = T

        def lorenz(x, y, z, s=10, r=28, b=2.667):
            x_dot = s * (y - x)
            y_dot = r * x - y - x * z
            z_dot = x * y - b * z
            return x_dot, y_dot, z_dot

        # Need one more for the initial values
        xs = np.empty((stepCnt + 1,))
        ys = np.empty((stepCnt + 1,))
        zs = np.empty((stepCnt + 1,))

        # Setting initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)

        # Stepping through "time".
        for i in range(stepCnt):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)

        z = np.zeros((T, 3))
        z[:, 0] = xs[:-1]
        z[:, 1] = ys[:-1]
        z[:, 2] = zs[:-1]
        return z

    def map_tanh(z, D, J):
        Wz_true = np.random.normal(0, 1, [D, J])
        mu = np.dot(z, Wz_true)
        return np.tanh(mu)

    N = 500
    J = 50
    D = 3

    z_all = dyn_lorenz(N)
    z_sim = z_all[-N:, :]
    z_sim_norm = z_sim - z_sim.mean(axis=0)
    z_sim_norm /= np.linalg.norm(z_sim_norm, axis=0, ord=np.inf)
    X = np.copy(z_sim_norm)

    F = 10. * map_tanh(X, D, J)
    F -= F.mean(axis=0)
    F /= F.std(axis=0)
    Y = F + rng.normal(size=(F.shape))

    t = np.linspace(-1, 1, Y.shape[0])

    return Dataset(rng, 'lorenz', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------

def load_cmu4(rng, test_split):
    Y = np.loadtxt(f'datasets/cmu4/CMU_01_04.csv')
    Y = Y[np.arange(0, Y.shape[0], 5)]
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'cmu4', False, Y, latent_dim=3, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cmu3(rng, test_split):
    Y = np.loadtxt(f'datasets/cmu3/CMU_01_03.csv')
    Y = Y[np.arange(0, Y.shape[0], 5)]
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'cmu3', False, Y, latent_dim=3, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cmu2(rng, test_split):
    Y = np.loadtxt(f'datasets/cmu2/CMU_01_02.csv')
    Y = Y[np.arange(0, Y.shape[0], 5)]
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'cmu2', False, Y, latent_dim=3, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cmu1(rng, test_split):
    Y = np.loadtxt(f'datasets/cmu1/CMU_01_01.csv')
    Y = Y[np.arange(0, Y.shape[0], 5)]
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'cmu1', False, Y, latent_dim=3, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cmu(rng, test_split):
    cmu = loadmat(f'datasets/cmu/CMU_0.mat')
    Y = cmu['data']
    Y = Y[np.arange(0, Y.shape[0], 5)]
    t = np.linspace(-1, 1, Y.shape[0])
    return Dataset(rng, 'cmu', False, Y, latent_dim=3, labels=t,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_bridges(rng, test_split):
    """Load NYC bridges dataset:
    https://data.cityofnewyork.us/Transportation/
      Bicycle-Counts-for-East-River-Bridges/gua4-p9wg
    """
    data = np.load(f'{BASE_DIR}datasets/bridges/bridges.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'bridges', True, Y=Y, labels=labels,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_cifar(rng, test_split):
    """Subset of Cifar-10.
    """
    data = np.load(f'{BASE_DIR}datasets/cifar/cifar10_small.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    assert (Y.shape[0] == labels.size)
    return Dataset(rng, 'cifar', True, Y=Y, labels=labels,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_congress(rng, test_split):
    """Congress 109 data:
    https://github.com/jgscott/STA380/blob/master/data/congress109.csv
    https://github.com/jgscott/STA380/blob/master/data/congress109members.csv
    """
    df1 = pd.read_csv(f'{BASE_DIR}datasets/congress109.csv')
    df2 = pd.read_csv(f'{BASE_DIR}datasets/congress109members.csv')
    assert (len(df1) == len(df2))

    # Ensure same ordering.
    df1 = df1.sort_values(by='name')
    df2 = df2.sort_values(by='name')

    Y = df1.values[:, 1:].astype(int)
    labels = np.array([0 if x == 'R' else 1 for x in df2.party.values])
    return Dataset(rng, 'congress109', True, Y, labels=labels,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_covid(rng, is_geo=False):
    """COVID-19 data:
    https://github.com/CSSEGISandData/COVID-19.
    """
    data = np.load(f'{BASE_DIR}datasets/covid/covid19_deaths.npy', allow_pickle=True)
    data = data[()]
    Y = data['cases']
    labels = data['dates']

    # Currently, Y is NxJ where N is # dates and J is # locations. Thus, X is
    # NxD; in other words, it is a low-dimensional structure across time.
    # However, obviously COVID-19 cases increases monotonically with time. We
    # explore transposing to see if we can find low-dimensional structure
    # across space.
    if is_geo:
        Y = Y.T

    geo_str = 'geo' if is_geo else 'time'
    dataset = Dataset(rng, f'covid19{geo_str}', 0, False, Y,
                      labels=labels)
    dataset.lat = data['lat']
    dataset.long = data['long']
    dataset.regions = data['regions']
    dataset.is_geo = is_geo

    return dataset


# -----------------------------------------------------------------------------
def load_fiji(rng, test_split):
    """Fiji children born dataset:
    https://data.princeton.edu/wws509/datasets/#ceb
    """
    data = np.load(f'{BASE_DIR}datasets/fiji/fiji.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'fiji', True, Y, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def load_highschool(rng, test_split):
    """High school dataset:
    https://stats.idre.ucla.edu/stata/dae/negative-binomial-regression/
    """
    data = np.load(f'{BASE_DIR}datasets/highschool/highschool.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'fiji', True, Y, labels=labels, latent_dim=1,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_hippocampus(rng, test_split):
    """Hippocampal place cells data from (Wu 2017).
    """
    data = np.load(f'{BASE_DIR}datasets/hippocampus/hippo_3regions.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    X = data['X_pos']
    labels = data['labels']
    return Dataset(rng, 'hippo', True, Y, X=X, labels=labels, latent_dim=3,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def _get_mnist(rng):
    N = 1000
    # data = np.load(f'{BASE_DIR}/mnist_small.npy', allow_pickle=True)
    data = np.load(f'{BASE_DIR}datasets/mnist/mnist.npy', allow_pickle=True)
    data = data[()]
    Y = data['data']
    labels = data['targets']
    Y = Y.reshape(-1, 28 * 28)
    inds = rng.permutation(len(Y))
    inds = inds[:N]
    Y = Y[inds]
    labels = labels[inds]
    return Y, labels


# -----------------------------------------------------------------------------
def load_mnist(rng, test_split):
    """Subset of MNIST.
    """
    Y, labels = _get_mnist(rng)
    return Dataset(rng, 'mnist', True, Y, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def load_brendan_faces(rng, test_split):
    import pods
    Y = pods.datasets.brendan_faces()['Y']
    labels = None

    return Dataset(rng, 'brendan_faces', False, Y, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def load_mnistb(rng, test_split):
    """Subset of binary MNIST.
    """
    Y, labels = _get_mnist(rng)
    Y[Y > 0] = 1
    assert (np.logical_or(Y == 0, Y == 1).all())
    return Dataset(rng, 'mnistb', True, Y, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def load_montreal(rng, test_split):
    """Montreal bicycle dataset:
    https://www.kaggle.com/pablomonleon/montreal-bike-lanes
    """
    data = np.load(f'{BASE_DIR}datasets/montreal/montreal.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'montreal', True, Y, labels=labels,
                   test_split=test_split)


# -----------------------------------------------------------------------------
def load_20newsgroups(rng, test_split):
    """20 Newsgroups.
    """
    data = np.load(f'{BASE_DIR}datasets/20newsgroups/20newsgroups.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, '20newsgroups', True, Y, labels=labels,  test_split=test_split)


# -----------------------------------------------------------------------------
def load_oil_phase_data(rng, test_split):
    """Oil phase data from Lawrence's original GPLVM paper:
    https://inverseprobability.com/3PhaseData.html.
    """
    data = np.load(f'{BASE_DIR}datasets/oil_phase_data/oil_data.npy', allow_pickle=True)
    data = data[()]
    Y = data['data']
    labels = data['labels']
    return Dataset(rng, 'oil', True, Y, labels=labels,  test_split=test_split)


# -----------------------------------------------------------------------------
def load_simdata(rng, test_split):
    """Synthetic data from (Wu 2017).
    """
    mat = loadmat(f'{BASE_DIR}datasets/simdata/simdata1.mat')
    data = mat['simdata'][0][0]
    Y = data['spikes'].astype(int)
    F = data['spikeRates']
    X = data['latentVariable']
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    labels = data['tgrid'].flatten()
    desc = 'GPTuningCurves'
    # {
    #     '1': 'GPTuningCurves',
    #     '2': 'GaussianBumpTuningCurves'
    # }[num]
    return Dataset(rng, desc, False, Y, X=X, F=F, labels=labels,  test_split=test_split)


# -----------------------------------------------------------------------------
def load_spam(rng, test_split):
    """SMS-spam dataset.
    """
    data = np.load(f'{BASE_DIR}datasets/spam/spam_small.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'spam', True, Y, labels=labels,  test_split=test_split)


# -----------------------------------------------------------------------------
def load_spikes(rng, test_split):
    """Synthetic data of grid cell responses during 2D random walks in real
    space.
    """
    data = np.load(f'{BASE_DIR}datasets/spikes/spks.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    X = data['X']
    return Dataset(rng, 'synthspikes', False, Y, X=X, test_split=test_split)


# -----------------------------------------------------------------------------
def load_yale(rng, test_split):
    """Yale Face Database:
    https://www.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html
    """
    data = np.load(f'{BASE_DIR}datasets/yale/yale.npy', allow_pickle=True)
    data = data[()]
    Y = data['Y']
    labels = data['labels']
    return Dataset(rng, 'yale', True, Y, labels=labels, test_split=test_split)


# -----------------------------------------------------------------------------
def gen_s_curve(rng, emissions, test_split):
    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    K = kern.RBF(input_dim=D, lengthscale=1).K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R,
                       latent_dim=D, labels=t, test_split=test_split)
    else:
        assert (emissions == 'poisson')
        print("Poission")
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)


# -----------------------------------------------------------------------------
def gen_s_curve_batch(rng, emissions, test_split):
    """Generate synthetic data from datasets generating process.
    """
    batch_size = 77
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    K = kern.RBF(input_dim=D, lengthscale=1).K(X)

    Yb = np.empty((batch_size, N, J))
    Fb = np.empty((batch_size, N, J))
    if emissions == 'negbinom':
        Rb = np.empty((batch_size, J))

    for i in range(batch_size):
        F = rng.multivariate_normal(np.zeros(N), K, size=J).T

        # Generate emissions using `F` and/or `K`.
        # ----------------------------------------
        if emissions == 'bernoulli':
            P = logistic(F)
            Y = rng.binomial(1, P).astype(np.double)
        if emissions == 'gaussian':
            Y = F + np.random.normal(0, scale=0.5, size=F.shape)
        elif emissions == 'multinomial':
            C = 100
            pi = np.exp(F - logsumexp(F, axis=1)[:, None])
            Y = np.zeros(pi.shape)
            for n in range(N):
                Y[n] = rng.multinomial(C, pi[n])
        elif emissions == 'negbinom':
            P = logistic(F)
            R = np.arange(1, J + 1, dtype=float)
            Y = rng.negative_binomial(R, 1 - P)
        else:
            assert (emissions == 'poisson')
            theta = np.exp(F)
            Y = rng.poisson(theta)

        Yb[i] = Y
        Fb[i] = F
        if emissions == 'negbinom':
            Rb[i] = R
    if emissions == 'bernoulli':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    if emissions == 'gaussian':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    elif emissions == 'multinomial':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, Rb, t,
                       test_split=test_split)
    else:
        assert (emissions == 'poisson')
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)


# -----------------------------------------------------------------------------
def gen_s_curve_torch(rng, emissions, test_split):

    from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
    import torch

    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]


    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    RBF_cov = ScaleKernel(RBFKernel())
    Period_cov = ScaleKernel(PeriodicKernel())
    # parameter setting
    RBF_cov.outputscale = 0.5
    RBF_cov.base_kernel.lengthscale = 1
    #
    Period_cov.outputscale = 0.5
    Period_cov.base_kernel.lengthscale = 1.0
    Period_cov.base_kernel.period_length = 4.5  # setting
    # Period_cov.base_kernel.period_length = 5    # setting 1
    # Period_cov.base_kernel.period_length = 4    # setting 2

    # K = Period_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    # K = RBF_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    K = RBF_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy() + \
        Period_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    # K = kern.RBF(input_dim=D, lengthscale=1).K(X)



    '''# -------------------------------------------------------'''
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R, latent_dim=D, labels=t, test_split=test_split)

    else:
        assert (emissions == 'poisson')
        print("Poission")
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

