from dataset_mine import load_dataset
from numpy.random import RandomState
import numpy as np
from visualizer import Visualizer


if __name__ == "__main__":
    rng = RandomState(0)
    dataset_type = ['bridges', 'cifar', 'mnist', 'montreal', 'newsgroups', 'spam', 'yale', 'highschool', 's-curve']
    for ds_name in dataset_type:
        ds = load_dataset(rng, ds_name, emissions='gaussian', test_split=0)
        if ds_name == "s-curve":
            Y = ds.Y.astype(int)
        else:
            Y = ds.Y
        name_str = './' + ds_name + '.npy'
        np.save(name_str, Y)
