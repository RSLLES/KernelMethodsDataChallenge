### Data ###
data_directory = "data/"
x_val = "test_data.pkl"
x = "training_data.pkl"
y = "training_labels.pkl"
k_folds_cross_val = 5

### Kernel ###
from kernels.WL import WeisfeilerLehmanKernel

kernel = WeisfeilerLehmanKernel(depth=3, use_cache=True)
