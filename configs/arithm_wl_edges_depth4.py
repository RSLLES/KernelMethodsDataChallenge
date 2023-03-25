### Data ###
data_directory = "data/"
x_val = "test_data.pkl"
x = "training_data.pkl"
y = "training_labels.pkl"
k_folds_cross_val = 6

### Kernel ###
from kernels.WL import ArithmWeisfeilerLehmanKernel

kernel = ArithmWeisfeilerLehmanKernel(depth=4, use_cache=True, enable_edges_labels=True)

### Save ###
export_directory = "export/"
results_directory = "results/"
