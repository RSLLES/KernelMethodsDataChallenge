### Data ###
data_directory = "data/"
x_val = "test_data.pkl"
x = "training_data.pkl"
y = "training_labels.pkl"
k_folds_cross_val = 6

### Kernel ###
from kernels.JSWL import JensenShannonWeisfeilerLehmanKernel

kernel = JensenShannonWeisfeilerLehmanKernel(
    depth=4, lambd=1.0, enable_edges_labels=True, use_cache=True
)

### Save ###
export_directory = "export/"
results_directory = "results/"
