### Data ###
data_directory = "data/"
x_val = "test_data.pkl"
x = "training_data.pkl"
y = "training_labels.pkl"
k_folds_cross_val = 6

### Kernel ###
from kernels.GWWL import GeneralizedWassersteinWeisfeilerLehmanKernelSum

kernel = GeneralizedWassersteinWeisfeilerLehmanKernelSum(
    root="../cost_matrices/",
    use_cache=True,
)

### Save ###
export_directory = "export/"
results_directory = "results/"
