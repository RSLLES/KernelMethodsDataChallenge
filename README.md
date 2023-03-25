# Kernel Methods : Data Challenge
## Authors

- Louis BLAZEJCZAK-BOULEGUE, [GitHub profile](https://github.com/louisbzk)
- Romain SEAILLES [GitHub profile](https://github.com/RSLLES) 

Feel free to contact us if you have any questions or feedback about our solution.

## Installation

To run the Support Vector Classifier, create a Python virtual environment for this project.

```sh
python -m venv env
source env/bin/activate
```
Please note that this project was developped using Python 3.10.

Then, install the required packages :

```sh
pip install -r requirements.txt
```
## Results

You can view the results of our kernels performance comparison in the `./results/README.md` file or by clicking [here](./results/README.md).

## Running

You can run the `run.py` script located at the root of the project. 
The script trains and validates a Support Vector Classifier based on the configuration provided by the config file specified as argument.

```sh
python3 run.py path/to/config_file.py
```

By default, the script will *not* make predictions for validation data. To make the script predict outputs, add the flag `--predict`.

```sh
python3 run.py path/to/config_file.py --predict
```

## Configuration

The configuration file should be a `.py` module with the following variables defined:

### Data

- `data_directory`: directory containing data files.
- `x_val`: filename of the validation data pickle file.
- `x`: filename of the training data pickle file.
- `y`: filename of the training labels pickle file.
- `k_folds_cross_val`: the number of folds for k-folds cross-validation.

### Kernel

The variable `kernel` must be set to an instance of one of the kernel classes (e.g. `WeisfeilerLehmanKernel`).

### Save

- `export_directory`: directory where y_pred is saved as a .csv according to the challenge format. Will only be used with the `--predict` flag.
- `results_directory`: directory where results files are saved.
