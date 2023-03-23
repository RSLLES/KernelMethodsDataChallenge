#!/bin/bash
set -e

echo "### Unit test ###"
python -m unittest -v kernels.test_kernels

echo "### Performance w/ cache ###"
python -m timeit -n 15 -s \
    "from preprocessing.load import load_file ; from kernels.WL import WeisfeilerLehmanKernel; G = load_file('data/training_data.pkl')[0:20]" \
    "WeisfeilerLehmanKernel(depth=3, use_cache=True)(G, G)"

echo "### Performance w/o cache ###"
python -m timeit -n 15 -s \
    "from preprocessing.load import load_file ; from kernels.WL import WeisfeilerLehmanKernel; G = load_file('data/training_data.pkl')[0:20]" \
    "WeisfeilerLehmanKernel(depth=3, use_cache=False)(G, G)"

