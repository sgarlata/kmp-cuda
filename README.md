# KMP with CUDA

A simple implementation of the KMP algorithm, which leverages GPU's power to parallelize the search of a given pattern within a text.

## Requirements
- A CUDA-capable GPU
- The NVIDIA CUDA Toolkit

## Installation
Clone the repository.

## Usage
1. Compile the `kmp.cu` file with the command `nvcc -o kmp kmp.cu`.
2. Execute `kmp` providing the name of the text file and the pattern to search within it, respectively as first and second arguments (e.g., `./kmp text.txt "pattern"`).
3. In case of at least a match, the program will indicate the position of the leftmost one.