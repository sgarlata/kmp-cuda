# KMP with CUDA
A simple implementation of the KMP algorithm, which leverages GPU's power to parallelize the search of a given pattern within a text.
> **Note:** For simplicity's sake, it uses 2 blocks of 4 threads each. Clearly, these numbers do not affect the program's outcome.

## Requirements
- A CUDA-capable GPU
- The NVIDIA CUDA Toolkit

## Installation
Clone the repository.

## Usage
1. Compile the `kmp.cu` file with the command `nvcc -o kmp kmp.cu`.
2. Execute `kmp` providing the name of the text file and the pattern to search within it, respectively as first and second arguments (e.g., `./kmp text.txt "pattern"`).
> **Tip:** You can find some sample input text files in the included `text` folder.

## Result
The program shows at which positions of a given line of the text the pattern has been found.

Each match is verified by being also found by the naive algorithm for pattern searching.

Finally, the total number of verified matches is shown, possibly together with the the number of missed matches (i.e., those found by the naive algorithm, but not by the KMP one).