# KMP with CUDA
A simple implementation of the KMP algorithm, which leverages GPU's power to parallelise the search of a given pattern within a text.
## Requirements
- A CUDA-capable GPU
- The NVIDIA CUDA Toolkit

## Installation
Just clone the repository.
## Usage
After having compiled the `kmp.cu` file, just run the obtained executable.
Both things can be performed at once with this simple command:
```
nvcc -o kmp kmp.cu -run
```
Once run, you will be asked to provide first the text and then the pattern. Afterwards, you will be told whether or not a match was found, together with the positions in the text in case of a positive answer.