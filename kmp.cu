/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM
VERSION: USING ALSO THE GPU (NOT POLISHED)
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

__managed__ int noMatch = 1;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void computeNext(int *next, char *pattern, int m) {
  int j = 1, t = 0;
  next[0] = -1;

  while (j < m) {
    while (t > 0 && pattern[j] != pattern[t])
      t = next[t];
    ++t;
    ++j;

    if (pattern[j] == pattern[t])
      next[j] = next[t];
    else
      next[j] = t;
  }
}

__global__ void patternMatch(char *pattern, char *text, int *next, int m, int n) {
  int j; // current position in pattern
  int k; // current position in text
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int sublength = ceilf( (float) n / (gridDim.x * blockDim.x));
  int start = idx * sublength;
  int stop = start + sublength;
  int proceed = 1;

    printf("Thread: %d. Start: %d. Stop: %d.\n", idx, start, stop);

  for (j = 0, k = start; j < m && k < n && proceed; ++j, ++k) {
    //printf("Thread: %d. Inside for with j = %d and k = %d\n", idx, j, k);
    while (j >= 0 && text[k] != pattern[j]) {
      //printf("Thread: %d. Inside while with j = %d and k = %d\n", idx, j, k);
      j = next[j];
      //printf("Thread: %d. j is now %d\n", idx, j);
      if (k - j >= stop) {
      //printf("Thread: %d. Inside if with k = %d.\n", idx, k);
        proceed = 0;
        }
    }
  }

    if (j == m) {
      noMatch = 0;
      printf("Match found by thread n. %d in positions %d through %d.\n", idx, k - m, k - 1);
    }
}

int main() {
    const int N = 26; // dimension of text
    const int M = 10; // dimension of pattern
    char *text, *pattern;
    int *next;

    size_t text_size = (N + 1) * sizeof(char);
    size_t pattern_size = (M + 1) * sizeof(char);
    size_t next_size = M * sizeof(int);

    checkCuda(cudaMallocManaged(&text, text_size));
    checkCuda(cudaMallocManaged(&pattern, pattern_size));
    checkCuda(cudaMallocManaged(&next, next_size));

    strncpy(text, "babcbabcabcaabcabcabcacabc", 26);
    strncpy(pattern, "abcabcacab", 10);

    size_t threads_per_block = 4;
    size_t number_of_blocks = 2;

    computeNext(next, pattern, M);
    patternMatch<<<number_of_blocks, threads_per_block>>> (pattern, text, next, M, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    if (noMatch)
      printf("No match found.\n");
      
    checkCuda(cudaFree(text));
    checkCuda(cudaFree(pattern));

    return 0;
}