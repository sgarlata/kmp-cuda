/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM
VERSION: ONLY USING THE CPU
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

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

    if (pattern[j] = pattern[t])
      next[j] = next[t];
    else
      next[j] = t;
  }
}

void patternMatch(char *pattern, char *text, int *next, int m, int n) {
  int j; // current position in pattern
  int k; // current position in text

  for (j = 0, k = 0; j < m && k < n; ++j, ++k) {
    while (j >= 0 && text[k] != pattern[j]) {
      j = next[j];
    }
  }

  if (j == m)
    printf("Match found in positions %d through %d.\n", k - m, k - 1);
}

int main() {
    const int N = 27; // dimension of text
    const int M = 4; // dimension of pattern
    char *text, *pattern;
    int *next;

    size_t text_size = (N + 1) * sizeof(char);
    size_t pattern_size = (M + 1) * sizeof(char);
    size_t next_size = M * sizeof(int);

    checkCuda(cudaMallocManaged(&text, text_size));
    checkCuda(cudaMallocManaged(&pattern, pattern_size));
    checkCuda(cudaMallocManaged(&next, next_size));

    strncpy(text, "this is a trial to try this", 27);
    strncpy(pattern, "this", 4);

    computeNext(next, pattern, M);
    patternMatch(pattern, text, next, M, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(text));
    checkCuda(cudaFree(pattern));

    return 0;
}