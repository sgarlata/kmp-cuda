/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX 50

__managed__ int match = 0;

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
  int sublength = ceilf( (float) n / (gridDim.x * blockDim.x)); // text characters divided by the number of threads
  int start = idx * sublength; // initial delimiter for each thread (included)
  int stop = start + sublength; // final delimiter for each thread (excluded)
  int proceed = 1;

  for (j = 0, k = start; j < m && k < n && proceed; ++j, ++k) {
    while (j >= 0 && text[k] != pattern[j]) {
      j = next[j];
      if (k - j >= stop)
        proceed = 0; // a match should only be found by a single thread, namely the one whose text's portion contains the initial character of the matched sequence
    }
  }

  if (j == m) {
    match = 1;
    printf("Match found in positions %d through %d.\n", k - m, k - 1); // found matches are not necessarily printed in order
  }
}

void kmp(char *text, char *pattern, int N, int M) {
  int *next; // auxiliary array to know how far to slide the pattern when a mismatch is detected

  checkCuda(cudaMallocManaged(&next, M * sizeof(int)));
  computeNext(next, pattern, M);
  
  size_t threads_per_block = 4;
  size_t number_of_blocks = 2;

  patternMatch<<<number_of_blocks, threads_per_block>>> (pattern, text, next, M, N);
  checkCuda(cudaGetLastError());
  
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(next));
}

int main() {
  char buffer[MAX+1], *text, *pattern;
  int N, M;

  printf("Enter text (at most %d characters): ", MAX);
  fgets(buffer, MAX+1, stdin);
  buffer[strcspn(buffer, "\n")] = 0; // to remove \n
  N = strlen(buffer);
  checkCuda(cudaMallocManaged(&text, (N + 1) * sizeof(char)));
  strncpy(text, buffer, N);

  printf("Enter pattern: (at most %d characters): ", MAX);
  fgets(buffer, MAX+1, stdin);
  buffer[strcspn(buffer, "\n")] = 0; // to remove \n
  M = strlen(buffer);
  checkCuda(cudaMallocManaged(&pattern, (M + 1) * sizeof(char)));
  strncpy(pattern, buffer, M);  

  kmp(text, pattern, N, M);

  if (!match)
    printf("No match found.\n");
    
  checkCuda(cudaFree(text));
  checkCuda(cudaFree(pattern));

  return 0;
}