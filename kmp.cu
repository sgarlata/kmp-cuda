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

void computeNext(int *next, char *pattern, int M) {
  int j = 1, t = 0;
  next[0] = -1;

  while (j < M) {
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

__global__ void patternMatch(char *pattern, char *text, int *next, int *matchedText, int M, int N) {
  int j; // current position in pattern
  int k; // current position in text
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int sublength = ceilf( (float) N / (gridDim.x * blockDim.x)); // text characters divided by the number of threads
  int start = idx * sublength; // initial delimiter for each thread (included)
  int stop = start + sublength; // final delimiter for each thread (excluded)
  int proceed = 1;

  for (j = 0, k = start; j < M && k < N && proceed; ++j, ++k) {
    while (j >= 0 && text[k] != pattern[j]) {
      j = next[j];
      if (k - j >= stop)
        proceed = 0; // a match should only be found by a single thread, namely the one whose text's portion contains the initial character of the matched sequence
    }
  }

  if (j == M) {
    match = 1;
    matchedText[k - M] = k - 1;
  }
}

void kmp(char *text, char *pattern, int N, int M) {
  int *next; // auxiliary array to know how far to slide the pattern when a mismatch is detected
  int *matchedText; // array to store the matched text's positions
  int i; // used to loop

  checkCuda(cudaMallocManaged(&next, M * sizeof(int)));
  computeNext(next, pattern, M);

  checkCuda(cudaMallocManaged(&matchedText, N * sizeof(int)));
  for (i = 0; i < N; ++i)
    matchedText[i] = -1;
  
  size_t threads_per_block = 4;
  size_t number_of_blocks = 2;

  patternMatch<<<number_of_blocks, threads_per_block>>> (pattern, text, next, matchedText, M, N);
  checkCuda(cudaGetLastError());
  
  checkCuda(cudaDeviceSynchronize());

  checkCuda(cudaFree(next));

  if (!match)
    printf("No match found.\n");
  else {
    printf("Match found in position(s):\n");
    for (i = 0; i < N; ++i)
      if (matchedText[i] != -1)
        printf("- %d through %d\n", i, matchedText[i]);
  }
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

  printf("Enter pattern (at most %d characters): ", MAX);
  fgets(buffer, MAX+1, stdin);
  buffer[strcspn(buffer, "\n")] = 0; // to remove \n
  M = strlen(buffer);
  checkCuda(cudaMallocManaged(&pattern, (M + 1) * sizeof(char)));
  strncpy(pattern, buffer, M);  

  kmp(text, pattern, N, M);
    
  checkCuda(cudaFree(text));
  checkCuda(cudaFree(pattern));

  return 0;
}