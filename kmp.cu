/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM

THIS VERSION CHECKS RESULTS FOR CORRECTNESS COMPARING THEM TO THE ONES OF THE NAIVE ALGORITHM
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX 100

__managed__ int match = 0; // to know whether at least a match in the whole file was found

// Utility to check for CUDA errors
inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// Naive algorithm to be used as baseline to check for correctness
void naiveSearch(char *text, char *pattern, int N, int M, int* naiveResult) {
  for (int i = 0; i <= N - M; ++i) {
    int j;
    for (j = 0; j < M && text[i + j] == pattern[j]; ++j);

    if (j == M) // match found from i to i + M - 1
      naiveResult[i] = i + M - 1;
  }
}

// To compute the auxiliary array which serves as a "failure table"
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

// Core of the KMP algorithm, separated from the 'kmp' wrapper function to be assigned to each thread
__global__ void patternMatch(char *pattern, char *text, int *next, int *kmpResult, int M, int N) {
  int j; // current position in pattern
  int k; // current position in text
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // thread's identifier
  int sublength = ceilf( (float) N / (gridDim.x * blockDim.x)); // text characters divided by the number of threads
  int start = idx * sublength; // initial delimiter for each thread (included)
  int stop = start + sublength; // final delimiter for each thread (excluded)

  for (j = 0, k = start; k < N && (k - j < stop); ++j, ++k) {
    while (j >= 0 && text[k] != pattern[j])
      j = next[j];

    if (j == M - 1) { // a match was found ('M - 1' because j hasn't been incremented yet)
      match = 1;
      kmpResult[k - M + 1] = k; // match found from k - M + 1 to k
      j = next[j - 1]; // to reset j
    }
  }
}

// Wrapper
void kmp(char *text, char *pattern, int N, int M, int line) {
  int *next; // auxiliary array to know how far to slide the pattern when a mismatch is detected
  int *kmpResult; // array to store the matched text's positions by KMP
  int *naiveResult; // array to store the matched text's positions by the naive algorithm
  int verified = 1; // flag to verify the result with respect to the naive algorithm

  checkCuda(cudaMallocManaged(&next, M * sizeof(int)));
  computeNext(next, pattern, M);

  checkCuda(cudaMallocManaged(&kmpResult, N * sizeof(int)));
  naiveResult = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i) {
    kmpResult[i] = -1;
    naiveResult[i] = -1;
  }
  
  size_t threads_per_block = 4;
  size_t number_of_blocks = 2;

  patternMatch<<<number_of_blocks, threads_per_block>>> (pattern, text, next, kmpResult, M, N);
  checkCuda(cudaGetLastError());
  
  checkCuda(cudaDeviceSynchronize());

  checkCuda(cudaFree(next));

  naiveSearch(text, pattern, N, M, naiveResult);

  for (int i = 0; i < N; ++i)
    if (kmpResult[i] != naiveResult[i])
      verified = 0;
  
  if (!verified)
    printf ("Results for line %d are not correct.\n", line);

  for (int i = 0; i < N; ++i)
    if (kmpResult[i] != -1 && naiveResult[i] != -1 && kmpResult[i] == naiveResult[i])
      printf ("Match found on line %d from position %d through %d\n", line, i + 1, kmpResult[i] + 1);
  
  checkCuda(cudaFree(kmpResult));
  free(naiveResult);
}

int main(int argc, char *argv[]) {
  FILE *fp;
  char buffer[MAX+1], *text, *pattern;
  int N, M, line;

  if (argc < 3) {
    printf ("Provide the name of the text file as the first argument and the pattern to search as the second one.\n");
    return EXIT_FAILURE;
  }

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("Error with file\n");
    return EXIT_FAILURE;
  }

  M = strlen(argv[2]);
  checkCuda(cudaMallocManaged(&pattern, (M + 1) * sizeof(char)));
  strncpy(pattern, argv[2], M);

  line = 0;
  while (fgets(buffer, MAX+1, fp) != NULL) { // we apply kmp to each line of the file
    buffer[strcspn(buffer, "\n")] = 0; // to remove \n
    N = strlen(buffer);
    checkCuda(cudaMallocManaged(&text, (N + 1) * sizeof(char)));
    strncpy(text, buffer, N);

    kmp(text, pattern, N, M, ++line);
    
    checkCuda(cudaFree(text));
  }

  checkCuda(cudaFree(pattern));

  if (!match)
    printf ("No match found.\n");

  return 0;
}