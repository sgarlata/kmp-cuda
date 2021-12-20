/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX 1024

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
      naiveResult[i] = 1;
  }
}

// To compute the auxiliary array which serves as "failure table"
void computeF(int *f, char *pattern, int M) {
  int j = 0, i = 1;
  f[0] = 0;

  while (i < M) {
    if (pattern[i] == pattern[j])
      f[i++] = ++j;
    else {
      if (j == 0)
        f[i++] = 0;
      else
        j = f[j - 1];
    }
  }
}

// Core of the KMP algorithm, separated from the 'kmp' wrapper function in order to be assignable to each thread
__global__ void patternMatch(char *pattern, char *text, int *f, int *kmpResult, int M, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // thread's identifier
  int sublength = ceilf((float) N / (gridDim.x * blockDim.x)); // text characters divided by the number of threads
  int start = idx * sublength; // initial delimiter for each thread (included)
  int stop = start + sublength; // final delimiter for each thread (excluded)

  int i = start; // index for text
  int j = 0; // index for pattern

  while (i < N && (i - j < stop)) {
    if (pattern[j] == text[i]) {
      ++j;
      ++i;
    }

    if (j == M) {
      kmpResult[i - M] = 1; // match found from i - M to i - 1
      j = f[j - 1];
    }
    else if (i < N && (i - j < stop) && pattern[j] != text[i]) {
      if (j == 0)
        ++i;
      else
        j = f[j - 1];
    }
  }
}

// Wrapper
void kmp(char *text, char *pattern, int N, int M, int line, int *onLineVerified, int *onLineMissed) {
  int *f; // auxiliary array to know how far to slide the pattern when a mismatch is detected
  int *kmpResult; // array to store the matched text's positions by KMP
  int *naiveResult; // array to store the matched text's positions by the naive algorithm

  checkCuda(cudaMallocManaged(&f, M * sizeof(int)));
  computeF(f, pattern, M);

  checkCuda(cudaMallocManaged(&kmpResult, N * sizeof(int)));
  naiveResult = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i) {
    kmpResult[i] = 0;
    naiveResult[i] = 0;
  }
  
  size_t threadsPerBlock = 32;
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  patternMatch<<<numberOfBlocks, threadsPerBlock>>> (pattern, text, f, kmpResult, M, N);
  checkCuda(cudaGetLastError());
  
  checkCuda(cudaDeviceSynchronize());

  checkCuda(cudaFree(f));

  naiveSearch(text, pattern, N, M, naiveResult);

  for (int i = 0; i < N; ++i) {
    if (naiveResult[i] == 1 && kmpResult[i] == 1) { // by both naive and kmp
      printf ("Verified match on line %d starting at column %d\n", line, i + 1);
      ++(*onLineVerified);
    }
    else if (naiveResult[i] == 1 && kmpResult[i] == 0) { // by naive only
      printf ("Missed match on line %d starting at column %d\n", line, i + 1);
      ++(*onLineMissed);
    }
    else if (naiveResult[i] == 0 && kmpResult[i] == 1) { // by kmp only
      printf ("Unverified match on line %d starting at column %d\n", line, i + 1);
      --(*onLineVerified);
    }
  }
  
  checkCuda(cudaFree(kmpResult));
  free(naiveResult);
}

int main(int argc, char *argv[]) {
  FILE *fp;
  char buffer[MAX+1], *text, *pattern;
  int N; // size of input text
  int M; // size of pattern
  int line = 0; // current line of the input text file
  int onLineVerified = 0; // sum of verified matches on the current line
  int onLineMissed = 0; // sum of missed matches on the current line
  int verified = 0; // total of verified matches
  int missed = 0; // total of missed matches

  if (argc < 3) {
    printf ("Provide the name of the text file as the first argument and the pattern to search as the second one.\n");
    return EXIT_FAILURE;
  }

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("Error with file.\n");
    return EXIT_FAILURE;
  }

  M = strlen(argv[2]);
  checkCuda(cudaMallocManaged(&pattern, (M + 1) * sizeof(char)));
  strncpy(pattern, argv[2], M);

  while (fgets(buffer, MAX+1, fp) != NULL) { // we apply kmp to each line of the file
    buffer[strcspn(buffer, "\n")] = 0; // to remove \n
    N = strlen(buffer);
    checkCuda(cudaMallocManaged(&text, (N + 1) * sizeof(char)));
    strncpy(text, buffer, N);

    kmp(text, pattern, N, M, ++line, &onLineVerified, &onLineMissed);
    verified += onLineVerified;
    missed += onLineMissed;
    
    onLineVerified = 0; // reset
    onLineMissed = 0; // reset
    checkCuda(cudaFree(text));
  }

  checkCuda(cudaFree(pattern));

  if (verified > 0)
    printf ("Verified matches: %d\n", verified);
  if (missed > 0)
    printf ("Missed matches: %d\n", missed);
  if (verified == 0 && missed == 0)
    printf ("No matches\n");

  return 0;
}