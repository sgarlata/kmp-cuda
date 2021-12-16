/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX 256

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
      naiveResult[i] = 1;
  }
}

// To compute the auxiliary array which serves as "failure table"
void computeNext(int *next, char *pattern, int M) {
  int j = 0, t = -1;
  next[0] = -1; // next[j] = -1 means that we are to slide the pattern all the way past the current text character

  while (j < M - 1) {
    while (t >= 0 && pattern[j] != pattern[t])
      t = next[t];
    ++t;
    ++j;

    if (pattern[j] == pattern[t])
      next[j] = next[t];
    else
      next[j] = t;
  }
}

// Core of the KMP algorithm, separated from the 'kmp' wrapper function in order to be assignable to each thread
__global__ void patternMatch(char *pattern, char *text, int *next, int *kmpResult, int M, int N, int *countPerThread) {
  int j; // current position in pattern
  int k; // current position in text
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // thread's identifier
  int sublength = ceilf( (float) N / (gridDim.x * blockDim.x)); // text characters divided by the number of threads
  int start = idx * sublength; // initial delimiter for each thread (included)
  int stop = start + sublength; // final delimiter for each thread (excluded)

  j = 0;
  k = start;

  while (k < N && (k - j < stop)) {
    while (j >= 0 && text[k] != pattern[j])
      j = next[j];
    
    ++j;
    ++k;

    if (j == M) { // a match was found
      ++countPerThread[idx];
      kmpResult[k - M] = 1; // match found from k - M to k - 1
      j = next[j]; // to reset j
    }
  }
}

// Wrapper
void kmp(char *text, char *pattern, int N, int M, int line, int *onLineVerified, int *onLineMissed) {
  int *next; // auxiliary array to know how far to slide the pattern when a mismatch is detected
  int *kmpResult; // array to store the matched text's positions by KMP
  int *naiveResult; // array to store the matched text's positions by the naive algorithm
  int *countPerThread; // array with as many cells as threads to store the number of matches found by each of them

  checkCuda(cudaMallocManaged(&next, M * sizeof(int)));
  computeNext(next, pattern, M);

  checkCuda(cudaMallocManaged(&kmpResult, N * sizeof(int)));
  naiveResult = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i) {
    kmpResult[i] = 0;
    naiveResult[i] = 0;
  }
  
  size_t numberOfBlocks = 2;
  size_t threadsPerBlock = 4;
  size_t threadsInGrid = numberOfBlocks * threadsPerBlock; 

  checkCuda(cudaMallocManaged(&countPerThread, threadsInGrid * sizeof(int)));
  for (int i = 0; i < threadsInGrid; ++i)
    countPerThread[i] = 0;

  patternMatch<<<numberOfBlocks, threadsPerBlock>>> (pattern, text, next, kmpResult, M, N, countPerThread);
  checkCuda(cudaGetLastError());
  
  checkCuda(cudaDeviceSynchronize());

  for (int i = 0; i < threadsInGrid; ++i)
    (*onLineVerified) += countPerThread[i];

  checkCuda(cudaFree(next));
  checkCuda(cudaFree(countPerThread));

  naiveSearch(text, pattern, N, M, naiveResult);

  for (int i = 0; i < N; ++i) {
    if (naiveResult[i] == 1 && kmpResult[i] == 1) // by both naive and kmp
      printf ("Verified match on line %d from positions %d through %d\n", line, i + 1, i + M);
    else if (naiveResult[i] == 1 && kmpResult[i] == 0) { // by naive only
      printf ("Missed match on line %d from positions %d through %d\n", line, i + 1, i + M);
      ++(*onLineMissed);
    }
    else if (naiveResult[i] == 0 && kmpResult[i] == 1) { // by kmp only
      printf ("Unverified match on line %d from positions %d through %d\n", line, i + 1, i + M);
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
    printf("Error with file\n");
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