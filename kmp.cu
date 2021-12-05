/*
CUDA-BASED IMPLEMENTATION OF THE KMP ALGORITHM

THIS VERSION CAN READ THE INPUT TEXT FROM A FILE.
THE CURRENT APPROACH IS TO WORK LINE BY LINE.
AN ALTERNATIVE ONE COULD BE TO CONCATENATE ALL THE LINES
AND THEN WORK AS BEFORE ON THE RESULTING SINGLE BIG STRING.
TODO: FIGURE OUT WHICH IS THE BEST APPROACH
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX 50

__managed__ int match = 0; // to know whether at least a match in the whole file was found

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

  for (j = 0, k = start; k < N && (k - j < stop); ++j, ++k) {
    //printf ("Inside for with j = %d and k = %d\n", j, k);

    while (j >= 0 && text[k] != pattern[j]) {
      //printf ("Inside while with j = %d and next[j] = %d\n", j, next[j]);
      j = next[j];
    }

    if (j == M - 1) { // a match was found ('M - 1' because j hasn't been incremented yet)
      match = 1;
      matchedText[k - M + 1] = k;
      j = next[j - 1]; // resetting j
    }
  }
}

void kmp(char *text, char *pattern, int N, int M, int line) {
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

  for (i = 0; i < N; ++i)
    if (matchedText[i] != -1)
      printf("Match found on line %d from position %d through %d\n", line, i + 1, matchedText[i] + 1);
}

int main(int argc, char *argv[]) {
  FILE *fp;
  char buffer[MAX+1], *text, *pattern;
  int N, M, line;

  if (argc < 3) {
    printf("You must provide the name of the text file as the first argument and the pattern as the second one\n");
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
    printf("No match was found.\n");

  return 0;
}