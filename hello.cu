#include <stdio.h> //C++ standard library for input and output, we need printf

__global__ void GPUKernel(){//Notice __global__? We are declaring this as a kernel

  printf("Hi, this is GPU.\n");
}

int main(){ //This will run on CPU

  printf("Hello, this is CPU.\n");

  GPUKernel<<<1, 1>>>(); //Execute kernel on GPU
  cudaDeviceSynchronize(); //Synchronize GPU threads
}
