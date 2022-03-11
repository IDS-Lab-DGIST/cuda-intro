#include <stdio.h> //C++ standard library for input and ouput, we need printf

#define N 100 //C++ define tells the compiler to replace all N to 10 before compiling

__global__ void addArrays(int *a, int *b, int *c){ //Notice __global__? We are declaring this as a kernel

  int idx = threadIdx.x; //threadIdx allows the kernel to identify what thread it is in
  c[idx] = a[idx] + b[idx];
}

void initArray(int *array, int num){ //Helper function - set every element of the array to a value

  for(int i = 0; i < N; ++i){
    array[i] = num;
  }
}

int main(){
  int *a; //Create a pointer for an int for first array
  int *b; //Create a pointer for an int for second array
  int *c; //Create a pointer for an int for result array

  size_t size = N * sizeof(int); //Calculate the memory size of an int array with length N

  //Allocating memory
  //cudaMallocManaged will allocate memory on both the CPU DRAM and GPU memory
  //cudaMallocManaged needs the memory address (&variable) and size to be allocated
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  //Initialize our arrays, let's just set every value in the array to one value for now
  initArray(a, 10);
  initArray(b, 32);
  initArray(c, 0);

  //Execute our kernel with our arrays that we have initialized
  addArrays<<<1, N>>>(a, b, c);
  cudaDeviceSynchronize(); //Wait for all threads to be executed

  //Let's check every value in the array has summed correctly
  for(int i = 0; i < N; i++){
    if(c[i] != 42){
      printf("FAIL: array[%d] - %d does not equal %d\n", i, c[i], 42);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");

  //Time to release the memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
