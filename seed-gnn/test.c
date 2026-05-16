#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("Device count: %d\n", count);
    printf("Error: %s\n", cudaGetErrorString(err));
    return 0;
}
