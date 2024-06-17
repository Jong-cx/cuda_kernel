#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK_ERROR() {                                          \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    if (deviceProp.major >= 6) {
        std::cout << "This GPU supports FP16 operations." << std::endl;

        // Add your FP16 operation code here, e.g., vectorizedReadWrite with __hadd2 and __hmul2
    } else {
        std::cerr << "This GPU does not support FP16 operations." << std::endl;
        return 1;
    }

    CUDA_CHECK_ERROR();
    return 0;
}