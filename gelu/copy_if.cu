#include<bits/stdc++.h>












//copy if dst > 0
//block 层面
// __global__ void  filter_warp_k(int* dst, int* nres, int* src, int N){





// }
//直接写在显存上
__global__ void filter_k(int* dst, int* nres, int* src, int N){
    int idx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if(gidx == 0)
    *nres = 0;
    __syncthreads();

    //原子加法：
    if(gidx<N && src[gidx]>0){
        // int pos = atomicAdd(nres, 1);
        // dst[pos] = src[gidx];     
        dst[atomicAdd(nres, 1)] = src[gidx];                    
    }
    __syncthreads();
    // 测试单个数据index是否正确
    if(gidx == N-10)
    printf("now nres is : %d\n",*nres);
}

// __global__ void filter_k(int* dst, int* nres, int* src, int N) {
//     __shared__ int mem;
//     int idx = threadIdx.x;
//     int gidx = blockDim.x * blockIdx.x + threadIdx.x;

//     // 初始化共享内存
//     if (idx == 0) {
//         mem = 0;
//     }
//     __syncthreads();

//     // 检查src数组中的元素并进行处理
//     if (gidx < N && src[gidx] > 0) {
//         int pos = atomicAdd(&mem, 1);
//         dst[pos] = src[gidx];
//     }
//     __syncthreads();

//     // 将结果计数累加到全局内存中
//     if (idx == 0) {
//         atomicAdd(nres, mem);
//     }
// }


bool CheckResult(int *out, int groudtruth, int n){
    if (*out != groudtruth) {
        return false;
    }
    return true;
}



int main(){
    float milliseconds = 0;
    int N = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int *src_h = (int *)malloc(N * sizeof(int));
    int *dst_h = (int *)malloc(N * sizeof(int));
    int *nres_h = (int *)malloc(1 * sizeof(int));
    int *dst, *nres;
    int *src;
    cudaMalloc((void **)&src, N * sizeof(int));
    cudaMalloc((void **)&dst, N * sizeof(int));
    cudaMalloc((void **)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++){
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++){
        if (src_h[j] > 0) {
            groudtruth += 1;
        }
    }


    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    filter_k<<<Grid, Block>>>(dst, nres, src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<"n_res_h is : "<<*nres_h<<std::endl;
    std::cout<<"groudtruth is : "<<groudtruth<<std::endl;
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%d ",*nres_h);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", milliseconds);    

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}
