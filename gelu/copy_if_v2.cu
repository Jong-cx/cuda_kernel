#include<bits/stdc++.h>












//copy if dst > 0
//block 层面
template<int blockSize>
__global__ void  filter_block_k(int* dst, int* nres, int* src, int N){
    __shared__ int mem[blockSize];
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    int step = blockDim.x * gridDim.x;

    __shared__ int index;
    if(idx == 0){
        index = 0;
    }
    __syncthreads();
    int index_i = -1;
    int index_o = -1;
    // if(gidx == 0){printf("now step is %d\n",step);}
    for(int i =0;i < N; i+= step){
        int x = src[gidx];
        mem[idx] = x;
        if(x>0 && i<N){
            index_i =  atomicAdd(&index,1);
        }
        __syncthreads();
        // 使用线程0将数据加载到全局计数器上
        if(idx == 0){
            index_o = atomicAdd(nres,index);
        }
        __syncthreads();
        // 当前线程偏执为线程原子加法加法前的值，即nres没有使用加法前的数值
        int d;
        if(index_i>= 0 && index_o>=0){
            d = index_i + index_i;
            dst[d] = mem[idx];
        }
        __syncthreads();
        // if(gidx == 0){printf("now index_i is : %d,index_o is %d\n",index_i,index_o);}
    }
}
//直接写在显存上


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
    filter_block_k<blockSize><<<Grid, Block>>>(dst, nres, src, N);
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
