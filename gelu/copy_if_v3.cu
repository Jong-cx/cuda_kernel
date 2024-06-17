#include<bits/stdc++.h>
 
//copy if dst > 0
//warp 层面

__device__ int get_index(int* index){
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = __activemask();
    int header = __ffs(mask) - 1;
    int offset = __popc(mask);
    // if(gidx == 256){printf("now gidx's offset is %d\n",offset);}
    int lane_mask_lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
    // 如果前面有多少个线程数，刚好可以作为保存数组的索引
    unsigned int idx = __popc(lane_mask_lt & mask);

    int offset_o;
    //前面没有活跃线程了，即头线程
    if(idx == 0){
        offset_o = atomicAdd(index,offset);
    }
    __syncwarp(mask);
    offset_o = __shfl_sync(mask,offset_o,header);
    // if(gidx == 256 * 10 || gidx == 0){printf("now gidx %d's index is %d\n",gidx,offset_o);}
    return offset_o + idx;
}

__global__ void filter_warp_k(int* dst,int* nres,int* src,int N){
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    // if(gidx<16){printf("now gidx is %d\n",gidx);}
    if(gidx == 0) * nres = 0;
    __syncthreads();
    if(gidx>N)
    return;
    // if(gidx == 0){printf("now src[0] is %d\n",src[0]);}
    if(src[gidx]>0){
        int index = get_index(nres);
        // dst[index] = src[gidx];
        // if(gidx < 16){printf("now index is %d\n",index);}
    }
    
}





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
    filter_warp_k<<<Grid, Block>>>(dst, nres, src, N);
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
