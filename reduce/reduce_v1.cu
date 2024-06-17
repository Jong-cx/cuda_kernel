#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

//使用sharememory做数据交换




template<int BlockSize>
__global__ void reduce_v0(int* d_a,int* d_out,int* sum_i){
    __shared__ int smem[BlockSize];
    int tid  = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    smem[tid] = d_a[gtid];
    //同步时间
    __syncthreads();

    //找到向量里的最大值
    for(int index = 1;index < blockDim.x;index *= 2){ 
        //使用位运算代替取模操作，这样可以有效简化代码
        //还可以使用
        if(tid & (2*index -1 )==0){
            smem[tid] += smem[tid+index];
        }
        __syncthreads();
    }
    //统一写入时，一定要注意这里的时间。
    
    //这里注意tid是单个block内部的索引，这里会有多个block的单个线程进入，进行赋值
    //gtid是全局的索引，如果更改后之后留下一个数值
    if(tid == 0){
        //不能使用一个值被多个线程访问，此时会产生block冲突，或者是写冲突
        // *sum_i +=1;
        d_out[blockIdx.x] = smem[0];
    }
}


bool checkResult(int* d_out, int Gridsize,int groudtruth){
    long long sum = 0;
    for(int i = 0;i<Gridsize;i++){
        // if(d_out[i] == 256){
            // std::cout<<"array right: "<<sum<<std::endl;
            sum +=d_out[i];
        // }    
    }
    std::cout<<"the result is "<<sum<<std::endl;
    if(sum != groudtruth) return false;
    else return true;
}


int main(){
    float millsec = 0;
    //设备属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    //设置设备属性以获取设备最大grid
    cudaGetDeviceProperties(&deviceProp, 0);
    const int N = 25600000;
    //设置block大小，block最大为1024
    const int BlockSize = 256;
    //使用最大化的block解决问题,grid最大size是2147483647
    int Gridsize = std::min(((N+255)/256),deviceProp.maxGridSize[0]);
    // std::cout<<Gridsize<<std::endl;
    // int Gridsize = 2;
    int *a = (int*)malloc(N*sizeof(int));
    int *d_a;
    cudaMalloc((void**)&d_a,N*sizeof(int));


    int *out = (int*)malloc(Gridsize*sizeof(int));
    int *d_out;
    cudaMalloc((void**)&d_out,(Gridsize * sizeof(int)));


    int* sum_hi = (int*)malloc(sizeof(int));
    int* sum_di;
    cudaMalloc((void**)&sum_di,sizeof(int));

    for(int i = 0;i<N;i++){
        a[i] = 1;
    }
    
    int groudtruth = N * 1;
    // for(int i = 0;i<256;i++)
    //     std::cout<<a[i]<<" ";
    // std::cout<<std::endl;
    cudaMemcpy(d_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
    dim3 Grid(Gridsize);
    dim3 Block(BlockSize);
    std::cout<<Grid.x<<std::endl;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<BlockSize><<<Grid,Block>>>(d_a,d_out,sum_di);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millsec,start,stop);

    cudaMemcpy(sum_hi,sum_di,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(out,d_out,(Gridsize * sizeof(int)),cudaMemcpyDeviceToHost);
    //对比CPU和GPU的运算结果
    std::cout<<"sum_hi is"<<*sum_hi<<std::endl;
    bool check = checkResult(out,Gridsize,groudtruth);
    // std::cout<<"the result is "<<"";
    // std::cout<<out<<std::endl;
    if(check) printf("result is right\n");
    else{
        printf("groudtruth is : %i\n",groudtruth);
    }
    printf("reduce_baseline latency = %f ms\n", millsec);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}


