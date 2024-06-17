#include <bits/stdc++.h>
#include<cuda.h>
#include<cuda_runtime.h>


template <int blockSize>
__device__ float WarpShuffle(float sum) {
    if(blockSize>=32)sum += __shfl_down_sync(0xffffffff, sum, 16); 
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    // 测试是否正常计算结果
    // if((blockIdx.x * blockSize + threadIdx.x)==0)
    // printf("%d sum is %f\n",0,sum);
    return sum;
}




template<int blockSize,int warpSize>
__global__ void reduce_get(float* data,float* result){
    //使用share memory
    float sum;
    int idx = threadIdx.x;
    int gidx = blockIdx.x * blockSize + threadIdx.x;
    sum = data[gidx];
    __syncthreads();
    //测试输入数据
    // if(gidx%256000==0)
    // printf("%d sum is %f\n",gidx/256000,sum);

    //使用warp层面的代码解决
    const int thread_idx = idx % warpSize;
    const int warp_idx = idx / warpSize;
    __shared__ float mem[blockSize/warpSize];
    //以warp作为基本单位计算所有block中的所有warp值
    //将warp中的值存入share mem
    sum = WarpShuffle<blockSize>(sum);
    mem[warp_idx] = sum;
    __syncthreads();
    //测试warp结果：
    // if(idx%32==0)
    // printf("%d sum is %f\n",gidx/256000,sum);

    //结果存入一个warp求和
    //只使用线程一做计算
    sum = (idx < blockSize / warpSize) ? mem[warp_idx] : 0.0;
    // if(gidx<16)  
    //     printf("gidx %d sum in one block is : %f",gidx,mem[idx]);
    sum = WarpShuffle<blockSize>(sum);

    if(idx == 0){
        result[blockIdx.x] = sum;
        __syncthreads();
        if(result[blockIdx.x] - 256.0 <= 1e-6 && result[blockIdx.x] -256 >= 1e-6)
        printf("index %d's result is wrong : %f\n",blockIdx.x,mem[idx]);
    }
}

template<int blockSize>
__global__ void reduce_get_final(float* data,float* result_final,int num){
    __shared__ float mem[blockSize];
    int idx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    
    float x;
    for(int i = idx ;i < num ; i += step){
        x += data[gidx];
        // __syncthreads();
    }
    __syncthreads();
    //测试上面的数据是否正确
    // if(gidx == 0 || gidx == 150){
    //     printf("the thread cucalation is : %f\n", x);
    // }
    // if(gidx == 160 || gidx == 200){
    //     printf("the thread cucalation is : %f\n", x);
    // }
    mem[idx] = x;
    __syncthreads();
    int index;
    for(index = blockSize/2; index > 32; index >>= 1){
        mem[idx] += mem[idx + index];
        __syncthreads();
    }

    mem[idx] += mem[idx + 32]; __syncwarp();
    mem[idx] += mem[idx + 16]; __syncwarp();
    mem[idx] += mem[idx + 8]; __syncwarp();
    mem[idx] += mem[idx + 4]; __syncwarp();
    mem[idx] += mem[idx + 2]; __syncwarp();
    mem[idx] += mem[idx + 1]; __syncwarp();

    if(idx == 0){
        *result_final = mem[0];
        printf("sum is %f",mem[0]);
    }
}


bool getResult(const int N,float sum){
    int i = N;
    if((float)i == sum) return true;
    else return false;
}

int main(){
    float ms;
    const int N = 256 * 100000;
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    
    const int blockSize  = 256;
    const int warpSize = 32;
    int gridSize = std::min((N+255)/256,deviceProp.maxGridSize[0]);

    //申请内存
    float* data = (float*)malloc((N * sizeof(float)));
    float* d_data;
    for(int i = 0;i<N;i++){
        data[i] = 1.0;
    }
    cudaMalloc((void**)&d_data,N*sizeof(float));
    cudaMemcpy(d_data,data,N*sizeof(float),cudaMemcpyHostToDevice);

    float* result = (float*)malloc(gridSize * sizeof(float));
    float* d_result;
    cudaMalloc((void**)&d_result,(gridSize * sizeof(float)));

    float* resultFinal = (float*)malloc(sizeof(float));
    float* d_resultFinal;
    cudaMalloc((void**)&d_resultFinal,(sizeof(float)));


    //运行kernel
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_get<blockSize,warpSize><<<gridSize,blockSize>>>(d_data,d_result);
    reduce_get_final<blockSize><<<1,blockSize>>>(d_result,d_resultFinal,gridSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(resultFinal,d_resultFinal,sizeof(float),cudaMemcpyDeviceToHost);

    bool check = getResult(N,*resultFinal);

    if(check){
        std::cout<<"the result is right \n"<<std::endl;
    }
    else{
        std::cout<<"the result is wrong "<<std::endl;
        std::cout<<"now result is "<< *resultFinal<<std::endl;
    }
    std::cout<< "time cost is :"<<ms<<std::endl;

    free(result);
    cudaFree(d_result);
    free(data);
    cudaFree(d_data);
    free(resultFinal);
    cudaFree(d_resultFinal);
}





