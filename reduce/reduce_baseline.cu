#include <bits/stdc++.h>
//设备管理与底层访问例如访问访存
#include<cuda.h>
//错误属性查询和设备属性查询
#include <cuda_runtime.h>

#define N 25600000

//使用单线程累加程序
__global__ void reduce_baseline(const int* input,int* output,size_t n){
    int sum = 0;
    for(int i =0;i<n;i++){
        sum += input[i];
    }
    *output = sum;
}

bool checkResult(int* d_out,int groudtruth){
    if(*d_out != groudtruth) return false;
    else return true;
}


//配置基础设置
int main(){
    float millsec = 0;
    //设备属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    //设置block大小
    const int BlockSize = 1;
    int Gridsize = 1;
    int *a = (int*)malloc(N*sizeof(int));
    int *d_a;
    cudaMalloc((void**)&d_a,N*sizeof(int));


    int *out = (int*)malloc(sizeof(int));
    int *d_out;
    cudaMalloc((void**)&d_out,sizeof(int));

    for(int i = 0;i<N;i++){
        a[i] = i;
    }
    int groudtruth = N * 1;
    cudaMemcpy(d_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
    dim3 Grid(Gridsize);
    dim3 Block(BlockSize);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<Grid,Block>>>(d_a,d_out,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millsec,start,stop);

    cudaMemcpy(out,d_out,sizeof(int),cudaMemcpyDeviceToHost);
    //对比CPU和GPU的运算结果
    bool check = checkResult(out,groudtruth);
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





