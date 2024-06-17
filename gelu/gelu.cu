#include<bits/stdc++.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>


template<typename T,int Size>
struct alignas(sizeof(T)*Size) AlignedVector{
    T val[Size];

    //重载[]直接访问元素
    __host__ __device__ inline const T& operator[](int i)const{return val[i];}
    __host__ __device__ inline T& operator[](int i){return val[i];}
};

//gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template<typename T>
struct GeluFunctor{
    static T alpha = static_cast<T>(0.7978845608028654);
    static T beta = static_cast<T>(0.044714998453855515);

    const T T_half = static_cast<T>(0.5);
    const T T_one = static_cast<T>(1);
    __device__ T operator()(T x) {
        return  x * T_half * (T_one + tanh(alpha * (x + beta * x * x * x)));
    } 
};

template<>
struct GeluFunctor<half>{
    //static_cast<float>
    static constexpr float alpha_val = 0.7978845608028654f;
    static constexpr float beta_val = 0.044714998453855515f;


    const float T_half = 0.5F;
    const float T_one = 1.0F;
    
    //gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
    __device__ void vector_cac(__half *y,const __half *x) const{
        const half2 x2 = *(reinterpret_cast<const half2*>(x));
        float2 f2 = __half22float2(__hmul2(__hadd2(__hmul2(__hmul2(__hmul2(x2,x2),x2),
        __float2half2_rn(beta_val)),x2),__float2half2_rn(alpha_val))
        );
        float2 out;
        out.x = tanhf(f2.x);
        out.y = tanhf(f2.y);
        __half2 out_y = __hmul2(__hmul2(__hadd2(__float22half2_rn(out),__float2half2_rn(T_one)),__float2half2_rn(T_half)),x2);
        *reinterpret_cast<half2*>(y) = out_y;
    }

};






template<int vectorSize>
__global__ void FP16GeluCUDAKernel(__half* x,__half *y,int n){
    // 向量化载入数据
    // int offset = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * vectorSize;
    // int stride = static_cast<int>(threadIdx.x) * vectorSize;
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) * vectorSize;
    int stride = (blockDim.x * gridDim.x) * vectorSize;
    using ArrT = AlignedVector<__half,vectorSize>;
    GeluFunctor<half> gelu;
    for(int i = offset; i < n; i += stride){
        for(int j = 0;j < vectorSize; j += 2){
            gelu.vector_cac(y + offset, x + offset + j);
        }
    }
    __syncthreads();
    if(offset == 4){
        printf("the result is %f\n",__half2float(y[4]));
    }
}




int main(){
    int n = 1000;
    __half *x = new __half[n];
    __half *y = new __half[n];

    for(int i = 0;i < n;i++){
        x[i] = (__half)(i);
    }

    __half *d_x,*d_y;
    cudaMalloc((void **)&d_x,sizeof(__half)*n);
    cudaMalloc((void **)&d_y,sizeof(__half)*n);
    cudaMemcpy(d_x,x,sizeof(__half)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,sizeof(__half)*n,cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    auto is_aligned = [](const void* p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);     

    if(n%8 == 0 && is_aligned(x,kAlignment) && is_aligned(y,kAlignment)){
        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
        int block = std::min<int>((n+thread-1)/thread,deviceProp.maxGridSize[0]);
        FP16GeluCUDAKernel<2><<<block,thread>>>(d_x,d_y,n);
        cudaMemcpy(y,d_y,sizeof(__half)*n,cudaMemcpyDeviceToHost);
    }

    for(int i = 0;i <20;i += 2 ){
        // std::cout<<"the result is "<<__half2float(y[i])<<" ";
        // x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
        float s = (float)(i);
        float result_c = s / 2 * (1 + tanhf(0.7978845608028654 * (s + 0.044714998453855515 * s * s * s)));
        std::cout<<"result_c is : "<<result_c<<std::endl;
    }
    delete[] x;
    x = nullptr;
    delete[] y;
    y = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
}


