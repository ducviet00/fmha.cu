#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace fmha_cu
{
__device__ __forceinline__ float warpReduceMax(float val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val)
{
    static __shared__ float shared[WARP_SIZE]; // Shared mem for 32 partial sums
    int                     lane = threadIdx.x % WARP_SIZE;
    int                     wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceMax(val); // Final reduce within first warp

    return val;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ half warpReduceSum(half val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = __hadd(val, __shfl_xor_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    static __shared__ float shared[WARP_SIZE]; // Shared mem for 32 partial sums
    int                     lane = threadIdx.x % WARP_SIZE;
    int                     wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val); // Final reduce within first warp

    return val;
}

__device__ void softmax_kernel(float* x, int size)
{
    const int thread_id  = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ float max_val;
    __shared__ float d_total;

    float m_partial = -FLT_MAX;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        m_partial = fmaxf(m_partial, x[elem_id]);

    m_partial = blockReduceMax(m_partial);

    if (thread_id == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
    {
        x[elem_id] = __expf(x[elem_id] - max_val);
        d_partial += x[elem_id];
    }

    d_partial = blockReduceSum(d_partial);
    if (thread_id == 0)
        d_total = d_partial;
    __syncthreads();

    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        x[elem_id] /= d_total;
}

template <typename scalar_t>
__global__ void multipass_fmha_kernel(scalar_t* Q,
                                    scalar_t* K,
                                    scalar_t* V,
                                    scalar_t* O,
                                    int       batch_size,
                                    int       n_heads,
                                    int       seq_len,
                                    int       head_dim,
                                    float     sm_scale)
{
    int stride_b = n_heads * seq_len * head_dim;
    int stride_h = seq_len * head_dim;
    int tx       = threadIdx.x;
    int s        = blockIdx.x;
    int h        = blockIdx.y;
    int b        = blockIdx.z;

    // printf("h: %d, s: %d\n", h, s);

    if (s >= seq_len)
        return;

    // get the query vector for this head
    scalar_t* _Q = Q + b * stride_b + h * stride_h;
    scalar_t* _K = K + b * stride_b + h * stride_h;
    scalar_t* _V = V + b * stride_b + h * stride_h;
    scalar_t* _O = O + b * stride_b + h * stride_h;

    // attention scores for this head
    extern __shared__ float att[];

    scalar_t* q = _Q + s * head_dim;
    for (int t = tx; t < seq_len; t += blockDim.x)
    {
        // get the key vector for this head and at this timestep
        scalar_t* k = _K + t * head_dim;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++)
        {
            score += float(q[i] * k[i]);
        }
        // save the score to the attention buffer
        att[t] = score * sm_scale;
    }

    __syncthreads();

    softmax_kernel(att, seq_len);

    __syncthreads();

    scalar_t* o = _O + s * head_dim;
    for (int i = tx; i < head_dim; i += blockDim.x)
    {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
        {
            // get the value vector for this head and at this timestep
            scalar_t* v = _V + t * head_dim;
            // get the attention weight for this timestep
            val += att[t] * float(v[i]);
        }
        o[i] = scalar_t(val);
    }
}


template <typename scalar_t>
__global__ void twopass_fmha_kernel(scalar_t* Q,
                                    scalar_t* K,
                                    scalar_t* V,
                                    scalar_t* O,
                                    int       batch_size,
                                    int       n_heads,
                                    int       seq_len,
                                    int       head_dim,
                                    float     sm_scale)
{
    int stride_b = n_heads * seq_len * head_dim;
    int stride_h = seq_len * head_dim;
    int tx       = threadIdx.x;
    int s        = blockIdx.x;
    int h        = blockIdx.y;
    int b        = blockIdx.z;

    // printf("h: %d, s: %d\n", h, s);

    if (s >= seq_len)
        return;

    // get the query vector for this head
    scalar_t* _Q = Q + b * stride_b + h * stride_h;
    scalar_t* _K = K + b * stride_b + h * stride_h;
    scalar_t* _V = V + b * stride_b + h * stride_h;
    scalar_t* _O = O + b * stride_b + h * stride_h;

    // attention scores for this head
    extern __shared__ float att[];

    __shared__ float max_val;
    __shared__ float inv_d_total;

    float m_max_partial = -FLT_MAX;

    scalar_t* q = _Q + s * head_dim;
    for (int t = tx; t < seq_len; t += blockDim.x)
    {
        // get the key vector for this head and at this timestep
        scalar_t* k = _K + t * head_dim;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++)
        {
            score += float(q[i] * k[i]);
        }
        // save the score to the attention buffer
        att[t] = score * sm_scale;
        m_max_partial = fmaxf(m_max_partial, att[t]);
        
    }

    if (tx == 0)
        max_val = m_max_partial;
    __syncthreads();

    double d_partial = 0.0f;
    for (int t = tx; t < seq_len; t += blockDim.x)
    {
        att[t] = __expf(att[t] - max_val);
        d_partial += att[t];
    }

    d_partial = blockReduceSum(d_partial);
    if (tx == 0)
        inv_d_total = 1 / d_partial;
    __syncthreads();

    scalar_t* o = _O + s * head_dim;
    for (int i = tx; i < head_dim; i += blockDim.x)
    {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
        {
            // get the value vector for this head and at this timestep
            scalar_t* v = _V + t * head_dim;
            // get the attention weight for this timestep
            val += att[t] * float(v[i]) * inv_d_total;
        }
        o[i] = scalar_t(val);
    }
}


at::Tensor cuda_multipass_fmha_fp16(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const int64_t op_code)
{
    at::Tensor O = torch::empty(Q.sizes(), Q.options());

    int   batch_size = Q.size(0);
    int   n_heads    = Q.size(1);
    int   seq_len    = Q.size(2);
    int   head_dim   = Q.size(3);
    float sm_scale   = 1.0 / sqrtf(head_dim);

    dim3 grid_dim(seq_len, n_heads, batch_size);
    dim3 block_dim(BLOCK_SIZE * 8);


    if (op_code == 1)
    {
        const long long sram_size = seq_len * sizeof(float) + 2 * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            Q.scalar_type(),
            "multihead_attention_fp16_kernel",
            (
                [&]
                {
                    twopass_fmha_kernel<scalar_t>
                        <<<grid_dim, block_dim, sram_size, at::cuda::getCurrentCUDAStream()>>>(
                            Q.data_ptr<scalar_t>(),
                            K.data_ptr<scalar_t>(),
                            V.data_ptr<scalar_t>(),
                            O.data_ptr<scalar_t>(),
                            batch_size,
                            n_heads,
                            seq_len,
                            head_dim,
                            sm_scale);
                }
            )                                        
        );
    }
    else 
    {
        const long long sram_size = seq_len * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            Q.scalar_type(),
            "multihead_attention_fp16_kernel",
            (
                [&]
                {
                    multipass_fmha_kernel<scalar_t>
                        <<<grid_dim, block_dim, sram_size, at::cuda::getCurrentCUDAStream()>>>(
                            Q.data_ptr<scalar_t>(),
                            K.data_ptr<scalar_t>(),
                            V.data_ptr<scalar_t>(),
                            O.data_ptr<scalar_t>(),
                            batch_size,
                            n_heads,
                            seq_len,
                            head_dim,
                            sm_scale);
                }
            )                                        
        );
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return O;
}

// Registers CUDA implementations for cpu_fmha_fp32
TORCH_LIBRARY_IMPL(fmha_cu, CUDA, m)
{
    m.impl("fmha", &cuda_multipass_fmha_fp16);
}

} // namespace fmha_cu
