
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr,                                                                                            \
                    "CUDA error (%s:%d): %s:%s\n",                                                                     \
                    __FILE__,                                                                                          \
                    __LINE__,                                                                                          \
                    cudaGetErrorName(status_),                                                                         \
                    cudaGetErrorString(status_));                                                                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX_SEQ_LEN 4096
#define DIVUP(x, y) ((x + y - 1) / y)
#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

float* alloc_mat(int R, int C)
{
    float* m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(float) * R * C));
    return m;
}

half* alloc_mat_half(int R, int C)
{
    half* m;
    CHECK_CUDA(cudaMallocHost(&m, sizeof(half) * R * C));
    return m;
}

void float_to_half(const float* fm, half* hm, int size)
{
    for (int i = 0; i < size; ++i)
    {
        hm[i] = __float2half(fm[i]);
    }
}

half* float_to_half(const float* fm, int size)
{
    half* hm = alloc_mat_half(1, size);
    for (int i = 0; i < size; ++i)
    {
        hm[i] = __float2half(fm[i]);
    }
    return hm;
}

void half_to_float(const half* hm, float* fm, int size)
{
    for (int i = 0; i < size; ++i)
    {
        fm[i] = __half2float(hm[i]);
    }
}

float* half_to_float(const half* hm, int size)
{
    float* fm = alloc_mat(1, size);
    for (int i = 0; i < size; ++i)
    {
        fm[i] = __half2float(hm[i]);
    }
    return fm;
}

void rand_mat(float* m, int R, int C)
{
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }
}

void print_mat(float* m, int R, int C)
{
    for (int i = 0; i < MIN(R, 10); ++i)
    {
        for (int j = 0; j < MIN(C, 10); ++j)
        {
            printf("%+.6f ", m[i * C + j]);
        }
        printf("\n");
    }
}

void vec_diff_check(float* original, float* changed, int size, bool verbose)
{
    float  max_diff   = 0.0F;
    double total_diff = 0.0F;
    if (verbose)
        printf("size: %d\n", size);
    for (int i = 0; i < size; ++i)
    {
        if (i > size - 10 && verbose)
            printf("%d CPU: %f, GPU: %f\n", i, original[i], changed[i]);
        float diff = fabsf(original[i] - changed[i]);
        max_diff   = fmaxf(max_diff, diff);
        total_diff += diff;
    }
    printf("Comparing CPU and GPU: Max diff %f, Avg diff: %f\n", max_diff, total_diff / size);
}

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

__global__ void multihead_attention_fp16_kernel(
    half* Q, half* K, half* V, half* O, int batch_size, int n_heads, int seq_len, int head_dim, float sm_scale)
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
    half* _Q = Q + b * stride_b + h * stride_h;
    half* _K = K + b * stride_b + h * stride_h;
    half* _V = V + b * stride_b + h * stride_h;
    half* _O = O + b * stride_b + h * stride_h;

    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    // __shared__ float max_val;
    // __shared__ float inv_d_total;

    // float m_max_partial = -FLT_MAX;

    half* q = _Q + s * head_dim;
    for (int t = tx; t < seq_len; t += blockDim.x)
    {
        // get the key vector for this head and at this timestep
        half* k = _K + t * head_dim;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++)
        {
            // score += __half2float(q[i] * k[i]);
            score += (float)q[i] * (float)k[i];
        }
        // save the score to the attention buffer
        att[t] = score * sm_scale;
        // m_max_partial = fmaxf(m_max_partial, att[t]);
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    // if (tx == 0)
    //     max_val = m_max_partial;

    __syncthreads();

    softmax_kernel(att, seq_len);
    // double d_partial = 0.0f;
    // for (int t = tx; t < seq_len; t += blockDim.x)
    // {
    //     att[t] = __expf(att[t] - max_val);
    //     d_partial += att[t];
    // }

    // d_partial = blockReduceSum(d_partial);
    // if (tx == 0)
    //     inv_d_total = 1 / d_partial;

    __syncthreads();

    half* o = _O + s * head_dim;
    for (int i = tx; i < head_dim; i += blockDim.x)
    {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
        {
            // get the value vector for this head and at this timestep
            half* v = _V + t * head_dim;
            // get the attention weight for this timestep
            val += att[t] * __half2float(v[i]);
        }
        o[i] = __float2half(val);
    }
}

void multihead_attention_fp16(
    half* Q, half* K, half* V, half* O, int batch_size, int n_heads, int seq_len, int head_dim)
{
    float sm_scale = 1.0 / sqrtf(head_dim);

    dim3 grid_dim(seq_len, n_heads, batch_size);
    dim3 block_dim(BLOCK_SIZE * 8);

    multihead_attention_fp16_kernel<<<grid_dim, block_dim>>>(
        Q, K, V, O, batch_size, n_heads, seq_len, head_dim, sm_scale);
    CHECK_CUDA(cudaGetLastError());
}

void multihead_attention_fp16_gpu(
    float* Q, float* K, float* V, float* O, int batch_size, int n_heads, int seq_len, int head_dim)
{

    int qkv_size = batch_size * n_heads * seq_len * head_dim;

    half *Q_d, *K_d, *V_d, *O_d;
    CHECK_CUDA(cudaMalloc(&Q_d, sizeof(half) * qkv_size));
    CHECK_CUDA(cudaMalloc(&K_d, sizeof(half) * qkv_size));
    CHECK_CUDA(cudaMalloc(&V_d, sizeof(half) * qkv_size));
    CHECK_CUDA(cudaMalloc(&O_d, sizeof(half) * qkv_size));

    CHECK_CUDA(cudaMemcpy(Q_d, float_to_half(Q, qkv_size), sizeof(half) * qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K_d, float_to_half(K, qkv_size), sizeof(half) * qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V_d, float_to_half(V, qkv_size), sizeof(half) * qkv_size, cudaMemcpyHostToDevice));

    int num_warmup = 1;
    int num_runs   = 1;

    for (int i = 0; i < num_warmup; ++i)
    {
        multihead_attention_fp16(Q_d, K_d, V_d, O_d, batch_size, n_heads, seq_len, head_dim);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < num_runs; ++i)
    {
        multihead_attention_fp16(Q_d, K_d, V_d, O_d, batch_size, n_heads, seq_len, head_dim);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float elapsed_time = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("GPU MHA FP16 - Time Avg: %f ms\n", (elapsed_time / num_runs));

    double ops = 2. * 2. * batch_size * n_heads * seq_len * seq_len * head_dim;
    printf("GPU MHA FP16 - Performance: %.2f TFLOPS\n", (ops * num_runs) / (elapsed_time * 1e9));

    half* O_h;
    CHECK_CUDA(cudaMallocHost(&O_h, sizeof(half) * qkv_size));
    CHECK_CUDA(cudaMemcpy(O_h, O_d, sizeof(half) * qkv_size, cudaMemcpyDeviceToHost));

    half_to_float(O_h, O, qkv_size);
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(O_d));
    CHECK_CUDA(cudaFree(K_d));
    CHECK_CUDA(cudaFree(V_d));
}

void softmax_cpu(float* x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void multihead_attention_fp32_cpu(
    float* Q, float* K, float* V, float* O, int batch_size, int n_heads, int seq_len, int head_dim)
{
    // Q, K, V is (batch_size, n_heads, seq_len, head_dim)
    // O is (batch_size, n_heads, seq_len, head_dim)
    int stride_b = n_heads * seq_len * head_dim;
    int stride_h = seq_len * head_dim;

    float sm_scale = 1.0 / sqrtf(head_dim);

    for (int b = 0; b < batch_size; b++)
    {
        for (int h = 0; h < n_heads; h++)
        {
            // get the query vector for this head
            // _Q, _K is (seq_len, head_dim)
            float* _Q = Q + b * stride_b + h * stride_h;
            float* _K = K + b * stride_b + h * stride_h;
            // _V is (seq_len, head_dim)
            float* _V = V + b * stride_b + h * stride_h;
            // _O = att @ V is (seq_len, head_dim)
            float* _O = O + b * stride_b + h * stride_h;
            // pass 1: calculate query dot key
            // iterate over all timesteps, including the current one
            for (int tq = 0; tq < seq_len; tq++)
            {
                // _Q @ _K^T is (seq_len, seq_len)
                float* _att = (float*)malloc(seq_len * sizeof(float));
                float* q    = _Q + tq * head_dim;
                for (int tk = 0; tk < seq_len; tk++)
                {
                    float* k = _K + tk * head_dim;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++)
                    {
                        score += q[i] * k[i];
                    }
                    score *= sm_scale;
                    // save the score to the attention buffer
                    _att[tk] = score;
                }
                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax_cpu(_att, seq_len);

                // get the attention weight for this timestep
                float* o = _O + tq * head_dim;
                for (int tv = 0; tv < head_dim; tv++)
                {
                    o[tv] = 0.0f;
                    // accumulate the weighted value into xb
                    for (int i = 0; i < seq_len; i++)
                    {
                        float* v = _V + i * head_dim;
                        o[tv] += _att[i] * v[tv];
                    }
                }
            }
        }
    }
}

void test_attn(bool verbose)
{
    printf("Multihead attention validating...\n");

    int batch_size = 1;
    int n_heads    = 12;
    int seq_len    = 128;
    int head_dim   = 32;
    int qkv_size   = batch_size * n_heads * seq_len * head_dim;

    float* Q = alloc_mat(qkv_size, 1);
    float* K = alloc_mat(qkv_size, 1);
    float* V = alloc_mat(qkv_size, 1);

    rand_mat(Q, qkv_size, 1);
    rand_mat(K, qkv_size, 1);
    rand_mat(V, qkv_size, 1);

    Q = half_to_float(float_to_half(Q, qkv_size), qkv_size);
    K = half_to_float(float_to_half(K, qkv_size), qkv_size);
    V = half_to_float(float_to_half(V, qkv_size), qkv_size);

    float* O_gpu = alloc_mat(qkv_size, 1);
    multihead_attention_fp16_gpu(Q, K, V, O_gpu, batch_size, n_heads, seq_len, head_dim);
    printf("device fp16: \n");
    print_mat(O_gpu, 1, qkv_size);
    printf("\n");

    float* O_cpu = alloc_mat(qkv_size, 1);
    multihead_attention_fp32_cpu(Q, K, V, O_cpu, batch_size, n_heads, seq_len, head_dim);
    printf("host fp32: \n");
    print_mat(O_cpu, 1, qkv_size);
    printf("\n");

    vec_diff_check(O_cpu, O_gpu, qkv_size, verbose);
}

int main(int argc, char** argv)
{
    // Seed the random number generator with the current time
    srand(time(NULL));
    bool verbose = atoi(argv[0]);
    test_attn(verbose);
}
