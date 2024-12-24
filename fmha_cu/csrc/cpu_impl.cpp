#include <torch/extension.h>

namespace fmha_cu
{

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

void cpu_multihead_attention_fp32(
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
                for (int i = 0; i < head_dim; i++)
                {
                    o[i] = 0.0f;
                    // accumulate the weighted value into xb
                    for (int tv = 0; tv < seq_len; tv++)
                    {
                        float* v = _V + tv * head_dim;
                        o[i] += _att[tv] * v[i];
                    }
                }
            }
        }
    }
}

at::Tensor cpu_fmha_fp32(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const int64_t op_code)
{
    at::Tensor O = torch::empty(Q.sizes(), Q.options());
    cpu_multihead_attention_fp32(Q.data_ptr<float>(),
                                 K.data_ptr<float>(),
                                 V.data_ptr<float>(),
                                 O.data_ptr<float>(),
                                 Q.size(0),
                                 Q.size(1),
                                 Q.size(2),
                                 Q.size(3));
    return O;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(fmha_cu, m)
{
    m.def("fmha(Tensor Q, Tensor K, Tensor V, int op_code) -> Tensor");
}

// Registers CUDA implementations for cpu_fmha_fp32
TORCH_LIBRARY_IMPL(fmha_cu, CPU, m)
{
    m.impl("fmha", &cpu_fmha_fp32);
}

} // namespace fmha_cu
