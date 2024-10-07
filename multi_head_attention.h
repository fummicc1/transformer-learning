#pragma once
#include <torch/torch.h>

class MultiHeadAttention : public torch::nn::Module
{
public:
    MultiHeadAttention(
        // int64_tは符号付き64ビット整数
        int64_t d_model,
        int64_t num_heads);
    torch::Tensor forward(torch::Tensor Query, torch::Tensor Key, torch::Tensor Value);

private:
    int64_t d_model, num_heads, d_k;
    torch::Tensor Weight_q, Weight_k, Weight_v, Weight_o;
};