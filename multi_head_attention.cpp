#include "multi_head_attention.h"
#include <torch/torch.h>

MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t num_heads)
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads)
{
    Weight_q = register_parameter("Weight_q", torch::nn::Linear(d_model, d_model)->weight);
    Weight_k = register_parameter("Weight_k", torch::nn::Linear(d_model, d_model)->weight);
    Weight_v = register_parameter("Weight_v", torch::nn::Linear(d_model, d_model)->weight);
    Weight_o = register_parameter("Weight_o", torch::nn::Linear(d_model, d_model)->weight);
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor Query, torch::Tensor Key, torch::Tensor Value)
{
    auto batch_size = Query.size(0);

    // Queryの入力から出力を取得
    auto Q = torch::matmul(Query, Weight_q).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);
    // Keyの入力から出力を取得
    auto K = torch::matmul(Key, Weight_k).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);
    // Valueの入力から出力を取得
    auto V = torch::matmul(Value, Weight_v).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);

    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);
    auto attn = torch::softmax(scores, -1);
    auto context = torch::matmul(attn, V);
}