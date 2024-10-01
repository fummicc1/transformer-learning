#include <torch/torch.h>
#include <iostream>

class MultiHeadAttention : public torch::nn::Module
{
public:
  MultiHeadAttention(int64_t d_model, int64_t num_heads)
      : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads)
  {
    TORCH_CHECK(d_model % num_heads == 0, "d_model must be divisible by num_heads");

    W_q = register_parameter("W_q", torch::nn::Linear(d_model, d_model)->weight);
    W_k = register_parameter("W_k", torch::nn::Linear(d_model, d_model)->weight);
    W_v = register_parameter("W_v", torch::nn::Linear(d_model, d_model)->weight);
    W_o = register_parameter("W_o", torch::nn::Linear(d_model, d_model)->weight);
  }

  torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
  {
    auto batch_size = Q.size(0);

    Q = torch::matmul(Q, W_q).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);
    K = torch::matmul(K, W_k).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);
    V = torch::matmul(V, W_v).view({batch_size, -1, num_heads, d_k}).transpose(1, 2);

    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);
    auto attn = torch::softmax(scores, -1);
    auto context = torch::matmul(attn, V);

    context = context.transpose(1, 2).contiguous().view({batch_size, -1, d_model});
    return torch::matmul(context, W_o);
  }

private:
  int64_t d_model, num_heads, d_k;
  torch::Tensor W_q, W_k, W_v, W_o;
};

class PositionWiseFeedForward : public torch::nn::Module
{
public:
  PositionWiseFeedForward(int64_t d_model, int64_t d_ff)
      : linear1(d_model, d_ff), linear2(d_ff, d_model)
  {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return linear2(torch::relu(linear1(x)));
  }

private:
  torch::nn::Linear linear1, linear2;
};

class EncoderLayer : public torch::nn::Module
{
public:
  EncoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, double dropout_rate)
      : self_attn(std::make_shared<MultiHeadAttention>(d_model, num_heads)),
        feed_forward(std::make_shared<PositionWiseFeedForward>(d_model, d_ff)),
        norm1(torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))),
        norm2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))),
        dropout_rate(dropout_rate)
  {
    register_module("self_attn", self_attn);
    register_module("feed_forward", feed_forward);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto attn_output = self_attn->forward(x, x, x);
    auto x1 = norm1(x + torch::dropout(attn_output, dropout_rate, is_training()));
    auto ff_output = feed_forward->forward(x1);
    return norm2(x1 + torch::dropout(ff_output, dropout_rate, is_training()));
  }

private:
  std::shared_ptr<MultiHeadAttention> self_attn;
  std::shared_ptr<PositionWiseFeedForward> feed_forward;
  torch::nn::LayerNorm norm1;
  torch::nn::LayerNorm norm2;
  double dropout_rate;
};

int main()
{
  int64_t d_model = 512;
  int64_t num_heads = 8;
  int64_t d_ff = 2048;
  double dropout_rate = 0.1;

  auto encoder_layer = std::make_shared<EncoderLayer>(512, 8, 2048, 0.1);

  auto input = torch::rand({32, 10, d_model}); // (batch_size, seq_len, d_model)
  auto output = encoder_layer->forward(input);

  std::cout << "Input shape: " << input.sizes() << std::endl;
  std::cout << "Output shape: " << output.sizes() << std::endl;
}
