#pragma once
#include <torch/extension.h>
#include <tuple>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExtractTopKCUDA(
    torch::Tensor geom, torch::Tensor binning, torch::Tensor image,
    int points, int rendered, int height, int width, int k);
torch::Tensor FeatureFieldsCUDA(torch::Tensor rgb, torch::Tensor alpha,
    torch::Tensor depth, torch::Tensor normal, torch::Tensor ids,
    torch::Tensor weights, float foreground);
