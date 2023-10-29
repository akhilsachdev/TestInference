#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

int main(int argc, char *argv[]) {
  // TODO: replace this with your code

  torch::jit::script::Module module = torch::jit::load("../model.pt");
  std::cout << "load model finished!" << std::endl;
  torch::Tensor input = torch::rand({1, 3, 224, 224});
  torch::Tensor output = module.forward({input}).toTensor(); // forward
  std::cout << output << std::endl;
  std::cout << "finished fine!" << std::endl;
}
