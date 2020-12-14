#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

static char model1[] = "../src/traced_resnet_model.pt";

int main(int argc, const char* argv[]) {

	// Context Network
	torch::jit::script::Module module1;
	module1 = torch::jit::load(model1);

	std::cout << model1 << " loaded\n";

	std::vector<torch::jit::IValue> input1;
	input1.push_back(torch::rand({1, 3, 224, 224}));
	auto output1 = module1.forward(input1);
	std::cout << output1 << std::endl;

}
