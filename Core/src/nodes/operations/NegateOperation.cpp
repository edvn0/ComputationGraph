#include "nodes/operations/NegateOperation.hpp"
#include "network/NeuralNetwork.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto NegateOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
#ifdef CG_DEBUG
	assert(consumer_outputs.size() == 1);
#endif
	value = -consumer_outputs[0];
}

std::vector<arma::mat> NegateOperation::propagate_gradient(
	const arma::mat &input)
{
	return {-input};
}

}  // namespace Core