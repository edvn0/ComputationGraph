#include "nodes/operations/AdditionOperation.hpp"
#include "network/NeuralNetwork.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto AdditionOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
#ifdef CG_DEBUG
	assert(consumer_outputs.size() == 2);
#endif
	auto left = consumer_outputs.at(0);
	const auto &right = consumer_outputs.at(1);
	const auto &transposed = right.t();

	// Broadcast right on left row by row
	left.each_row() += transposed;

	value = left;
}

std::vector<arma::mat> AdditionOperation::propagate_gradient(const arma::mat &)
{
	return {arma::mat(1, 1), arma::mat(1, 1)};
}

}  // namespace Core