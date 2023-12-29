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
	if (consumer_outputs[0].n_cols != consumer_outputs[1].n_cols ||
		consumer_outputs[0].n_rows != consumer_outputs[1].n_rows)
	{
		throw std::invalid_argument("Matrices cannot be added");
	}
#endif
	auto left = consumer_outputs.at(0);
	const auto &right = consumer_outputs.at(1);
	const auto &transposed = right.t();

	// Broadcast right on left row by row
	left.each_row() += transposed;

	value = left;
}
}  // namespace Core