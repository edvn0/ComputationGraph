#include "nodes/operations/MatrixMultiplyOperation.hpp"

#include "network/NeuralNetwork.hpp"
#include "nodes/Node.hpp"
#include <utility>

namespace Core
{

auto MatrixMultiplyOperation::forward(
	const std::vector<arma::mat> &consumer_outputs) -> void
{
#ifdef CG_DEBUG
	assert(consumer_outputs.size() == 2);
	if (consumer_outputs[0].n_cols != consumer_outputs[1].n_rows)
	{
		throw std::runtime_error("Matrices cannot be multiplied");
	}
#endif
	const auto &left = consumer_outputs.at(0);
	const auto &right = consumer_outputs.at(1);

	value = left * right;
}

}  // namespace Core