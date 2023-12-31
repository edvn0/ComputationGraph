#include "nodes/operations/SoftmaxOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto SoftmaxOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
    // the typical dimensions of this BatchSize X Features
    const auto& input = consumer_outputs.at(0);

    // Ensure numerical stability by subtracting the max value from each row
    const auto maxValues = arma::max(input, 1);
    const auto replicated = arma::repmat(maxValues, 1, input.n_cols);
    const auto subtraction = input- replicated;

	// Calculate the softmax
	const auto expInput = arma::exp(subtraction);
	const auto sumExp = arma::sum(expInput, 1);
    const auto replicated_sum = arma::repmat(sumExp, 1, input.n_cols);

	value = expInput / replicated_sum;
}

std::vector<arma::mat> SoftmaxOperation::propagate_gradient(
	const arma::mat &)
{
	return {arma::mat(1,1)};
}


}  // namespace Core