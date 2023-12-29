#include "nodes/operations/SoftmaxOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto SoftmaxOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	arma::mat input = consumer_outputs.at(0);

	// Ensure numerical stability by subtracting the max value from each row
	arma::rowvec maxValues = arma::max(input, 0);
	input.each_row() -= maxValues;

	// Calculate the softmax
	arma::mat expInput = arma::exp(input);
	arma::rowvec sumExp = arma::sum(expInput, 0);
	value = expInput.each_row() / sumExp;
}

}  // namespace Core