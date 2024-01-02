#include "nodes/operations/ReduceSumOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto ReduceSumOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	assert(consumer_outputs.size() == 1);
	const auto &copied = consumer_outputs.at(0);
	if (axis == 0)
	{
		value = arma::sum(copied, 0);  // Returns a row vector
	}
	else if (axis == 1)
	{
		value = arma::sum(copied, 1);  // Returns a column vector
	}
	else
	{
		double total_sum = arma::accu(copied);
		value = arma::mat(1, 1).fill(
			total_sum);	 // Returns a 1x1 matrix with the total sum
	}
}

std::vector<arma::mat> ReduceSumOperation::propagate_gradient(const arma::mat &)
{
	fmt::print("ReduceSumOperation Gradients\n");

	return {arma::mat(1, 1)};
}

}  // namespace Core