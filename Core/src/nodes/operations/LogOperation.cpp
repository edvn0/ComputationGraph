#include "nodes/operations/LogOperation.hpp"
#include "nodes/Node.hpp"
#include "nodes/ValueNode.hpp"

namespace Core
{

auto LogOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	assert(consumer_outputs.size() == 1);
	const auto &copied = consumer_outputs.at(0);
	value = arma::log(copied);
}

std::vector<arma::mat> LogOperation::propagate_gradient(const arma::mat &input)
{
	const auto first_input = ValueNode::extract_matrix_unsafe(inputs.at(0));

	const auto constructed =
		arma::repmat(input, first_input.n_rows, first_input.n_cols);

	fmt::print("LogOperation Gradients");

	return {constructed / first_input};
}

}  // namespace Core