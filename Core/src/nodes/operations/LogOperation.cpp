#include "nodes/operations/LogOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto LogOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	assert(consumer_outputs.size() == 1);
	const auto &copied = consumer_outputs.at(0);
	value = arma::log(copied);
}

}  // namespace Core