#include "nodes/operations/ReLUOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto ReLUOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	assert(consumer_outputs.size() == 1);
	auto copied = consumer_outputs.at(0);
	copied.transform([](double val) { return val > 0 ? val : 0.0; });

	value = copied;
}

}  // namespace Core