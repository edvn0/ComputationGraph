#include "nodes/OperationNode.hpp"

namespace Core
{

auto OperationNode::get_consumer_outputs(const OperationNode &node)
	-> std::vector<arma::mat>
{
	std::vector<arma::mat> outputs;
	outputs.reserve(node.inputs.size());
	for (auto &input : node.inputs)
	{
		outputs.push_back(input->value);
	}
	return outputs;
}

}  // namespace Core