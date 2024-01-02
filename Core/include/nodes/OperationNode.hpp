#pragma once

#include "nodes/Node.hpp"

namespace Core
{

class OperationNode : public Node
{
  public:
	~OperationNode() override = default;
	explicit OperationNode(const std::vector<Ref<Node>> &inputs) : Node(inputs)
	{
	}

	auto get_units() const -> u32 override
	{
		return 0;
	}
	auto get_weight_statistics() const -> WeightStatistics override
	{
		return {};
	}

	static auto get_consumer_outputs(const OperationNode &)
		-> std::vector<arma::mat>;
};

template <class OperationType>
auto make_operation(const std::vector<Ref<Node>> &inputs) -> Ref<Node>
{
	auto output = std::make_shared<OperationType>(inputs);
	output->add_consumers();
	return output;
}

template <class OperationType> auto make_operation(Ref<Node> &left) -> Ref<Node>
{
	auto output = std::make_shared<OperationType>(std::vector{left});
	output->add_consumers();
	return output;
}

template <class OperationType, typename... Args>
auto make_operation(Ref<Node> &left, Args &&...args) -> Ref<Node>
{
	auto output = std::make_shared<OperationType>(std::vector{left},
												  std::forward<Args>(args)...);
	output->add_consumers();
	return output;
}

template <class OperationType>
auto make_operation(Ref<Node> &left, Ref<Node> &right) -> Ref<Node>
{
	auto output = std::make_shared<OperationType>(std::vector{left, right});
	output->add_consumers();
	return output;
}

}  // namespace Core