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

	auto get_units() const -> std::uint32_t override
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

class OptimizerNode : public Node
{
  public:
	~OptimizerNode() override = default;
	explicit OptimizerNode(const Ref<Node> &inputs) : Node({}), node(inputs)
	{
	}

	auto get_units() const -> std::uint32_t override
	{
		return 0;
	}
	auto get_weight_statistics() const -> WeightStatistics override
	{
		return {};
	}

	auto get_root() -> auto &
	{
		return node;
	}

  private:
	Ref<Node> node;
};

template <class OperationType>
auto make_operation(const std::vector<Ref<Node>> &inputs) -> Ref<Node>
{
	return std::make_shared<OperationType>(inputs);
}

template <class OperationType> auto make_operation(Ref<Node> &left) -> Ref<Node>
{
	return std::make_shared<OperationType>(std::vector{left});
}

template <class OperationType, typename... Args>
auto make_operation(Ref<Node> &left, Args &&...args) -> Ref<Node>
{
	return std::make_shared<OperationType>(std::vector{left},
										   std::forward<Args>(args)...);
}

template <class OperationType>
auto make_operation(Ref<Node> &left, Ref<Node> &right) -> Ref<Node>
{
	return std::make_shared<OperationType>(std::vector{left, right});
}

}  // namespace Core